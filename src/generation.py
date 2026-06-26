import copy
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.train_utils import unwrap_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.metrics_utils import Metrics as PPLMetrics, compute_bleu, compute_rouge
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import (
    mask_after_eos, shift_left,
    _generate_samples_single_batch, _dlm_decode_batch,
    _build_run_name,
)

try:
    import wandb
except ImportError:
    wandb = None


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _build_eval_model(state, use_compile: bool = False) -> nn.Module:
    """Return an eval-mode model copy loaded with EMA params (if available)."""
    model = unwrap_model(state.model)
    eval_model = copy.deepcopy(model)
    if state.ema_params1:
        eval_model.load_state_dict(state.ema_params1)
    eval_model.eval()
    if use_compile:
        log_for_0("Compiling eval model with torch.compile (first batch will be slower)...")
        eval_model = torch.compile(eval_model)
    return eval_model


# ============================================
# Generation Helper
# ============================================
def run_generation(
    state,
    encoder: nn.Module,
    eval_dataset,
    tokenizer,
    config,
    generator: torch.Generator,
    local_batch_size: int,
):
    """Run test generation."""
    for sc_idx, sc in enumerate(config.sampling_configs):
        if len(config.sampling_configs) > 1:
            log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
        common_kwargs = dict(
            state=state,
            tokenizer=tokenizer,
            generator=generator,
            config=config,
            sampling_config=sc,
            batch_size=local_batch_size,
            num_samples=config.num_samples,
        )
        if eval_dataset is None:
            test_generation_uncond(**common_kwargs)
        else:
            test_generation_cond(
                **common_kwargs, encoder=encoder, dataset=eval_dataset,
            )


# ============================================
# Unconditional generation
# ============================================
def test_generation_uncond(
    state,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test unconditional generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              UNCONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim
    log_for_0(f"Per-device batch size: {batch_size}")

    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    wandb_tables = {}
    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )

    world = _world()
    rank = _rank()
    param_dtype = next(model.parameters()).dtype

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        # Shard work across ranks: each rank generates ceil(num_samples/world);
        # the extras are trimmed after the gather on rank 0.
        local_num_samples = (num_samples + world - 1) // world
        local_generated = []
        generation_time = 0.0
        decode_time = 0.0
        num_batches = (local_num_samples + batch_size - 1) // batch_size
        local_processed = 0

        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=(rank != 0)):
            if local_processed >= local_num_samples:
                break
            current_batch = min(batch_size, local_num_samples - local_processed)
            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )
            if device.type == "cuda":
                z = torch.randn(
                    (current_batch, config.max_length, d_model),
                    dtype=param_dtype, device=device,
                ) * config.denoiser_noise_scale
            else:
                z = (torch.randn((current_batch, config.max_length, d_model),
                                 generator=generator, dtype=param_dtype)
                     * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=None, cond_seq_mask=None,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            generation_time += time.time() - gen_start

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            decode_time += time.time() - dec_start

            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

            for i in range(predicted_ids.shape[0]):
                if local_processed >= local_num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                local_generated.append(text)
                local_processed += 1

        # Gather shards to rank 0, then assemble final ID-tagged list.
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_generated)
            if rank == 0:
                merged = []
                for shard in gathered:
                    merged.extend(shard)
                all_generated = [(i, txt) for i, txt in enumerate(merged[:num_samples])]
            else:
                all_generated = []
        else:
            all_generated = [(i, txt) for i, txt in enumerate(local_generated[:num_samples])]

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="uncond",
        )

        out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, gen in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

        ppl_results = None
        if config.online_eval and _rank() == 0:
            log_for_0("\n" + "=" * 70)
            log_for_0("              PPL EVALUATION")
            log_for_0("=" * 70)
            ppl_metrics.reset()
            with open(out_path, "r", encoding="utf-8") as f:
                text_samples = [json.loads(line)["generated"] for line in f]
            nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
            skipped = len(text_samples) - len(nonempty_samples)
            if skipped > 0:
                log_for_0(f"PPL eval: skipped {skipped} empty samples")
            if not nonempty_samples:
                log_for_0("PPL eval: all samples empty; skipping perplexity computation")
            else:
                ppl_results = ppl_metrics.record_generative_perplexity(
                    text_samples=nonempty_samples,
                    max_length=config.eval_ppl_max_length,
                    retokenize=True,
                )
                log_for_0(f"Perplexity: {ppl_results['ppl']:.4f}")
                log_for_0(f"Mean Entropy: {ppl_results['mean_entropy']:.4f}")
            log_for_0("=" * 70 + "\n")

        if _rank() == 0:
            if ppl_results is not None:
                metrics_line = {
                    "epoch": epoch_val, "step": step_val,
                    "ppl": ppl_results["ppl"], "mean_entropy": ppl_results["mean_entropy"],
                }
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

            if config.use_wandb and wandb is not None:
                table = wandb.Table(columns=["sample_id", "text"])
                for tid, gen in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, gen)
                wandb_tables[f"generated_samples_uncond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if ppl_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/ppl": ppl_results["ppl"],
                        f"generation/{name}/mean_entropy": ppl_results["mean_entropy"],
                    })

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")
    log_for_0("=" * 70 + "\n")


# ============================================
# Conditional generation
# ============================================
def test_generation_cond(
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test conditional generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              CONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    dataloader = get_dataloader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    wandb_tables = {}
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        all_generated = []
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0

        rank = _rank()
        total_batches = (num_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Generating samples (cond)", disable=(rank != 0))
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()

            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=next(model.parameters()).dtype,
            )

            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attention_mask,
                encoder=encoder, latent_mean=encode_latent_mean, latent_std=encode_latent_std,
            ).to(next(model.parameters()).dtype)

            z = (torch.randn((bsz, config.max_length, d_model),
                             generator=generator, dtype=next(model.parameters()).dtype)
                 * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            generation_time += time.time() - gen_start

            gen_length = config.max_length - config.max_input_length
            cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :gen_length]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            decode_time += time.time() - dec_start

            original_texts = [batch["target"][i] for i in range(bsz)]
            context_texts = [batch["input"][i] for i in range(bsz)]

            for i in range(bsz):
                if samples_processed >= num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                all_generated.append((samples_processed, original_texts[i], text, context_texts[i]))
                samples_processed += 1
            pbar.update(1)
        pbar.close()

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
        )

        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, orig, gen, ctx in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

            cond_eval_results = None
            if config.online_eval and all_generated:
                hypotheses = [gen for _, _, gen, _ in all_generated]
                references = [orig for _, orig, _, _ in all_generated]
                bleu_score = compute_bleu(hypotheses, references)
                rouge_scores = compute_rouge(hypotheses, references)
                cond_eval_results = {"bleu": bleu_score, **rouge_scores}
                log_for_0(
                    f"BLEU: {bleu_score:.2f}  ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge_scores['rouge2']:.2f}  ROUGE-L: {rouge_scores['rougeL']:.2f}"
                )

            if config.use_wandb and wandb is not None:
                table = wandb.Table(columns=["sample_id", "context", "original", "generated"])
                for tid, orig, gen, ctx in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, ctx, orig, gen)
                wandb_tables[f"generated_samples_cond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if cond_eval_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/bleu": cond_eval_results["bleu"],
                        f"generation/{name}/rouge1": cond_eval_results["rouge1"],
                        f"generation/{name}/rouge2": cond_eval_results["rouge2"],
                        f"generation/{name}/rougeL": cond_eval_results["rougeL"],
                    })
            if cond_eval_results is not None:
                metrics_line = {"epoch": epoch_val, "step": step_val, **cond_eval_results}
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")
    log_for_0("=" * 70 + "\n")
