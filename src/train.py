#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
import logging
import os
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
    attach_lr_scheduler, unwrap_model,
)
from utils.router_metrics import write_geometry_router_metrics
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig
from modules.geometry_router import (
    geometry_model_kwargs, validate_geometry_router_config, set_gate_warmup_alpha,
)
from modules.model import ELF_models
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step import train_step

try:
    import wandb
except ImportError:
    wandb = None

# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


def _init_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def run_training(config, *, force_cpu: bool = False):
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device(f"cuda:{local_rank}")
    rank = _rank()
    world = _world_size()

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max sequence length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"HF Repo ID: {config.hf_repo_id}")
    log_for_0(f"Batch size per device: {config.batch_size}")
    log_for_0(f"Number of epochs: {config.epochs}")
    log_for_0(f"PyTorch device: {device}, world_size={world}")
    log_for_0(f"BF16 autocast: {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"Gradient checkpointing: {bool(getattr(config, 'gradient_checkpointing', True))}")
    log_for_0("=" * 60)

    if config.use_wandb and rank == 0 and wandb is not None:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
        )
        resume_suffix = f" (resume={config.wandb_resume}, id={config.wandb_run_name})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Per-rank seed so stochastic draws (decoder/denoiser branch coin,
    # timesteps, noise) diverge across ranks. A shared seed would make every
    # rank take the same branch in lockstep, producing spiky decoder gradients
    # instead of an evenly-mixed CE/L2 reduction.
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)

    # TF32 for fp32 matmuls on Ampere/Hopper (no hyperparameter change).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    log_for_0(f"Creating {config.model} model...")
    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    geometry_router_enabled = bool(getattr(config, "geometry_router_enabled", False))
    gate_warmup_steps = int(getattr(config, "geometry_router_gate_warmup_steps", 0))
    validate_geometry_router_config(config)  # rejects reserved/unimplemented options
    if geometry_router_enabled:
        log_for_0(
            "Geometry router: enabled | "
            f"layers={getattr(config, 'geometry_router_layers', 'all')}, "
            f"denoiser_only={bool(getattr(config, 'geometry_router_denoiser_only', False))}, "
            f"learnable_bias={bool(getattr(config, 'geometry_router_learnable_bias', False))}, "
            f"gate_warmup_steps={gate_warmup_steps}, "
            f"hyperbolic_score={getattr(config, 'geometry_hyperbolic_score', 'busemann_proxy')}, "
            f"sphere_score={getattr(config, 'geometry_sphere_score', 'cosine')}, "
            f"sphere_k={getattr(config, 'geometry_router_sphere_k', '0.25,0.5,1.0,2.0,4.0')}, "
            f"sample_size={getattr(config, 'geometry_router_sample_size', 32)}, "
            f"quad_samples={getattr(config, 'geometry_router_quad_samples', 512)}"
        )
    else:
        log_for_0("Geometry router: disabled")
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
        **geometry_model_kwargs(config),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_for_0(f"ELF parameters: {total_params:,}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(f"Total trainable parameters: {total_trainable:,}")

    # Keep initialization identical across ranks, then make runtime stochastic
    # ops (e.g. dropout) rank-specific.
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // world
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * world
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    # Gradient accumulation: LR schedule is parameterized in optimizer steps
    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    # Effective learning rate (scaled with effective batch size, including grad accum)
    if config.lr is None or config.lr <= 0:
        if config.lr is not None:
            log_for_0(f"Configured lr={config.lr} is non-positive; recomputing from blr={config.blr}")
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"World={world} | batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={num_train_steps}, "
        f"warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )
    if grad_accum_steps > 1:
        log_for_0(
            f"Grad accum={grad_accum_steps}, effective batch={total_batch_size * grad_accum_steps}, "
            f"optimizer steps={num_optimizer_steps}"
        )

    lr_fn = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(model, config, lr=config.lr, grad_accum_steps=grad_accum_steps)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)

    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_params1=TrainState.init_ema(model),
        step=0, epoch=0, dropout_generator=g,
    )

    # Auto-resume: if no explicit resume path, check output_dir for existing checkpoints
    if not config.resume:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    resume_epoch_fractional = 0.0  # Fractional epoch for save-point tracking
    if config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            state, resume_step = load_checkpoint(ckpt_path, state)
            resume_epoch_fractional = float(state.epoch)
            start_epoch = int(state.epoch)
            log_for_0(f"Resumed from step {resume_step} (epoch {resume_epoch_fractional:.2f})")
        except Exception as e:
            log_for_0(f"Error loading checkpoint: {e}")
            log_for_0("Continuing training from scratch")

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if device.type == "cuda":
        log_for_0("Compiling ELF model with torch.compile (first step will be slower)...")
        state = state.replace(model=torch.compile(state.model))

    if world > 1:
        # find_unused_parameters=False is safe: 0-mult sinks in train_step
        # (`0 * net_out.sum()` for CE, `0 * decoder_logits.sum()` for L2)
        # keep every head in the autograd graph on every step.
        state = state.replace(model=DDP(
            state.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        ))

    os.makedirs(config.output_dir, exist_ok=True)

    if rank == 0:
        config_dict = {
            k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in vars(config).items()
        }
        config_path = os.path.join(config.output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=(world > 1),
    )

    log_for_0("\n" + "=" * 60)
    log_for_0("Checkpoint and Evaluation Schedule")
    log_for_0("=" * 60)
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s), eval every {config.eval_freq} epoch(s)"
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")

    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    if resume_step > 0:
        global_step = resume_step
        # Skip already-processed batches within the current epoch on resume
        steps_to_skip_in_epoch = resume_step - start_epoch * steps_per_epoch
    else:
        global_step = start_epoch * steps_per_epoch
        steps_to_skip_in_epoch = 0
    state.step = global_step

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()

    # Track last save point for fractional save_freq; use fractional epoch from
    # checkpoint to avoid re-saving immediately after resume.
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)
    stop_training = False
    max_train_steps = getattr(config, "max_train_steps", None)

    end_epoch = config.epochs
    if max_train_steps is not None and int(max_train_steps) > global_step:
        extra_epochs = (int(max_train_steps) - global_step + steps_per_epoch - 1) // steps_per_epoch
        end_epoch = max(end_epoch, start_epoch + max(1, extra_epochs))

    for epoch in range(start_epoch, end_epoch):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if world > 1 and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=rank != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            is_first_step = step_in_epoch == 0 and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue
            batch = prepare_batch(batch, config, generator=g)
            # Gate warmup schedule: linearly decay alpha 1->0 over the first
            # gate_warmup_steps, forcing the geometry branches to train early.
            if geometry_router_enabled and gate_warmup_steps > 0:
                alpha = max(0.0, 1.0 - global_step / gate_warmup_steps)
                set_gate_warmup_alpha(unwrap_model(state.model), alpha)
            state, metrics = train_step(state, encoder=encoder, batch=batch, config=config)

            # Sync only on first step to measure torch.compile time;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log_for_0("First training step (torch.compile + execution) completed...")

            global_step += 1
            train_metrics.append(metrics)
            epoch_pbar.update(1)

            if global_step % config.log_freq == 0:
                stacked = torch.stack([
                    torch.stack([m["loss"] for m in train_metrics]).mean(),
                    torch.stack([m["l2_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["ce_loss"] for m in train_metrics]).mean(),
                ])
                # Average each metric across DDP ranks before logging — done
                # once per log_freq so we never sync on every train step.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                    stacked = stacked / dist.get_world_size()
                avg_loss, avg_l2, avg_ce = (float(x) for x in stacked.tolist())
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = state.optimizer.param_groups[0]["lr"]

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if rank == 0:
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2:.4f}, ce={avg_ce:.4f}, "
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb and wandb is not None:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb.log({
                                "train_loss": avg_loss, "train_l2_loss": avg_l2,
                                "train_ce_loss": avg_ce, "lr": current_lr,
                                "epoch": current_epoch_progress, "step": global_step,
                            }, step=global_step)
                        except Exception:
                            pass

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            # Intra-epoch checkpoint saving (fractional save_freq, e.g., 0.1 epoch)
            if 0 < config.save_freq < 1:
                progress = epoch + (global_step - epoch * steps_per_epoch) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                    log_for_0(f"Saved checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

            if max_train_steps is not None and global_step >= int(max_train_steps):
                log_for_0(f"Reached max_train_steps={max_train_steps}; stopping training loop")
                stop_training = True
                break

        epoch_pbar.close()
        if stop_training:
            break
        current_epoch = epoch + 1
        state.epoch = current_epoch

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                local_batch_size=local_batch_size,
            )
            last_log_step = global_step
            last_log_time = time.time()

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Generation")
    log_for_0("=" * 60)
    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if bool(getattr(config, "geometry_router_log_metrics", False)):
        metrics_path = getattr(config, "geometry_router_metrics_path", None)
        if not metrics_path:
            metrics_path = os.path.join(config.output_dir, "geometry_router_metrics_train.json")
        write_geometry_router_metrics(
            state.model, metrics_path,
            extra={"step": int(global_step), "epoch": float(state.epoch)},
        )
    if config.use_wandb and rank == 0 and wandb is not None:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config, force_cpu=args.use_cpu)


if __name__ == "__main__":
    main()
