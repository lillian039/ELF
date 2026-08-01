#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
import copy
import yaml
import logging
import os
import sys
import time
from functools import partial


def _configure_distributed_runtime():
    """Make the training script compatible with torchrun-style launchers.

    torchrun exports WORLD_SIZE/RANK/LOCAL_RANK/LOCAL_WORLD_SIZE/MASTER_ADDR/MASTER_PORT.
    We map those into the environment variables that JAX distributed initialization expects.
    """
    if "WORLD_SIZE" in os.environ:
        os.environ.setdefault("JAX_NUM_PROCESSES", os.environ["WORLD_SIZE"])
    if "RANK" in os.environ:
        os.environ.setdefault("JAX_PROCESS_ID", os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:
        os.environ.setdefault("JAX_LOCAL_PROCESS_ID", os.environ["LOCAL_RANK"])
    if "LOCAL_WORLD_SIZE" in os.environ:
        os.environ.setdefault("JAX_LOCAL_PROCESS_COUNT", os.environ["LOCAL_WORLD_SIZE"])
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        os.environ.setdefault(
            "JAX_COORDINATOR_ADDRESS",
            f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        )


_configure_distributed_runtime()

# Bind each torchrun process to its assigned GPU when available.
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ and "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", "0")

# Initialize JAX distributed BEFORE importing other JAX modules.
# When torchrun is used, prefer the explicit process information it exports
# instead of letting JAX auto-detect distributed settings and probe TPU metadata.
import jax


def _initialize_jax_distributed():
    # This environment is a local single-host GPU machine. JAX's distributed
    # initialization attempts cloud TPU metadata probing and fails here, so we
    # skip it entirely and rely on the single-process/local-device behavior.
    return


_initialize_jax_distributed()

# Ensure repo root on sys.path so imports work when run as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import get_metrics, shard
from tqdm import tqdm
from transformers import AutoTokenizer
try:
    import wandb
except Exception:  # wandb is optional; only required when config.use_wandb is True
    wandb = None

from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_encoder_checkpoint, load_checkpoint,
    find_latest_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
)
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig
from modules.model import ELF_models
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step import train_step


# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
for _name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
    logging.getLogger(_name).setLevel(logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (JAX).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    return parser.parse_args()


# ============================================
# Main Training Loop
# ============================================
def run_training(config):

    # Print configuration
    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (JAX/Flax)")
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
    log_for_0(f"JAX devices: {jax.device_count()}")
    log_for_0(f"JAX backend: {jax.default_backend()}")
    log_for_0(f"Distributed process index/count: {jax.process_index()}/{jax.process_count()}")
    log_for_0("=" * 60)

    # Initialize wandb
    if config.use_wandb and jax.process_index() == 0:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
            settings=wandb.Settings(start_method="thread"),
        )
        resume_suffix = f" (resume={config.wandb_resume}, id={config.wandb_run_name})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    rng = jax.random.PRNGKey(config.seed)

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    if (getattr(config, "decorrelation_weight", 0.0) > 0
            or getattr(config, "manifold_cf_weight", 0.0) > 0):
        from utils.data_utils import add_lexicon_labels
        log_for_0("decorrelation/cf regularizer active: precomputing lexicon sent/gender labels.")
        train_dataset = add_lexicon_labels(train_dataset, tokenizer)

    # ============================================
    # Load frozen encoder
    # ============================================
    # Always instantiate the model to get d_model (architecture) and model.apply (static arg).
    log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)

    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    log_for_0("Encoder weights loaded.")
    encoder_params = jax_utils.replicate(encoder_params)
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    # ============================================
    # Create ELF Model
    # ============================================
    log_for_0(f"Creating {config.model} model...")
    model_fn = ELF_models[config.model]
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Dummy inputs for initialization (2x dim if self_cond_prob > 0 to init self_cond_proj layer)
    max_length = config.max_length
    input_dim = 2 * encoder_config.d_model if config.self_cond_prob > 0 else encoder_config.d_model
    dummy_x = jnp.ones((1, max_length, input_dim))
    dummy_t = jnp.ones((1,))
    dummy_self_cond_cfg_scale = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
    log_for_0(f"Dummy shapes: x={dummy_x.shape}, t={dummy_t.shape}")

    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    model = model_fn(
        text_encoder_dim=encoder_config.d_model, max_length=max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        num_phi_tokens=config.num_phi_tokens if config.semantic_factorization else 0,
        manifold_dim=config.manifold_dim if config.semantic_factorization else 0,
        bottleneck_dim=config.bottleneck_dim,
    )

    log_for_0("Initializing ELF model...")
    dummy_phi = (
        jnp.ones((1, encoder_config.d_model)) if config.semantic_factorization else None
    )
    init_args = dict(
        x=dummy_x, t=dummy_t, deterministic=True,
        self_cond_cfg_scale=dummy_self_cond_cfg_scale, phi=dummy_phi,
    )
    elf_params = model.init(init_rng, **init_args)
    log_for_0("\n" + model.tabulate(init_rng, **init_args))
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(elf_params))
    log_for_0(f"ELF parameters: {total_params:,}")

    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // num_hosts
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * num_devices
        local_batch_size = config.batch_size * num_local_devices
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
        f"Hosts={num_hosts}, local_devices={num_local_devices}, total_devices={num_devices} | "
        f"batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={num_train_steps}, warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )
    if grad_accum_steps > 1:
        log_for_0(
            f"Grad accum={grad_accum_steps}, effective batch={total_batch_size * grad_accum_steps}, "
            f"optimizer steps={num_optimizer_steps}"
        )

    lr_schedule = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(config, lr_schedule, grad_accum_steps=grad_accum_steps)
    state = TrainState.create(
        apply_fn=model.apply, params=elf_params["params"], tx=optimizer,
        dropout_rng=dropout_rng, ema_params1=copy.deepcopy(elf_params["params"]),
    )
    total_trainable = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log_for_0(f"Total trainable parameters: {total_trainable:,}")

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

    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(
        partial(train_step, encoder_apply_fn=encoder_model.apply, config=config),
        axis_name="batch", donate_argnums=(0,),
    )

    os.makedirs(config.output_dir, exist_ok=True)

    config_dict = {
        k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
        for k, v in vars(config).items()
    }
    config_path = os.path.join(config.output_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    log_for_0(f"Config saved to {config_path}")

    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
    )

    # ============================================
    # Checkpoint and Evaluation Schedule (Epoch-based)
    # ============================================
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

    # ============================================
    # Training Loop
    # ============================================
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

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()

    # Track last save point for fractional save_freq; use fractional epoch from
    # checkpoint to avoid re-saving immediately after resume.
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            jax.clear_caches()

        train_dataloader.sampler.set_epoch(epoch)
        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=jax.process_index() != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            is_first_step = step_in_epoch == 0 and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue
            rng, batch_rng = jax.random.split(rng, 2)
            batch = prepare_batch(batch, config, rng=batch_rng)
            batch = {k: v for k, v in batch.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
            batch = shard(batch)
            state, metrics = p_train_step(state, encoder_params, batch=batch)

            # Sync only on first step to measure XLA compilation time;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
                log_for_0("First training step (XLA compilation + execution) completed...")

            train_metrics.append(metrics)
            global_step += 1
            epoch_pbar.update(1)

            if global_step % config.log_freq == 0:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), state.params)
                gathered = get_metrics(train_metrics)
                avg_loss = float(jnp.mean(gathered["loss"]))
                avg_l2_loss = float(jnp.mean(gathered["l2_loss"]))
                avg_ce_loss = float(jnp.mean(gathered["ce_loss"]))
                avg_cyc_loss = float(jnp.mean(gathered["cyc_loss"])) if "cyc_loss" in gathered else 0.0
                avg_ib_loss = float(jnp.mean(gathered["ib_loss"])) if "ib_loss" in gathered else 0.0
                avg_dec_loss = float(jnp.mean(gathered["dec_loss"])) if "dec_loss" in gathered else 0.0
                avg_cf_loss = float(jnp.mean(gathered["cf_loss"])) if "cf_loss" in gathered else 0.0
                avg_cf_on = float(jnp.mean(gathered["cf_on"])) if "cf_on" in gathered else 0.0
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = lr_schedule((global_step - 1) // grad_accum_steps)

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2_loss:.4f}", "ce": f"{avg_ce_loss:.4f}",
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if jax.process_index() == 0:
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2_loss:.4f}, ce={avg_ce_loss:.4f}, "
                        f"cyc={avg_cyc_loss:.4f}, ib={avg_ib_loss:.4f}, dec={avg_dec_loss:.4f}, "
                        f"cf={avg_cf_loss:.4f}, cf_on={avg_cf_on:.4f}, "
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb.log({
                                "train_loss": avg_loss, "train_l2_loss": avg_l2_loss,
                                "train_ce_loss": avg_ce_loss, "lr": current_lr,
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

        epoch_pbar.close()
        current_epoch = epoch + 1

        state = jax_utils.replicate(jax_utils.unreplicate(state).replace(epoch=current_epoch))

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            rng = run_generation(
                state=state, encoder_params=encoder_params, encoder_apply_fn=encoder_model.apply,
                eval_dataset=eval_dataset, tokenizer=tokenizer, config=config,
                rng=rng, local_batch_size=local_batch_size,
            )
            last_log_step = global_step
            last_log_time = time.time()

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Generation")
    log_for_0("=" * 60)
    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if config.use_wandb and jax.process_index() == 0:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config)


if __name__ == "__main__":
    main()
