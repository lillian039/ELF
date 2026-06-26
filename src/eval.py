#!/usr/bin/env python
"""Evaluation script for trained ELF models: load a checkpoint and generate text samples."""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState, get_optimizer
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id
from generation import test_generation_uncond, test_generation_cond
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model by generating text samples")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--config_override", action="append", default=[],
                        help="Override config values (field_name=value). Repeatable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --seeds is not specified)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to evaluate (e.g. '42,123,456'). Overrides --seed.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint file (e.g. outputs/elf_b-owt/checkpoint_19000) or HF repo id.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def _init_distributed():
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Eval only gathers CPU objects (text strings), so gloo is sufficient.
        dist.init_process_group(backend="gloo")


def main():
    args = parse_args()
    _init_distributed()

    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_for_0("Loading configuration...")
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")

    world = dist.get_world_size() if dist.is_initialized() else 1
    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size for evaluation: {config.global_batch_size}")
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

    log_for_0(f"Config loaded from {args.config}")
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Max length: {config.max_length}")
    log_for_0(f"Max input length: {config.max_input_length}")
    log_for_0(f"Num samples: {config.num_samples}")
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")
    log_for_0(f"BF16 autocast (sampling): {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"torch.compile (eval model): {bool(getattr(config, 'use_compile', False))}")

    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds is not None else [args.seed]
    log_for_0(f"Seeds to evaluate: {seed_list}")

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    eval_dataset = None
    if config.eval_data_path is not None:
        log_for_0("Loading dataset for conditional generation...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(
                config.eval_data_path, tokenizer,
                input_key="input", output_key="output",
            )
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval dataset size: {len(eval_dataset)}")

    # Encoder (HuggingFace T5)
    log_for_0(f"Loading Encoder: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ELF model
    log_for_0(f"Creating {config.model} model...")
    vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)

    # Train state template (only used to plumb EMA params + step/epoch).
    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_ema(model), step=0, epoch=0,
        dropout_generator=g,
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    log_for_0(f"Loading checkpoint from: {args.checkpoint_path}")
    state, _ = load_checkpoint(args.checkpoint_path, state)
    state.model = state.model.to(device).eval()

    rank = dist.get_rank() if dist.is_initialized() else 0

    for seed_idx, seed_val in enumerate(seed_list):
        if len(seed_list) > 1:
            log_for_0(f"\n{'#' * 70}")
            log_for_0(f"Seed {seed_idx + 1}/{len(seed_list)}: {seed_val}")
            log_for_0(f"{'#' * 70}")

        # Per-rank offset so ranks generate different samples when sharding;
        # rank 0 keeps the original seed for single-GPU reproducibility.
        per_rank_seed = seed_val + rank * 1_000_003
        seed_gen = torch.Generator(device="cpu").manual_seed(per_rank_seed)
        torch.manual_seed(per_rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(per_rank_seed)
        original_output_dir = config.output_dir
        if len(seed_list) > 1:
            config.output_dir = os.path.join(original_output_dir, f"seed_{seed_val}")

        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
            common_kwargs = dict(
                state=state, tokenizer=tokenizer, generator=seed_gen,
                config=config, sampling_config=sc,
                batch_size=local_batch_size, num_samples=config.num_samples,
            )
            if eval_dataset is None:
                test_generation_uncond(**common_kwargs)
            else:
                test_generation_cond(
                    **common_kwargs, encoder=encoder, dataset=eval_dataset,
                )

        config.output_dir = original_output_dir

    log_for_0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
