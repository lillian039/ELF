#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from transformers import PreTrainedTokenizerBase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.config import apply_config_overrides, load_config_from_yaml, load_sampling_configs
from torch_elf.checkpoints import load_torch_checkpoint, resolve_torch_checkpoint
from torch_elf.data import get_pad_token_id, load_jsonl_dataset
from torch_elf.device import detect_device, format_device_info
from torch_elf.encoder import T5TextEncoder
from torch_elf.model import ELF_models
from torch_elf.sampling import decode_latents, generate_latents, mask_after_eos


logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch ELF port")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--allow_random_init", action="store_true")
    return parser.parse_args()


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None, device: torch.device) -> str:
    if checkpoint_path is None:
        return "random-init"
    resolved = resolve_torch_checkpoint(checkpoint_path)
    if resolved is None:
        return "unresolved"
    payload = load_torch_checkpoint(resolved, map_location=device)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict, strict=False)
    return resolved


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    device_info = detect_device(args.device)
    logger.info(format_device_info(device_info))

    encoder = T5TextEncoder.from_pretrained(model_name=config.encoder_model_name, tokenizer_name=config.tokenizer_name or config.encoder_model_name, latent_mean=config.latent_mean, latent_std=config.latent_std, device=device_info.device)
    tokenizer: PreTrainedTokenizerBase = encoder.tokenizer
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    model = ELF_models[config.model](text_encoder_dim=encoder.d_model, max_length=config.max_length, attn_drop=config.attn_dropout, proj_drop=config.proj_dropout, num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens, vocab_size=vocab_size, num_model_mode_tokens=config.num_model_mode_tokens, bottleneck_dim=config.bottleneck_dim).to(device_info.device)
    checkpoint_status = maybe_load_checkpoint(model, args.checkpoint_path, device_info.device)
    logger.info("checkpoint_status=%s", checkpoint_status)
    if checkpoint_status == "unresolved" and not args.allow_random_init:
        raise RuntimeError("No PyTorch checkpoint could be resolved from --checkpoint_path. Use the converter first or pass --allow_random_init for a smoke test.")

    model.eval()
    sampling_config = config.sampling_configs[0]
    cfg_scale = sampling_config.cfgs[0] if getattr(sampling_config, "cfgs", None) else 1.0
    self_cond_cfg_scale = sampling_config.self_cond_cfg_scales[0] if getattr(sampling_config, "self_cond_cfg_scales", None) else 1.0

    cond_seq = None
    cond_seq_mask = None
    if config.eval_data_path and config.eval_data_path.endswith(".jsonl"):
        dataset = load_jsonl_dataset(config.eval_data_path, tokenizer)
        sample_inputs = dataset[: args.num_samples]
        pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
        input_ids = []
        for item in sample_inputs:
            tokens = item["condition_input_ids"][: (config.max_input_length or len(item["condition_input_ids"]))]
            tokens = tokens[: config.max_length]
            tokens = tokens + [pad_token_id] * max(0, config.max_length - len(tokens))
            input_ids.append(tokens)
        input_ids_tensor = torch.tensor(input_ids, device=device_info.device, dtype=torch.long)
        attention_mask = (input_ids_tensor != pad_token_id).long()
        cond_seq = encoder.encode(input_ids=input_ids_tensor, attention_mask=attention_mask)
        cond_seq_mask = attention_mask.float()

    latents = generate_latents(model=model, batch_size=args.num_samples, seq_len=config.max_length, d_model=encoder.d_model, config=config, sampling_config=sampling_config, device=device_info.device, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)
    predicted_ids = decode_latents(model, latents, self_cond_cfg_scale=self_cond_cfg_scale)
    eos_token_id = int(getattr(tokenizer, "eos_token_id", 1) or 1)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in predicted_ids]

    output_path = args.output_path or os.path.join(config.output_dir, "torch_eval_samples.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(texts):
            f.write(json.dumps({"id": idx, "generated": text}, ensure_ascii=False) + "\n")
    logger.info("Saved %s samples to %s", len(texts), output_path)
    for idx, text in enumerate(texts[: min(3, len(texts))]):
        logger.info("sample[%s]=%r", idx, text)


if __name__ == "__main__":
    main()
