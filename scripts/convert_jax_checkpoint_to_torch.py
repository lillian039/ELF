#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.config import load_config_from_yaml
from torch_elf.checkpoints import save_torch_checkpoint
from torch_elf.model import ELF_models


def flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_tree(value, next_prefix))
    else:
        items[prefix] = tree
    return items


def extract_source_tree(payload: Any, params_key: str) -> dict[str, Any]:
    tree = payload.get("raw_jax_tree", payload) if isinstance(payload, dict) else payload
    if not isinstance(tree, dict):
        raise TypeError(f"Expected dict-like payload, got {type(tree)!r}")
    if params_key in tree and isinstance(tree[params_key], dict):
        return tree[params_key]
    return tree


def infer_dims(source_tree: dict[str, Any]) -> tuple[int, int]:
    text_encoder_dim = int(np.asarray(source_tree["proj_bias"]).shape[0])
    vocab_size = int(np.asarray(source_tree["unembed_bias"]).shape[0])
    return text_encoder_dim, vocab_size


def normalize_key(source_key: str) -> str:
    exact_map = {
        "proj_kernel": "proj.weight",
        "proj_bias": "proj.bias",
        "unembed_kernel": "unembed.weight",
        "unembed_bias": "unembed.bias",
    }
    if source_key in exact_map:
        return exact_map[source_key]

    key = re.sub(r"blocks_(\d+)", r"blocks.\1", source_key)
    key = key.replace(".kernel", ".weight")
    return key


def should_transpose(source_key: str, array: np.ndarray) -> bool:
    if source_key in {"proj_kernel", "unembed_kernel"}:
        return True
    return source_key.endswith(".kernel") and array.ndim == 2


def to_torch_tensor(source_key: str, value: Any) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype.name == "bfloat16":
        array = array.astype(np.float32)
    if should_transpose(source_key, array):
        array = array.T
    return torch.from_numpy(np.ascontiguousarray(array))


def build_model_from_config(config_path: str, text_encoder_dim: int, vocab_size: int) -> torch.nn.Module:
    config = load_config_from_yaml(config_path)
    model = ELF_models[config.model](
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    return model


def convert_tree_to_state_dict(source_tree: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    flat = flatten_tree(source_tree)
    state_dict: dict[str, torch.Tensor] = {}
    summary: dict[str, dict[str, Any]] = {}
    for source_key, value in flat.items():
        if not hasattr(value, "shape"):
            continue
        target_key = normalize_key(source_key)
        tensor = to_torch_tensor(source_key, value)
        state_dict[target_key] = tensor
        summary[target_key] = {
            "source_key": source_key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return state_dict, summary


def validate_against_model(model: torch.nn.Module, converted_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    expected = model.state_dict()
    converted_keys = set(converted_state)
    expected_keys = set(expected)

    missing = sorted(expected_keys - converted_keys)
    unexpected = sorted(converted_keys - expected_keys)
    shape_mismatches: list[dict[str, Any]] = []
    for key in sorted(expected_keys & converted_keys):
        expected_shape = tuple(expected[key].shape)
        actual_shape = tuple(converted_state[key].shape)
        if expected_shape != actual_shape:
            shape_mismatches.append(
                {"key": key, "expected": list(expected_shape), "actual": list(actual_shape)}
            )
    return missing, unexpected, shape_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an exported JAX/Flax ELF tree into a loadable PyTorch checkpoint")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--params_key", type=str, default="ema_params1")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        payload = pickle.load(f)

    source_tree = extract_source_tree(payload, args.params_key)
    text_encoder_dim, vocab_size = infer_dims(source_tree)
    model = build_model_from_config(args.config, text_encoder_dim=text_encoder_dim, vocab_size=vocab_size)
    converted_state, conversion_summary = convert_tree_to_state_dict(source_tree)
    missing, unexpected, shape_mismatches = validate_against_model(model, converted_state)

    if missing or unexpected or shape_mismatches:
        problems = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
        }
        raise RuntimeError(json.dumps(problems, indent=2, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_checkpoint(
        str(output_path),
        {
            "model": converted_state,
            "source_tree_key": args.params_key,
            "text_encoder_dim": text_encoder_dim,
            "vocab_size": vocab_size,
        },
    )

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(conversion_summary, f, indent=2, ensure_ascii=False)

    print(f"Saved loadable PyTorch checkpoint to {output_path}")
    print(f"Saved conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
