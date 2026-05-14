#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from torch_elf.checkpoints import save_torch_checkpoint


def flatten_tree(tree, prefix=""):
    items = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            subprefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_tree(value, subprefix))
    else:
        items[prefix] = tree
    return items


def main():
    parser = argparse.ArgumentParser(description="Convert an exported JAX/Flax ELF tree into an inspectable PyTorch payload")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        payload = pickle.load(f)

    flat = flatten_tree(payload)
    summary = {k: tuple(getattr(v, "shape", ())) for k, v in flat.items()}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_torch_checkpoint(args.output, {"raw_jax_tree": payload, "summary": summary})
    summary_path = f"{args.output}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved inspectable payload to {args.output}")
    print(f"Saved shape summary to {summary_path}")


if __name__ == "__main__":
    main()
