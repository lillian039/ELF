#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_tree(value, next_prefix))
    else:
        items[prefix] = tree
    return items


def maybe_snapshot_download(repo_id_or_path: str) -> Path:
    candidate = Path(os.path.expanduser(repo_id_or_path)).resolve()
    if candidate.exists():
        return candidate
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id_or_path, repo_type="model")
    return Path(local_dir)


def build_restore_args(metadata_tree: Any, device: Any) -> Any:
    import jax
    import orbax.checkpoint as ocp
    from jax.sharding import SingleDeviceSharding

    cpu_sharding = SingleDeviceSharding(device)

    def make_arg(_: Any) -> Any:
        return ocp.ArrayRestoreArgs(restore_type=np.ndarray, sharding=cpu_sharding)

    return jax.tree_util.tree_map(make_arg, metadata_tree)


def load_orbax_tree(checkpoint_dir: Path) -> tuple[Any, Any]:
    import jax
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    step_metadata = checkpointer.metadata(checkpoint_dir)
    metadata = step_metadata.item_metadata
    device = jax.local_devices(backend="cpu")[0]
    restore_args = build_restore_args(metadata, device)
    restored = checkpointer.restore(
        checkpoint_dir,
        args=ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args),
    )

    def to_numpy(x: Any) -> Any:
        if hasattr(x, "shape"):
            return np.asarray(x)
        return x

    numpy_tree = jax.tree_util.tree_map(to_numpy, restored)
    return numpy_tree, step_metadata


def select_checkpoint_subdir(repo_root: Path, checkpoint_subdir: str | None) -> Path:
    if checkpoint_subdir:
        candidate = repo_root / checkpoint_subdir
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint subdir not found: {candidate}")
        return candidate
    default_candidate = repo_root / "checkpoint_0"
    if default_candidate.exists():
        return default_candidate
    raise FileNotFoundError(
        f"Could not find checkpoint directory under {repo_root}. Pass --checkpoint_subdir explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an Orbax/OCDBT checkpoint to a Python-loadable pickle tree")
    parser.add_argument("--input", required=True, help="Local path or Hugging Face model repo id")
    parser.add_argument("--checkpoint_subdir", default=None, help="Checkpoint directory inside the repo (default: checkpoint_0)")
    parser.add_argument("--output", required=True, help="Output pickle path")
    parser.add_argument("--metadata_output", default=None, help="Optional output path for checkpoint metadata JSON")
    parser.add_argument("--summary_output", default=None, help="Optional output path for flattened shape summary JSON")
    args = parser.parse_args()

    repo_root = maybe_snapshot_download(args.input)
    checkpoint_dir = select_checkpoint_subdir(repo_root, args.checkpoint_subdir)
    tree, metadata = load_orbax_tree(checkpoint_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(tree, f)

    flat = flatten_tree(tree)
    summary = {k: {"shape": list(getattr(v, "shape", [])), "dtype": str(getattr(v, "dtype", type(v).__name__))} for k, v in flat.items()}

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_json = {
        "init_timestamp_nsecs": getattr(metadata, "init_timestamp_nsecs", None),
        "commit_timestamp_nsecs": getattr(metadata, "commit_timestamp_nsecs", None),
        "item_handlers": getattr(metadata, "item_handlers", None),
        "custom_metadata": getattr(metadata, "custom_metadata", None),
        "item_metadata_repr": repr(getattr(metadata, "item_metadata", None)),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2, ensure_ascii=False)

    print(f"Exported Orbax tree from {checkpoint_dir} to {output_path}")
    print(f"Saved flattened summary to {summary_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
