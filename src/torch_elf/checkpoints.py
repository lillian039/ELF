from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch


def save_torch_checkpoint(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def load_torch_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def resolve_torch_checkpoint(checkpoint_path: str) -> Optional[str]:
    candidate = Path(os.path.expanduser(checkpoint_path))
    if candidate.exists():
        return str(candidate)
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=checkpoint_path, repo_type="model")
    except Exception:
        return None
    for pattern in ("*.pt", "*.bin", "*.safetensors"):
        matches = list(Path(local_dir).rglob(pattern))
        if matches:
            return str(matches[0])
    return None
