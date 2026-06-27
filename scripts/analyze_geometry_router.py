#!/usr/bin/env python
"""Offline analysis of geometry-router scores, no training required.

Simulates hidden states with different geometric structure (random Gaussian,
line, circle, tree-like) and prints the router's e_H / e_S estimates and the
resulting gates at several diffusion times. Useful for tuning tau/bias before
a real run. Later this can be extended to hook live ELF blocks.

Usage:
    PYTHONPATH=src python scripts/analyze_geometry_router.py
"""

import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.geometry_router import GeometryRouter  # noqa: E402


def make_hidden(kind: str, n: int = 64, dim: int = 64) -> torch.Tensor:
    """(1, n, dim) point clouds with controlled geometry."""
    g = torch.Generator().manual_seed(0)
    if kind == "gaussian":
        return torch.randn(1, n, dim, generator=g)
    if kind == "line":
        x = torch.zeros(1, n, dim)
        x[0, :, 0] = torch.arange(n, dtype=torch.float32)
        return x
    if kind == "circle":
        theta = torch.linspace(0, 2 * math.pi, n + 1)[:n]
        x = torch.zeros(1, n, dim)
        x[0, :, 0] = theta.cos()
        x[0, :, 1] = theta.sin()
        return x
    if kind == "tree":
        # Binary-tree-ish: children near parents, branches diverge — should
        # read as hyperbolic-like (low e_H).
        pts = [torch.zeros(dim)]
        for i in range(1, n):
            parent = pts[(i - 1) // 2]
            offset = torch.randn(dim, generator=g)
            pts.append(parent + offset / (1 + math.log2(i + 1)))
        return torch.stack(pts).unsqueeze(0)
    raise ValueError(kind)


def main() -> None:
    router = GeometryRouter()
    times = torch.tensor([0.05, 0.5, 0.95])
    print(f"{'hidden':10s} {'t':>5s} {'e_H':>7s} {'e_S':>7s} | {'g_E':>6s} {'g_H':>6s} {'g_S':>6s}")
    for kind in ("gaussian", "line", "circle", "tree"):
        hidden = make_hidden(kind)
        for t in times:
            gates, scores = router(hidden, t.view(1), None)
            print(f"{kind:10s} {t.item():5.2f} "
                  f"{scores['e_H'].item():7.3f} {scores['e_S'].item():7.3f} | "
                  f"{gates[0, 0].item():6.3f} {gates[0, 1].item():6.3f} {gates[0, 2].item():6.3f}")


if __name__ == "__main__":
    main()
