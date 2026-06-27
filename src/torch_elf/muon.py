"""Muon (MomentUm Orthogonalized by Newton-schulz) optimizer for PyTorch.

Based on KellerJordan/Muon (https://github.com/KellerJordan/Muon).
Muon is used for 2D+ weight matrices; 1D parameters use AdamW fallback.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients maximize the slope at zero.
    Reference: https://github.com/KellerJordan/Muon
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(dtype=G.dtype)


class MuonWithAdamW(Optimizer):
    """Muon for 2D+ parameters, AdamW fallback for 1D parameters.

    Parameter groups with `use_muon=True` use Muon; others use AdamW.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum_beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            use_muon = group.get("use_muon", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Weight decay (decoupled, AdamW-style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if use_muon and p.ndim >= 2 and not p.is_sparse:
                    # ---- Muon path ----
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1 - momentum_beta)
                    update = buf if not nesterov else grad.lerp(buf, momentum_beta)

                    # Handle Conv4d weight [out, in, *spatial]
                    shape_original = update.shape
                    if update.ndim == 4:
                        update = update.view(update.size(0), -1)

                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update = update.view(shape_original)

                    # Scale by sqrt(max_dim / min_dim) for non-square matrices
                    if update.ndim >= 2 and update.size(-2) > 1 and update.size(-1) > 1:
                        scale = max(1, update.size(-2) / update.size(-1)) ** 0.5
                        update.mul_(scale)

                    p.add_(update, alpha=-lr)
                else:
                    # ---- AdamW fallback ----
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = betas

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
