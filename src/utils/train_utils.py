"""Training-time utilities: train state, optimizer/schedule helpers.

Extracted from train.py for reuse and readability.
"""

import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from utils.logging_utils import log_for_0


# ============================================
# Train State with EMA
# ============================================
@dataclass
class TrainState:
    """Lightweight container for the trainable model and its EMA shadow."""

    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: Any = None
    ema_params1: Dict[str, torch.Tensor] = field(default_factory=dict)
    step: int = 0
    epoch: int = 0
    dropout_generator: Optional[torch.Generator] = None
    def replace(self, **kwargs) -> "TrainState":
        new = TrainState(
            model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            ema_params1=self.ema_params1, step=self.step, epoch=self.epoch,
            dropout_generator=self.dropout_generator,
        )
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new

    @staticmethod
    def init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.named_parameters()}


def prefetch_to_device(iterator, size: int = 2):
    """Prefetch batches asynchronously via a background thread."""
    q = queue.Queue(maxsize=size)

    def enqueue():
        for item in iterator:
            q.put(item)
        q.put(None)

    threading.Thread(target=enqueue, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


# ============================================
# Optimizer
# ============================================
def get_optimizer(model: nn.Module, config, lr: float, grad_accum_steps: int = 1):
    """Build optimizer (AdamW or Muon). Gradient clipping is applied in train.py."""
    if config.optimizer == "muon":
        from utils.muon_utils import muon_with_aux_adam
        opt = muon_with_aux_adam(model, lr=lr)
        log_for_0("Using Muon optimizer")
    elif config.optimizer == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=lr, weight_decay=config.weight_decay,
            betas=(config.adam_b1, config.adam_b2),
        )
        log_for_0("Using AdamW optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}. Choose 'adamw' or 'muon'.")
    return opt


# ============================================
# Learning Rate Schedule
# ============================================
def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    schedule: str = "constant",
    min_lr: float = 0.0,
):
    """Create learning rate schedule with linear warmup."""
    alpha = (min_lr / learning_rate) if learning_rate > 0 else 0.0

    def fn(step: int) -> float:
        step = int(step)
        if num_warmup_steps > 0 and step < num_warmup_steps:
            return learning_rate * step / max(1, num_warmup_steps)
        if schedule == "cosine":
            progress = (step - num_warmup_steps) / max(1, num_train_steps - num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return learning_rate * (alpha + (1.0 - alpha) * cosine)
        return learning_rate
    return fn


def attach_lr_scheduler(optimizer: Optimizer, lr_fn) -> LambdaLR:
    """Wrap `lr_fn` in a LambdaLR (lambda returns multiplier on base lr)."""
    base_lr = optimizer.param_groups[0]["lr"]
    return LambdaLR(optimizer, lr_lambda=lambda step: lr_fn(step) / max(base_lr, 1e-12))


# ============================================
# EMA update
# ============================================
def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP (`.module`) and `torch.compile` (`._orig_mod`) wrappers."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            break
    return model


@torch.no_grad()
def ema_update(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    """In-place EMA over trainable params: `ema = decay*ema + (1-decay)*param`."""
    inner = unwrap_model(model)
    for name, param in inner.named_parameters():
        if name in ema_state:
            ema_state[name].lerp_(param.detach(), 1.0 - decay)
