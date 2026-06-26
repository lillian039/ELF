#!/usr/bin/env python
"""Frozen T5 text embedder, wrapping `transformers.T5EncoderModel`."""

from typing import Any, Optional

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


class T5EncoderConfig:
    """Configuration class for T5Encoder."""

    def __init__(self, model_name: str, dtype: Any):
        self.model_name = model_name
        self.dtype = dtype
        self.vocab_size: int = 0
        self.d_model: int = 0
        self.d_kv: int = 0
        self.d_ff: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.is_gated_act: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str, dtype: Any = torch.float32) -> "T5EncoderConfig":
        cfg = cls(model_name, dtype)
        defaults = {
            "t5-small": dict(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
                             num_layers=6, num_heads=8, is_gated_act=False),
            "t5-base":  dict(vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
                             num_layers=12, num_heads=12, is_gated_act=False),
            "t5-large": dict(vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
                             num_layers=24, num_heads=16, is_gated_act=False),
        }
        if model_name in defaults:
            for k, v in defaults[model_name].items():
                setattr(cfg, k, v)
        return cfg


class T5Encoder(nn.Module):
    """T5 encoder used as a frozen text embedder."""

    def __init__(self, config: T5EncoderConfig, *, pretrained: bool = True):
        super().__init__()
        from transformers import T5EncoderModel, T5Config

        if pretrained:
            self.model = T5EncoderModel.from_pretrained(config.model_name)
        else:
            hf_config = T5Config.from_pretrained(config.model_name)
            self.model = T5EncoderModel(hf_config)

        hf = self.model.config
        config.vocab_size = hf.vocab_size
        config.d_model = hf.d_model
        config.d_kv = hf.d_kv
        config.d_ff = hf.d_ff
        config.num_layers = hf.num_layers
        config.num_heads = hf.num_heads
        config.is_gated_act = bool(getattr(hf, "is_gated_act", False))
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        was_training = self.model.training
        if deterministic:
            self.model.eval()
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            if not deterministic and was_training:
                self.model.train()
        return out.last_hidden_state


def get_encoder(model_name: str, dtype: Any):
    """Return `(config, model)`. Weights are downloaded on first use."""
    log_for_0(f"Loading T5 Encoder: {model_name}...")
    config = T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
    model = T5Encoder(config, pretrained=True)
    if dtype is not None:
        model = model.to(dtype)
    return config, model
