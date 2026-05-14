from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5EncoderModel


@dataclass
class T5TextEncoder:
    model: Any
    tokenizer: PreTrainedTokenizerBase
    latent_mean: float
    latent_std: float

    @classmethod
    def from_pretrained(cls, model_name: str, tokenizer_name: str | None = None, latent_mean: float = 0.0, latent_std: float = 1.0, device: torch.device | None = None) -> "T5TextEncoder":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        model = T5EncoderModel.from_pretrained(model_name)
        model.eval()
        if device is not None:
            model = cast(Any, model).to(device)
        return cls(model=model, tokenizer=tokenizer, latent_mean=latent_mean, latent_std=latent_std)

    @property
    def d_model(self) -> int:
        return int(self.model.config.d_model)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        latents = outputs.last_hidden_state
        return (latents - self.latent_mean) / self.latent_std
