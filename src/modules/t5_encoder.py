#!/usr/bin/env python
"""Frozen T5 text embedder, wrapping `transformers.T5EncoderModel`."""

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

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

    def _encode_with_pairwise_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Run the T5 encoder with an ELF pairwise self-attention mask.

        ELF builds a 3D mask so condition tokens cannot attend to target tokens
        during latent encoding. Recent transformers versions expand T5 encoder
        masks as if every mask were 2D, which turns a 3D mask into an invalid
        5D tensor. This path mirrors the encoder stack forward pass but supplies
        the intended 4D additive mask directly to each T5 block.
        """
        encoder = self.model.encoder
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = encoder.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        if attention_mask.shape != (batch_size, seq_length, seq_length):
            raise ValueError(
                "Pairwise T5 attention mask must have shape "
                f"{(batch_size, seq_length, seq_length)}, got {tuple(attention_mask.shape)}."
            )

        causal_mask = attention_mask[:, None, :, :].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        cache_position = torch.arange(seq_length, device=inputs_embeds.device)

        output_attentions = bool(encoder.config.output_attentions)
        output_hidden_states = bool(encoder.config.output_hidden_states)
        return_dict = bool(encoder.config.use_return_dict)

        head_mask = encoder.get_head_mask(None, encoder.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        hidden_states = encoder.dropout(inputs_embeds)
        for i, layer_module in enumerate(encoder.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # T5Block.forward changed across transformers versions
            # (past_key_value -> past_key_values, cache_position added);
            # build kwargs from the actual signature so both APIs work.
            block_params = inspect.signature(layer_module.forward).parameters
            layer_kwargs = dict(
                layer_head_mask=head_mask[i],
                use_cache=False,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            if "past_key_values" in block_params:
                layer_kwargs["past_key_values"] = None
            elif "past_key_value" in block_params:
                layer_kwargs["past_key_value"] = None
            if "cache_position" in block_params:
                layer_kwargs["cache_position"] = cache_position
            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                **layer_kwargs,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            values = [hidden_states, None, all_hidden_states, all_attentions, None]
            return tuple(v for v in values if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=None,
        )

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
            if attention_mask is not None and attention_mask.ndim == 3:
                # Older transformers (e.g. 4.44) expand 3D masks correctly via
                # get_extended_attention_mask; prefer the stock path and only
                # fall back to the manual stack walk on versions where the 3D
                # mask is mis-expanded (raises a shape/dim error).
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                except (ValueError, RuntimeError, IndexError):
                    out = self._encode_with_pairwise_mask(input_ids=input_ids, attention_mask=attention_mask)
            else:
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
