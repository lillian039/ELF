from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear(layer: nn.Linear, zero: bool = False, normal_std: Optional[float] = None) -> nn.Linear:
    if zero:
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer
    if normal_std is not None:
        nn.init.normal_(layer.weight, std=normal_std)
    else:
        nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class TextRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, pt_seq_len: int = 512, ft_seq_len: Optional[int] = None, theta: float = 10000.0, num_empty_token: int = 0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

    def _freqs(self, total_len: int, device: torch.device, dtype: torch.dtype):
        main_len = max(total_len - self.num_empty_token, 0)
        ft_seq_len = self.ft_seq_len or max(main_len, self.pt_seq_len)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)[: self.dim // 2] / self.dim))
        pos = torch.arange(main_len, device=device, dtype=torch.float32) / max(ft_seq_len, 1) * self.pt_seq_len
        freqs_main = torch.einsum("n,d->nd", pos, freqs).repeat_interleave(2, dim=-1)
        d = freqs_main.shape[-1] if main_len > 0 else self.dim
        cos_parts, sin_parts = [], []
        if self.num_empty_token > 0:
            cos_parts.append(torch.ones((self.num_empty_token, d), device=device, dtype=torch.float32))
            sin_parts.append(torch.zeros((self.num_empty_token, d), device=device, dtype=torch.float32))
        if main_len > 0:
            cos_parts.append(torch.cos(freqs_main))
            sin_parts.append(torch.sin(freqs_main))
        cos = torch.cat(cos_parts, dim=0) if len(cos_parts) > 1 else cos_parts[0]
        sin = torch.cat(sin_parts, dim=0) if len(sin_parts) > 1 else sin_parts[0]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        cos, sin = self._freqs(seq_len, t.device, t.dtype)
        while cos.ndim < t.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = init_linear(nn.Linear(text_encoder_dim, bottleneck_dim, bias=False))
        self.proj2 = init_linear(nn.Linear(bottleneck_dim, hidden_size, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = init_linear(nn.Linear(frequency_embedding_size, hidden_size), normal_std=0.02)
        self.mlp_2 = init_linear(nn.Linear(hidden_size, hidden_size), normal_std=0.02)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half, 1))
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))
        return self.mlp_2(F.silu(t_emb))


def _expand_attention_mask(attn_mask: torch.Tensor, num_heads: int, target_len: int) -> torch.Tensor:
    if attn_mask.ndim == 2:
        mask = attn_mask[:, None, None, :]
    elif attn_mask.ndim == 3:
        mask = attn_mask[:, None, :, :]
    else:
        mask = attn_mask
    mask = mask.to(dtype=torch.bool)
    if mask.shape[-2] == 1 and target_len != 1:
        mask = mask.expand(mask.shape[0], mask.shape[1], target_len, mask.shape[-1])
    if mask.shape[1] == 1 and num_heads != 1:
        mask = mask.expand(mask.shape[0], num_heads, mask.shape[-2], mask.shape[-1])
    return mask


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scale_factor = 1.0 / math.sqrt(query.shape[-1])
    attn_weight = torch.einsum("bhld,bhsd->bhls", query.float(), key.float()) * scale_factor
    if attn_mask is not None:
        mask = _expand_attention_mask(attn_mask, query.shape[1], query.shape[-2])
        attn_weight = attn_weight.masked_fill(~mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = F.softmax(attn_weight, dim=-1)
    return torch.einsum("bhls,bhsd->bhld", attn_weight.to(value.dtype), value)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = init_linear(nn.Linear(dim, dim * 3, bias=qkv_bias))
        self.proj = init_linear(nn.Linear(dim, dim))
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, rope_fn: Optional[TextRotaryEmbeddingFast], attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        head_dim = dim // self.num_heads
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        x = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, dim)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = init_linear(nn.Linear(dim, 2 * hidden_dim, bias=bias))
        self.w3 = init_linear(nn.Linear(hidden_dim, dim, bias=bias))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.drop(F.silu(x1) * x2)
        return self.w3(hidden)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = init_linear(nn.Linear(hidden_size, patch_size * patch_size * out_channels), zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))
