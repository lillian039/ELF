"""Layer primitives for the ELF transformer."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Init defaults:
# - Linear weights: xavier_uniform; biases: 0
# - TimestepEmbedder MLPs and learned tokens: normal(0.02)
# - final_layer.linear: 0 (zero init)
def DEFAULT_KERNEL_INIT(weight: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weight)


def DEFAULT_BIAS_INIT(bias: torch.Tensor) -> None:
    nn.init.zeros_(bias)


def ZERO_INIT(t: torch.Tensor) -> None:
    nn.init.zeros_(t)


def NORMAL_INIT_002(t: torch.Tensor) -> None:
    nn.init.normal_(t, mean=0.0, std=0.02)


def _make_linear(in_features: int, out_features: int, bias: bool = True,
                 kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT) -> nn.Linear:
    """nn.Linear with explicit initializers."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    kernel_init(layer.weight)
    if bias and bias_init is not None:
        bias_init(layer.bias)
    return layer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class TextRotaryEmbeddingFast(nn.Module):
    """1D Rotary Position Embedding for text/sequence models."""

    def __init__(self, dim: int, pt_seq_len: int = 512,
                 ft_seq_len: Optional[int] = None, theta: float = 10000.0,
                 num_empty_token: int = 0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token
        freqs_cos, freqs_sin = self._compute_freqs()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _compute_freqs(self) -> tuple:
        dim = self.dim
        ft_seq_len = self.ft_seq_len
        pt_seq_len = self.pt_seq_len

        freqs = 1.0 / (self.theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim
        ))
        pos = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len

        freqs_main = torch.einsum('..., f -> ... f', pos, freqs)
        freqs_main = repeat(freqs_main, '... n -> ... (n r)', r=2)

        D = freqs_main.shape[-1]
        cos_parts, sin_parts = [], []
        # 1. Empty tokens (no rotation): cos=1, sin=0
        if self.num_empty_token > 0:
            cos_parts.append(torch.ones((self.num_empty_token, D), dtype=freqs.dtype))
            sin_parts.append(torch.zeros((self.num_empty_token, D), dtype=freqs.dtype))
        # 2. Main tokens (RoPE positions 0 to pt_seq_len-1)
        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))

        freqs_cos = torch.cat(cos_parts, dim=0) if len(cos_parts) > 1 else cos_parts[0]
        freqs_sin = torch.cat(sin_parts, dim=0) if len(sin_parts) > 1 else sin_parts[0]
        return freqs_cos, freqs_sin

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs_cos = self.freqs_cos.to(t.dtype)
        freqs_sin = self.freqs_sin.to(t.dtype)
        return t * freqs_cos + rotate_half(t) * freqs_sin


class RMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps).to(input_dtype)
        return self.weight.to(input_dtype) * (hidden_states * inv_std)


class BottleneckTextProj(nn.Module):
    """Text projection with bottleneck."""

    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = _make_linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = _make_linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = _make_linear(
            frequency_embedding_size, hidden_size, bias=True,
            kernel_init=NORMAL_INIT_002, bias_init=DEFAULT_BIAS_INIT,
        )
        self.mlp_2 = _make_linear(
            hidden_size, hidden_size, bias=True,
            kernel_init=NORMAL_INIT_002, bias_init=DEFAULT_BIAS_INIT,
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Sinusoidal timestep embeddings: (N,) ints -> (N, dim) floats."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))
        return self.mlp_2(F.silu(t_emb))


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    query/key/value: (B, num_heads, L|S, head_dim).
    attn_mask: optional int mask (B, S) or (B, L, S); 1=valid, 0=masked.
    Returns: (B, num_heads, L, head_dim).
    """
    bool_mask: Optional[torch.Tensor] = None
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            bool_mask = attn_mask[:, None, None, :]
        elif attn_mask.dim() == 3:
            bool_mask = attn_mask[:, None, :, :]
        else:
            bool_mask = attn_mask
        bool_mask = bool_mask.bool()
    return F.scaled_dot_product_attention(query, key, value, attn_mask=bool_mask)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        head_dim = dim // num_heads
        self.qkv = _make_linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = _make_linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module],
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        """x: (B, N, C). attention_mask: optional int mask (B, N), 1=valid, 0=padded."""
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        x = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop > 0.0:
            x = F.dropout(x, p=self.proj_drop, training=not deterministic)
        return x


# ============================================================
# Geometry-routed attention (curvature-aware operator mixing)
# ============================================================

def apply_attention_mask_to_logits(logits: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mask invalid key positions in attention logits.

    logits: (B, H, N, N). attention_mask: (B, N) or (B, N, N), 1=valid.
    Uses a large finite negative instead of -inf so a fully-masked row
    softmaxes to uniform rather than NaN.
    """
    if attention_mask is None:
        return logits
    if attention_mask.dim() == 2:
        key_mask = attention_mask[:, None, None, :]  # (B, 1, 1, N)
    elif attention_mask.dim() == 3:
        key_mask = attention_mask[:, None, :, :]  # (B, 1, N, N)
    else:
        key_mask = attention_mask
    neg = torch.tensor(-1e4, dtype=logits.dtype, device=logits.device)
    return torch.where(key_mask.bool(), logits, neg)


def softmax_attention_from_logits(logits: torch.Tensor, v: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None,
                                  dropout_p: float = 0.0,
                                  deterministic: bool = True) -> torch.Tensor:
    """Softmax attention from explicit logits. logits: (B, H, N, N), v: (B, H, N, D)."""
    probs = F.softmax(apply_attention_mask_to_logits(logits, attention_mask), dim=-1)
    if dropout_p > 0.0 and not deterministic:
        probs = F.dropout(probs, p=dropout_p, training=True)
    return probs @ v


def poincare_exp0(u: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Exponential map at the origin of the Poincare ball with curvature -c.

    u: (..., D) tangent vectors -> points inside the ball of radius 1/sqrt(c),
    clamped to (1/sqrt(c)) * (1 - eps) for numerical safety.
    """
    sqrt_c = math.sqrt(c)
    norm = u.norm(dim=-1, keepdim=True)
    # tanh(sqrt(c) * n) / (sqrt(c) * n) -> 1 as n -> 0.
    scaled = sqrt_c * norm
    coef = torch.where(scaled > eps, torch.tanh(scaled) / scaled.clamp(min=eps),
                       torch.ones_like(scaled))
    x = coef * u
    max_norm = (1.0 / sqrt_c) * (1.0 - eps)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.where(x_norm > max_norm, x * (max_norm / x_norm), x)


def busemann_proxy_score(q: torch.Tensor, k: torch.Tensor, c: float = 1.0,
                         eps: float = 1e-6) -> torch.Tensor:
    """First-order Busemann/PV proxy attention logits, (B, H, N, D) -> (B, H, N, N).

    Maps queries into the Poincare ball via exp0 and treats normalized keys as
    ideal boundary directions p_j; the logit is the (negated) Busemann-style
    function log(1 - c||q_i||^2) - log(||sqrt(c) q_i - p_j||^2). This is a
    first-version proxy of PV/Busemann attention, replaceable later — it is
    NOT claimed to be the exact PV operator.
    """
    qf, kf = q.float(), k.float()
    sqrt_c = math.sqrt(c)
    q_ball = poincare_exp0(qf, c=c, eps=eps)
    p = F.normalize(kf, dim=-1, eps=eps)  # ideal boundary directions
    q_norm2 = q_ball.pow(2).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
    # ||sqrt(c) q_i - p_j||^2 = c||q_i||^2 - 2 sqrt(c) <q_i, p_j> + 1
    dot = torch.matmul(q_ball, p.transpose(-2, -1))  # (B, H, N, N)
    denom = (c * q_norm2 - 2.0 * sqrt_c * dot + 1.0).clamp(min=eps)
    score = torch.log((1.0 - c * q_norm2).clamp(min=eps)) - torch.log(denom)
    return score.to(q.dtype)


def poincare_distance_score(q: torch.Tensor, k: torch.Tensor, c: float = 1.0,
                            eps: float = 1e-6) -> torch.Tensor:
    """Negative squared Poincare distance logits, (B, H, N, D) -> (B, H, N, N)."""
    qf, kf = q.float(), k.float()
    head_dim = q.shape[-1]
    q_ball = poincare_exp0(qf, c=c, eps=eps)
    k_ball = poincare_exp0(kf, c=c, eps=eps)
    q_norm2 = q_ball.pow(2).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
    k_norm2 = k_ball.pow(2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, H, 1, N)
    # ||q_i - k_j||^2 expanded via the gram matrix.
    diff2 = (q_norm2 + k_norm2
             - 2.0 * torch.matmul(q_ball, k_ball.transpose(-2, -1))).clamp(min=0.0)
    denom = ((1.0 - c * q_norm2) * (1.0 - c * k_norm2)).clamp(min=eps)
    acosh_arg = (1.0 + 2.0 * c * diff2 / denom).clamp(min=1.0 + eps)
    dist = torch.acosh(acosh_arg) / math.sqrt(c)
    score = -dist.pow(2) / math.sqrt(head_dim)
    return score.to(q.dtype)


def sphere_score(q: torch.Tensor, k: torch.Tensor, mode: str = "cosine",
                 eps: float = 1e-6) -> torch.Tensor:
    """Spherical/angular attention logits, (B, H, N, D) -> (B, H, N, N)."""
    head_dim = q.shape[-1]
    q_s = F.normalize(q.float(), dim=-1, eps=eps)
    k_s = F.normalize(k.float(), dim=-1, eps=eps)
    dot = torch.matmul(q_s, k_s.transpose(-2, -1)).clamp(-1.0 + eps, 1.0 - eps)
    if mode == "cosine":
        score = dot / math.sqrt(head_dim)
    elif mode == "negative_angular":
        score = -torch.acos(dot).pow(2) / math.sqrt(head_dim)
    else:
        raise ValueError(f"Unknown sphere_score mode: {mode}")
    return score.to(q.dtype)


class GeometryRoutedAttention(nn.Module):
    """Multi-head self-attention mixing Euclidean / hyperbolic / spherical operators.

    Same qkv / q_norm / k_norm / proj structure (and initialization) as
    Attention, so parameter shapes and names line up. Only the attention
    logits differ per branch; values stay in R^d in all branches, so the
    branch outputs live in the same Euclidean coordinates and can be mixed
    directly by the router gates.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 geometry_router: Optional[nn.Module] = None,
                 hyperbolic_curvature: float = 1.0,
                 hyperbolic_score: str = "busemann_proxy",
                 sphere_score: str = "cosine"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        head_dim = dim // num_heads
        self.qkv = _make_linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = _make_linear(dim, dim, bias=True)

        self.geometry_router = geometry_router
        self.hyperbolic_curvature = hyperbolic_curvature
        if hyperbolic_score not in ("busemann_proxy", "poincare_distance"):
            raise ValueError(f"Unknown hyperbolic_score: {hyperbolic_score}")
        self.hyperbolic_score = hyperbolic_score
        if sphere_score not in ("cosine", "negative_angular"):
            raise ValueError(f"Unknown sphere_score: {sphere_score}")
        self.sphere_score_mode = sphere_score

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module],
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True,
                t: Optional[torch.Tensor] = None,
                geometry_mask: Optional[torch.Tensor] = None,
                routing_active: bool = True,
                router_row_active: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, N, C). attention_mask: optional (B, N), 1=valid. t: (B,).

        routing_active=False bypasses routing for the whole call (original
        SDPA path). router_row_active: optional (B,) with 1=routed row,
        0=row forced to the pure Euclidean gate (used by denoiser-only mode
        for decoder-mode rows in the mixed training forward).
        """
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        if self.geometry_router is None or not routing_active:
            # Fallback: behave like the original Attention.
            out = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            if t is None:
                raise ValueError(
                    "GeometryRoutedAttention requires the diffusion time t; "
                    "ELFBlock should pass it through."
                )
            eps = self.geometry_router.eps
            logits_e = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if self.hyperbolic_score == "busemann_proxy":
                logits_h = busemann_proxy_score(q, k, c=self.hyperbolic_curvature, eps=eps)
            else:
                logits_h = poincare_distance_score(q, k, c=self.hyperbolic_curvature, eps=eps)
            logits_s = sphere_score(q, k, mode=self.sphere_score_mode, eps=eps)

            out_e = softmax_attention_from_logits(
                logits_e, v, attention_mask, self.attn_drop, deterministic)
            out_h = softmax_attention_from_logits(
                logits_h, v, attention_mask, self.attn_drop, deterministic)
            out_s = softmax_attention_from_logits(
                logits_s, v, attention_mask, self.attn_drop, deterministic)

            gates, _ = self.geometry_router(x, t, geometry_mask)  # (B, 3)
            if router_row_active is not None:
                # Rows flagged inactive get the pure Euclidean gate [1, 0, 0].
                euclid = torch.zeros_like(gates)
                euclid[:, 0] = 1.0
                act = router_row_active.to(gates.dtype).view(-1, 1)
                gates = act * gates + (1.0 - act) * euclid
            g = gates.to(out_e.dtype).view(B, 3, 1, 1, 1)
            out = g[:, 0] * out_e + g[:, 1] * out_h + g[:, 2] * out_s

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        if self.proj_drop > 0.0:
            out = F.dropout(out, p=self.proj_drop, training=not deterministic)
        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim_eff = int(hidden_dim * 2 / 3)
        self.drop = drop
        self.w12 = _make_linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.w3 = _make_linear(hidden_dim_eff, dim, bias=bias)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        if self.drop > 0.0:
            hidden = F.dropout(hidden, p=self.drop, training=not deterministic)
        return self.w3(hidden)


class FinalLayer(nn.Module):
    """The final layer of ELF."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        # Zero-init linear (kernel & bias both zero).
        self.linear = _make_linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))
