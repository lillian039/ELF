"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, GeometryRoutedAttention,
    RMSNorm, SwiGLUFFN, TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
    _make_linear,
)
from modules.geometry_router import GeometryRouter, parse_layer_spec


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 geometry_router_enabled: bool = False,
                 geometry_router: Optional[nn.Module] = None,
                 hyperbolic_curvature: float = 1.0,
                 hyperbolic_score: str = "busemann_proxy",
                 sphere_score: str = "cosine"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        if geometry_router_enabled and geometry_router is not None:
            self.attn = GeometryRoutedAttention(
                hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                attn_drop=attn_drop, proj_drop=proj_drop,
                geometry_router=geometry_router,
                hyperbolic_curvature=hyperbolic_curvature,
                hyperbolic_score=hyperbolic_score,
                sphere_score=sphere_score,
            )
        else:
            self.attn = Attention(
                hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                attn_drop=attn_drop, proj_drop=proj_drop,
            )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True,
                t: Optional[torch.Tensor] = None,
                geometry_mask: Optional[torch.Tensor] = None,
                routing_active: bool = True,
                router_row_active: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_normed = self.norm1(x)
        if isinstance(self.attn, GeometryRoutedAttention):
            attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                                 deterministic=deterministic, t=t,
                                 geometry_mask=geometry_mask,
                                 routing_active=routing_active,
                                 router_row_active=router_row_active)
        else:
            attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                                 deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        geometry_router_enabled: bool = False,
        geometry_router_layers: str = "all",
        geometry_router_denoiser_only: bool = False,
        geometry_router_sample_size: int = 32,
        geometry_router_quad_samples: int = 512,
        geometry_router_tau_h: float = 4.0,
        geometry_router_tau_s: float = 4.0,
        geometry_router_bias_e: float = 2.0,
        geometry_router_bias_h: float = -2.0,
        geometry_router_bias_s: float = -2.0,
        geometry_router_learnable_bias: bool = False,
        geometry_router_time_e_bias: float = 1.0,
        geometry_router_time_h_bias: float = 0.0,
        geometry_router_time_s_bias: float = 0.0,
        geometry_router_sphere_k: str = "0.25,0.5,1.0,2.0,4.0",
        geometry_router_eps: float = 1e-6,
        geometry_router_detach_scores: bool = True,
        geometry_router_log_metrics: bool = False,
        geometry_hyperbolic_curvature: float = 1.0,
        geometry_hyperbolic_score: str = "busemann_proxy",
        geometry_sphere_score: str = "cosine",
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )

        # Geometry router: one (parameter-free) router per routed block, so
        # the disabled model keeps an identical state_dict to the original.
        self.geometry_router_enabled = geometry_router_enabled
        self.geometry_router_denoiser_only = geometry_router_denoiser_only
        routed_layers = (
            parse_layer_spec(geometry_router_layers, depth)
            if geometry_router_enabled else set()
        )

        def _make_router() -> GeometryRouter:
            return GeometryRouter(
                sample_size=geometry_router_sample_size,
                quad_samples=geometry_router_quad_samples,
                sphere_k_candidates=geometry_router_sphere_k,
                tau_h=geometry_router_tau_h,
                tau_s=geometry_router_tau_s,
                bias_e=geometry_router_bias_e,
                bias_h=geometry_router_bias_h,
                bias_s=geometry_router_bias_s,
                learnable_bias=geometry_router_learnable_bias,
                time_e_bias=geometry_router_time_e_bias,
                time_h_bias=geometry_router_time_h_bias,
                time_s_bias=geometry_router_time_s_bias,
                eps=geometry_router_eps,
                detach_scores=geometry_router_detach_scores,
                log_metrics=geometry_router_log_metrics,
            )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            block_routed = i in routed_layers
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
                geometry_router_enabled=block_routed,
                geometry_router=_make_router() if block_routed else None,
                hyperbolic_curvature=geometry_hyperbolic_curvature,
                hyperbolic_score=geometry_hyperbolic_score,
                sphere_score=geometry_sphere_score,
            ))

        # Final flow-matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def geometry_router_metrics(self) -> dict:
        """Return latest per-layer router metrics for offline inspection."""
        metrics = {}
        for idx, block in enumerate(self.blocks):
            attn = getattr(block, "attn", None)
            router = getattr(attn, "geometry_router", None)
            if router is None:
                continue
            entry = {}
            if router.latest_gate_mean is not None:
                gate = router.latest_gate_mean.detach().float().cpu()
                entry.update({
                    "gate_e": float(gate[0]),
                    "gate_h": float(gate[1]),
                    "gate_s": float(gate[2]),
                })
            if router.latest_e_h_mean is not None:
                entry["e_H"] = float(router.latest_e_h_mean.detach().float().cpu())
            if router.latest_e_s_mean is not None:
                entry["e_S"] = float(router.latest_e_s_mean.detach().float().cpu())
            if router.latest_logits_mean is not None:
                logits = router.latest_logits_mean.detach().float().cpu()
                entry.update({
                    "logit_e": float(logits[0]),
                    "logit_h": float(logits[1]),
                    "logit_s": float(logits[2]),
                })
            if getattr(router, "learnable_bias", False):
                bias = router.bias.detach().float().cpu()
                entry.update({
                    "bias_e": float(bias[0]),
                    "bias_h": float(bias[1]),
                    "bias_s": float(bias[2]),
                })
            if entry:
                metrics[f"layer_{idx}"] = entry
        return metrics

    def build_context(self, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid."""
        B = x.shape[0]

        # Geometry router statistics only consider the actual text/latent
        # tokens, never the prepended control tokens, so remember the
        # original-token validity before any prefix concatenation.
        original_token_mask = None
        if self.geometry_router_enabled:
            if attention_mask is None:
                original_token_mask = torch.ones(
                    (B, x.shape[1]), dtype=torch.float32, device=x.device)
            else:
                original_token_mask = attention_mask.to(torch.float32).clone()

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Sequence layout after prepends is [prefix_tokens, mode_tokens,
        # actual_tokens]; control tokens stay valid for attention but are
        # excluded from the router's geometry statistics.
        geometry_mask = None
        if self.geometry_router_enabled:
            geometry_mask = torch.cat([
                torch.zeros((B, prefix_len + model_mode_offset),
                            dtype=torch.float32, device=x.device),
                original_token_mask,
            ], dim=1)

        # Denoiser-only routing: decoder-mode rows fall back to the pure
        # Euclidean gate; an all-decoder forward (decode path at generation,
        # decoder_step_active=True) bypasses routing entirely. A forward
        # without decoder_step_active is a denoiser call and stays routed.
        routing_active = True
        router_row_active = None
        if (self.geometry_router_enabled and self.geometry_router_denoiser_only
                and decoder_step_active is not None):
            if isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                router_row_active = 1.0 - decoder_step_active.to(torch.float32).reshape(-1)
            elif bool(decoder_step_active):
                routing_active = False

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=self.feat_rope, attention_mask=attention_mask,
                                 deterministic=deterministic, t=t, geometry_mask=geometry_mask,
                                 routing_active=routing_active,
                                 router_row_active=router_row_active)

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask,
                          deterministic=deterministic, t=t, geometry_mask=geometry_mask,
                          routing_active=routing_active,
                          router_row_active=router_row_active)

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            output = self.final_layer(x.float())
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
