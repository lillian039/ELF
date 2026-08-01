import jax
import jax.numpy as jnp
import flax.linen as nn

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
)


class ManifoldCode(nn.Module):
    """Low-rank semantic-manifold code (M2): pooled latent -> c in R^k -> phi = U c.

    The k-dim bottleneck (code_dim << out_dim) forces phi to carry only reusable
    global structure. Returns (phi_lift, mu, logvar). The forward uses a
    deterministic code c = mu; logvar is produced for the information-bottleneck
    KL only. (Upgrade path: sample c = mu + exp(.5*logvar)*eps when not
    deterministic, threading a 'code' rng — left off for the scaffold.)
    """
    code_dim: int      # k
    out_dim: int       # d (text_encoder_dim)
    hidden: int = 256

    @nn.compact
    def __call__(self, pooled):
        h = nn.gelu(nn.Dense(self.hidden, name='enc_in')(pooled))
        mu = nn.Dense(self.code_dim, name='enc_mu')(h)
        logvar = nn.Dense(self.code_dim, name='enc_logvar')(h)
        c = mu  # deterministic code for the scaffold
        phi = nn.Dense(self.out_dim, use_bias=False, name='lift')(c)  # U c
        return phi, mu, logvar


def apply_manifold_code(manifold_params, pooled, code_dim, out_dim):
    """Re-probe the trained manifold encoder standalone (M2 cycle / steering).

    Reuses the sub-params created by the ManifoldCode inside ELF.__call__
    (stored under params['manifold']), so it shares weights with the forward
    pass without needing a second @compact method on ELF.
    Returns (phi_lift, mu, logvar).
    """
    return ManifoldCode(code_dim, out_dim).apply({"params": manifold_params}, pooled)


class ELFBlock(nn.Module):
    """ELF Transformer block."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x)
        attn_out = Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=self.attn_drop, proj_drop=self.proj_drop, name='attn',
        )(x_normed, rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        x = x + attn_out

        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x)
        mlp_out = SwiGLUFFN(self.hidden_size, mlp_hidden_dim, drop=self.proj_drop, name='mlp')(
            x_normed, deterministic=deterministic,
        )
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer."""
    text_encoder_dim: int
    max_length: int
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    bottleneck_dim: int = 128
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 0  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    num_phi_tokens: int = 0  # If > 0, prepend in-context tokens carrying the semantic code phi(s)
    manifold_dim: int = 0  # M2: if > 0, phi is produced by a low-rank ManifoldCode (k = manifold_dim)
    vocab_size: int = 0  # Vocabulary size for decoder unembedding

    def build_context(self, t, self_cond_cfg_scale=None, phi=None, phi_alt=None):
        """Returns (prefix_tokens, phi_token_start). phi_alt, when given, appends a
        SECOND phi token group sharing phi_proj/phi_tokens parameters (used by the
        depth-window localization eval; no new parameters are created)."""
        prefix_tokens = []
        B = t.shape[0]

        def _make_prefix(emb, n_tokens, param_name):
            tokens = self.param(param_name, NORMAL_INIT_002, (1, n_tokens, self.hidden_size))
            return jnp.tile(tokens, (B, 1, 1)) + jnp.expand_dims(emb, 1)

        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        time_emb = TimestepEmbedder(self.hidden_size, name='t_embedder')(t)
        prefix_tokens.append(_make_prefix(time_emb, self.num_time_tokens, 't_emb_tokens'))

        if self_cond_cfg_scale is not None:
            sc_emb = TimestepEmbedder(self.hidden_size, name='self_cond_cfg_embedder')(self_cond_cfg_scale)
            if self.num_self_cond_cfg_tokens > 0:
                prefix_tokens.append(_make_prefix(sc_emb, self.num_self_cond_cfg_tokens, 'self_cond_cfg_tokens'))

        # Semantic code phi(s): (B, text_encoder_dim) projected to hidden_size and
        # broadcast across num_phi_tokens learnable conditioning slots. Dropping phi
        # (passing zeros) yields the unconditional branch for CFG.
        phi_token_start = sum(p.shape[1] for p in prefix_tokens)
        if phi is not None and self.num_phi_tokens > 0:
            phi_proj = nn.Dense(
                self.hidden_size, use_bias=True,
                kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT, name='phi_proj',
            )
            # Declare the slot bank ONCE (flax params are single-declaration);
            # both phi groups share it, so the parameter structure is unchanged.
            phi_slots = self.param('phi_tokens', NORMAL_INIT_002,
                                   (1, self.num_phi_tokens, self.hidden_size))
            tiled = jnp.tile(phi_slots, (B, 1, 1))
            prefix_tokens.append(tiled + jnp.expand_dims(phi_proj(phi), 1))
            if phi_alt is not None:
                prefix_tokens.append(tiled + jnp.expand_dims(phi_proj(phi_alt), 1))

        return prefix_tokens, phi_token_start

    @nn.compact
    def __call__(
        self, x, t, attention_mask=None, deterministic=True,
        self_cond_cfg_scale=None, decoder_step_active=None, phi=None,
        phi_lifted=False, phi_alt=None, phi_layer_select=None,
    ):
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid.
        phi: optional (N, text_encoder_dim) semantic code injected as a conditioning prefix.
        phi_lifted: static bool; True means phi is already the lifted conditioning
        vector U c, so the ManifoldCode is bypassed (counterfactual training pass).
        phi_alt / phi_layer_select: depth-window localization (inference only).
        phi_alt is a second conditioning vector appended as a parallel phi token
        group (shared parameters); phi_layer_select is a static length-`depth`
        sequence of 0/1 where 1 means layer i attends to the PRIMARY phi group
        and 0 means it attends to the ALT group (the other group is masked out
        of that layer's keys). Both default to None: exact original behavior."""
        patch_size = 1
        head_dim = self.hidden_size // self.num_heads
        B = x.shape[0]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = nn.Dense(
                self.text_encoder_dim, use_bias=True,
                kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT, name='self_cond_proj',
            )(x)

        # Text projection (with bottleneck)
        x = BottleneckTextProj(
            self.text_encoder_dim, self.hidden_size, self.bottleneck_dim, name='text_proj',
        )(x)

        # Prepend learnable model-mode tokens (gated: zero unless decoder_step_active=True)
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = jnp.tile(
                self.param('mode_tokens', NORMAL_INIT_002,
                           (1, self.num_model_mode_tokens, self.hidden_size)),
                (B, 1, 1),
            )
            active_gate = jnp.array(False) if decoder_step_active is None else decoder_step_active
            mode_tokens = mode_tokens * active_gate.astype(mode_tokens.dtype)
            x = jnp.concatenate([mode_tokens, x], axis=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = jnp.ones((B, self.num_model_mode_tokens), dtype=attention_mask.dtype)
                attention_mask = jnp.concatenate([mode_mask, attention_mask], axis=1)

        # M2: when manifold_dim > 0, the incoming phi is the pooled latent (B, C);
        # map it through the low-rank ManifoldCode to get the conditioning vector.
        # M1 (manifold_dim == 0) uses phi (the masked mean) directly.
        phi_for_prefix = phi
        if phi is not None and self.manifold_dim > 0 and not phi_lifted:
            manifold = ManifoldCode(self.manifold_dim, self.text_encoder_dim, name='manifold')
            phi_for_prefix, _, _ = manifold(phi)
            if phi_alt is not None:
                phi_alt, _, _ = manifold(phi_alt)

        prefix_len = 0
        phi_token_start = 0
        context_prefix_tokens, phi_token_start = self.build_context(
            t, self_cond_cfg_scale, phi=phi_for_prefix, phi_alt=phi_alt)
        if context_prefix_tokens:
            prefix_tokens = jnp.concatenate(context_prefix_tokens, axis=1)
            prefix_len = prefix_tokens.shape[1]
            x = jnp.concatenate([prefix_tokens, x], axis=1)
            if attention_mask is not None:
                prefix_mask = jnp.ones((B, prefix_len), dtype=attention_mask.dtype)
                attention_mask = jnp.concatenate([prefix_mask, attention_mask], axis=1)

        feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=self.max_length,
            num_empty_token=prefix_len + model_mode_offset, name='feat_rope',
        )

        # Depth-window localization: per-layer key masks hiding one phi group.
        # Positions are static; the loop below is unrolled Python, so per-layer
        # masks are ordinary traced arrays. attention_mask may be None in the
        # standard path but is required here to carry the group masking.
        layer_masks = None
        if phi is not None and phi_alt is not None and phi_layer_select is not None:
            if attention_mask is None:
                attention_mask = jnp.ones((B, x.shape[1]), dtype=jnp.int32)
            p0 = phi_token_start
            p1 = p0 + self.num_phi_tokens          # primary group [p0, p1)
            p2 = p1 + self.num_phi_tokens          # alt group     [p1, p2)
            mask_primary = attention_mask.at[:, p1:p2].set(0)  # layer sees PRIMARY
            mask_alt = attention_mask.at[:, p0:p1].set(0)      # layer sees ALT
            layer_masks = [mask_primary if int(s) == 1 else mask_alt
                           for s in phi_layer_select]

        q1, q3 = self.depth // 4, self.depth // 4 * 3
        for i in range(self.depth):
            in_drop_range = q3 > i >= q1
            block = ELFBlock(
                self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_drop=self.attn_drop if in_drop_range else 0.0,
                proj_drop=self.proj_drop if in_drop_range else 0.0,
                name=f'blocks_{i}',
            )
            mask_i = layer_masks[i] if layer_masks is not None else attention_mask
            x = block(x, rope_fn=feat_rope, attention_mask=mask_i, deterministic=deterministic)

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        decoder_logits = None
        bn = self.text_encoder_dim
        proj_kernel = self.param('proj_kernel', DEFAULT_KERNEL_INIT, (self.hidden_size, bn))
        proj_bias = self.param('proj_bias', DEFAULT_BIAS_INIT, (bn,))
        unembed_kernel = self.param('unembed_kernel', DEFAULT_KERNEL_INIT, (bn, self.vocab_size))
        unembed_bias = self.param('unembed_bias', DEFAULT_BIAS_INIT, (self.vocab_size,))
        if decoder_step_active is not None:
            decoder_logits = jax.lax.cond(
                decoder_step_active,
                lambda xi: jax.nn.gelu(xi @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias,
                lambda xi: jnp.zeros((*xi.shape[:2], self.vocab_size), dtype=xi.dtype),
                x,
            )

        output = FinalLayer(self.hidden_size, patch_size, self.text_encoder_dim, name='final_layer')(x)
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
