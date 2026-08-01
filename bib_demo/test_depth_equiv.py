#!/usr/bin/env python3
"""Bisect the depth-window equivalence gap: single forward pass, plain vs
dual-phi masked, same checkpoint. Exact agreement expected. If this passes
but the trajectory-level check fails, the bug is in the sampler interaction
(e.g. the CFG uncond branch), not the masking."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

from modules.model import ELF_models
from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState
from configs.config import load_config_from_yaml, apply_config_overrides

CFG = sys.argv[1] if len(sys.argv) > 1 else "tinystories_demo/train_tinystories_SM-ELF-M2.yml"
CKPT = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16

cfg = load_config_from_yaml(CFG)
cfg = apply_config_overrides(cfg, [f"manifold_dim={K}"])
tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
d, L = 512, cfg.max_length
m0 = ELF_models[cfg.model](
    text_encoder_dim=d, max_length=L, attn_drop=0.0, proj_drop=0.0,
    num_time_tokens=cfg.num_time_tokens,
    num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
    vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
    num_phi_tokens=cfg.num_phi_tokens, manifold_dim=0,
    bottleneck_dim=cfg.bottleneck_dim,
)
m2 = ELF_models[cfg.model](
    text_encoder_dim=d, max_length=L, attn_drop=0.0, proj_drop=0.0,
    num_time_tokens=cfg.num_time_tokens,
    num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
    vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
    num_phi_tokens=cfg.num_phi_tokens, manifold_dim=cfg.manifold_dim,
    bottleneck_dim=cfg.bottleneck_dim,
)
rng = jax.random.PRNGKey(0)
input_dim = 2 * d if cfg.self_cond_prob > 0 else d
params_init = m2.init(rng, x=jnp.ones((1, L, input_dim)), t=jnp.ones((1,)),
                      deterministic=True,
                      self_cond_cfg_scale=jnp.ones((1,)),
                      phi=jnp.ones((1, d)))
state = TrainState.create(apply_fn=m2.apply, params=params_init["params"],
                          tx=optax.adamw(1e-4), dropout_rng=rng,
                          ema_params1=copy.deepcopy(params_init["params"]))
state, _ = load_checkpoint(CKPT, state)
params = state.ema_params1 if cfg.eval_use_ema else state.params
m0_params = {k: v for k, v in params.items() if k != "manifold"}

B = 4
x = jax.random.normal(jax.random.PRNGKey(1), (B, L, input_dim))
t = jnp.full((B,), 0.5)
scc = jnp.full((B,), 1.0)
phi_st = jax.random.normal(jax.random.PRNGKey(2), (B, d))
phi_b = jax.random.normal(jax.random.PRNGKey(3), (B, d))
depth = {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}[cfg.model]

plain_st, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                       self_cond_cfg_scale=scc, phi=phi_st)
plain_b, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                      self_cond_cfg_scale=scc, phi=phi_b)
for name, sel, ref in (("all-primary", (1,) * depth, plain_st),
                       ("all-alt", (0,) * depth, plain_b)):
    out, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                      self_cond_cfg_scale=scc, phi=phi_st,
                      phi_alt=phi_b, phi_layer_select=sel)
    diff = float(jnp.max(jnp.abs(out - ref)))
    print(f"SINGLE_FWD {name}: max|diff|={diff:.2e} "
          f"({'PASS' if diff < 1e-4 else 'FAIL'})")

# Numerical-floor control: phi_alt == phi, so BOTH groups carry identical
# values and any select is mathematically identical to the plain forward.
# Residual difference here is pure kernel/reduction noise from the longer
# (S+4) graph, not a masking-logic bug.
for name, sel in (("floor-primary", (1,) * depth), ("floor-alt", (0,) * depth)):
    out, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                      self_cond_cfg_scale=scc, phi=phi_st,
                      phi_alt=phi_st, phi_layer_select=sel)
    diff = float(jnp.max(jnp.abs(out - plain_st)))
    print(f"SINGLE_FWD {name} (identical-values control): max|diff|={diff:.2e}")

# Dual-graph self-consistency: within the SAME graph shape, do two
# runs with swapped group roles agree exactly?
out_a, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                    self_cond_cfg_scale=scc, phi=phi_st,
                    phi_alt=phi_b, phi_layer_select=(1,) * depth)
out_b, _ = m0.apply({"params": m0_params}, x=x, t=t, deterministic=True,
                    self_cond_cfg_scale=scc, phi=phi_b,
                    phi_alt=phi_st, phi_layer_select=(0,) * depth)
print(f"SINGLE_FWD role-swap (same graph): max|diff|="
      f"{float(jnp.max(jnp.abs(out_a - out_b))):.2e}")
