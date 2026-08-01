#!/usr/bin/env python
"""Depth-resolved leakage: WHICH LAYERS of the denoiser read the code for
on-target control, and which for off-target leakage?

Complement of eval_leakage_trajectory.py (time windows): here the steered
code is visible only to a contiguous LAYER window, with the unsteered code
visible to every other layer, constant across all integration steps. The
model prepends BOTH phi token groups (shared parameters, see
modules/model.py phi_alt/phi_layer_select) and per-layer key masks select
which group each layer reads.

Readout frame: pooled full-embedding estimates always add back the BASE phi,
so window deltas isolate the denoiser's residual response to where the
steered code is readable; the additive phi delta itself (which \S mitig shows
carries no off-target content) is excluded by construction in every
condition, including the references.

Correctness: exact logic equality of the dual-phi masked forward against the
plain forward is proven at single-forward level by
bib_demo/test_depth_equiv.py (~4e-6 under float32 matmuls). At trajectory
level the graph-shape change seeds fp32 differences that the recursive
sampler amplifies, so every MEASURED condition here (base, full-steer
references, windows) runs through the same dual-phi graph with paired noise
and the shape floor cancels in the differences; the residual plain-vs-dual
gap is reported as a SHAPE_FLOOR diagnostic, not a gate.

Usage:
  python3 src/eval_leakage_depth.py --config <cfg> --checkpoint_path <ckpt> \
      [--layer-windows 0:3,3:6,6:9,9:12] [--samples 64] [--alpha 3] \
      --out paper/depth_<tag>.json
"""

import argparse
import copy
import json
import os
import sys

import jax

# TF32 matmuls (Ampere+/Hopper default) inject ~1e-3 shape-dependent rounding
# that swamps the equivalence checks; this eval requires true float32.
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.t5_encoder import get_encoder
from modules.model import ELF_models, apply_manifold_code
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState
from utils.data_utils import load_dataset_split, get_pad_token_id
from utils.semantic_utils import compute_phi
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _sde_step, _ode_step
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_steering import _pad_batch, _encode
from eval_leakage_pairs import ATTRS, attr_value, pos_neg_masks, fit_axis, decontaminate
from eval_leakage_continuous import _embed_texts, _fit_linear_classifier

DEPTHS = {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--layer-windows", type=str, default="",
                   help="Comma list 'a:b' of layer index windows that read the "
                        "steered code (base code elsewhere). Default: quartiles "
                        "plus per-layer singletons.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        sys.exit("requires an SM-ELF model.")
    is_m2 = cfg.manifold_dim > 0
    depth = DEPTHS[cfg.model]
    sc = cfg.sampling_configs[0]
    steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
    sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    L = cfg.max_length
    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d = enc_cfg.d_model

    def build(manifold_dim):
        return ELF_models[cfg.model](
            text_encoder_dim=d, max_length=L,
            attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
            num_time_tokens=cfg.num_time_tokens,
            num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
            vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
            num_phi_tokens=cfg.num_phi_tokens, manifold_dim=manifold_dim,
            bottleneck_dim=cfg.bottleneck_dim,
        )
    m2, m0 = build(cfg.manifold_dim), build(0)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    input_dim = 2 * d if cfg.self_cond_prob > 0 else d
    params_init = m2.init(
        init_rng, x=jnp.ones((1, L, input_dim)), t=jnp.ones((1,)), deterministic=True,
        self_cond_cfg_scale=jnp.ones((1,)) if cfg.num_self_cond_cfg_tokens > 0 else None,
        phi=jnp.ones((1, d)),
    )
    state = TrainState.create(
        apply_fn=m2.apply, params=params_init["params"], tx=optax.adamw(1e-4),
        dropout_rng=rng, ema_params1=copy.deepcopy(params_init["params"]),
    )
    state, _ = load_checkpoint(args.checkpoint_path, state)
    params = state.ema_params1 if cfg.eval_use_ema else state.params
    m0_params = {k: v for k, v in params.items() if k != "manifold"}
    U = np.asarray(params["manifold"]["lift"]["kernel"]) if is_m2 else None

    # --- axes (code space) + frozen classifiers (embedding space) ---
    val = load_dataset_split(cfg.eval_data_path)
    N = min(args.label_stories, len(val))
    raw = [val[i]["input_ids"] for i in range(N)]
    texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
    ids, valid = _pad_batch(raw, L, pad_id)
    mus, pools = [], []
    for s in range(0, N, 64):
        x0 = _encode(ids[s:s + 64], valid[s:s + 64], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, valid[s:s + 64])[:, 0, :]
        pools.append(np.asarray(pooled))
        if is_m2:
            _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
            mus.append(np.asarray(mu_b))
        else:
            mus.append(np.asarray(pooled))
    mu = np.concatenate(mus, 0)
    emb = np.concatenate(pools, 0)
    labels = {a: np.array([attr_value(a, t) for t in texts]) for a in ATTRS}
    masks = {a: pos_neg_masks(a, labels[a]) for a in ATTRS}
    axes = {a: fit_axis(mu, *masks[a]) for a in ATTRS}
    u = decontaminate(axes["sentiment"], [axes[b] for b in ATTRS if b != "sentiment"])
    w_g, _ = _fit_linear_classifier(emb[masks["gender"][0] | masks["gender"][1]],
                                    np.where(masks["gender"][0], 1, -1)[masks["gender"][0] | masks["gender"][1]])
    w_s, _ = _fit_linear_classifier(emb[masks["sentiment"][0] | masks["sentiment"][1]],
                                    np.where(masks["sentiment"][0], 1, -1)[masks["sentiment"][0] | masks["sentiment"][1]])
    c0 = mu.mean(0)

    def phi_of(c):
        return (c.astype(np.float32) @ U) if is_m2 else c.astype(np.float32)

    M = args.samples
    rng, nrng, trng = jax.random.split(jax.random.PRNGKey(args.seed + 1), 3)
    z0 = jax.random.normal(nrng, (M, L, d)) * cfg.denoiser_noise_scale
    t_steps = get_sampling_steps(trng, n_steps=steps, time_schedule=sc.time_schedule,
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
    cond_seq = jnp.zeros((M, L, d))
    cond_seq_mask = jnp.zeros((M, L))

    def batched(v):
        return jnp.asarray(np.repeat(v[None, :], M, axis=0))

    def run(phi_vec, sample_rng, phi_base_vec=None, layer_select=None):
        """Full trajectory; returns final pooled estimate in the BASE-phi frame.
        layer_select: length-`depth` tuple of 0/1 (1 = layer reads phi_vec,
        0 = layer reads phi_base_vec). None = plain single-phi forward."""
        if layer_select is not None:
            base = batched(phi_base_vec)
            apply_fn = (lambda variables, *a, **kw: m0.apply(
                variables, *a, phi_alt=base, phi_layer_select=tuple(layer_select), **kw))
        else:
            apply_fn = m0.apply
        kwargs = dict(model_apply_fn=apply_fn, model_params=m0_params, config=cfg,
                      cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                      cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                      phi=batched(phi_vec))
        z = z0
        x_pred = jnp.zeros_like(z)
        gamma = getattr(sc, "sde_gamma", 0.0)
        n_pairs = len(t_steps) - 2
        for i in range(n_pairs):
            t, t_next = t_steps[i], t_steps[i + 1]
            if sc.sampling_method == "sde":
                step_rng = jax.random.fold_in(sample_rng, i)
                z, x_pred = _sde_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                                      gamma=gamma, rng=step_rng, **kwargs)
            else:
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **kwargs)
        z, x_pred = _ode_step(z=z, t=t_steps[-2], t_next=t_steps[-1],
                              x_pred_prev=x_pred, **kwargs)
        # BASE-phi frame in every condition (see module docstring).
        full = x_pred + batched(phi0)[:, None, :]
        return np.asarray(full.mean(axis=1))  # (M, d)

    phi0 = phi_of(c0)
    # ALL measured conditions share the dual-phi graph (S includes both phi
    # groups) so shape-dependent fp32 noise, chaotically amplified by the
    # recursive sampler, cancels exactly in the paired differences. base =
    # every layer reads the ALT (base) group; steered ref = every layer reads
    # the PRIMARY (steered) group.
    base_final = run(phi0, rng, phi_base_vec=phi0, layer_select=(0,) * depth)

    # --- shape-floor diagnostic (NOT gating): plain single-phi graph vs the
    # dual graph. Exact logic equality is proven by the single-forward test
    # (bib_demo/test_depth_equiv.py, ~4e-6); at trajectory level the graph
    # shape difference seeds fp32 divergence that the sampler amplifies, so
    # this reports the floor rather than enforcing a bar.
    plain_base = run(phi0, rng)
    plain_steer = run(phi_of(c0 + args.alpha * u), rng)
    dual_steer = run(phi_of(c0 + args.alpha * u), rng, phi_base_vec=phi0,
                     layer_select=(1,) * depth)
    print(f"SHAPE_FLOOR base: max|diff|={float(np.max(np.abs(base_final - plain_base))):.2e}  "
          f"steered: max|diff|={float(np.max(np.abs(dual_steer - plain_steer))):.2e}")

    # --- window sweep ---
    if args.layer_windows:
        specs = args.layer_windows.split(",")
    else:
        q = max(depth // 4, 1)
        specs = [f"{a}:{a + q}" for a in range(0, depth, q)]
        specs += [f"{i}:{i + 1}" for i in range(depth)]

    steered_final = {}
    for sgn in (+1, -1):
        steered_final[sgn] = run(phi_of(c0 + sgn * args.alpha * u), rng,
                                 phi_base_vec=phi0, layer_select=(1,) * depth)
    full_ctrl = 0.5 * sum(sgn * float(((steered_final[sgn] - base_final) @ w_s).mean())
                          for sgn in (+1, -1))
    full_leak = 0.5 * sum(abs(float(((steered_final[sgn] - base_final) @ w_g).mean()))
                          for sgn in (+1, -1))
    print(f"DEPTH_REF k={cfg.manifold_dim} full_ctrl={full_ctrl:+.4f} full_leak={full_leak:.4f}")

    out = {}
    for spec in specs:
        a, b = (int(x) for x in spec.split(":"))
        sel = tuple(1 if a <= i < b else 0 for i in range(depth))
        gs, ss = [], []
        for sgn in (+1, -1):
            wfin = run(phi_of(c0 + sgn * args.alpha * u), rng,
                       phi_base_vec=phi0, layer_select=sel)
            dfin = wfin - base_final
            gs.append(abs(float((dfin @ w_g).mean())))
            ss.append(sgn * float((dfin @ w_s).mean()))
        out[spec] = {"ctrl": 0.5 * (ss[0] + ss[1]), "leak": 0.5 * (gs[0] + gs[1])}
        print(f"DEPTH_WINDOW k={cfg.manifold_dim} layers={spec} "
              f"ctrl={out[spec]['ctrl']:+.4f} leak={out[spec]['leak']:.4f} "
              f"ctrl_share={out[spec]['ctrl'] / (full_ctrl + 1e-9):+.3f} "
              f"leak_share={out[spec]['leak'] / (full_leak + 1e-9):+.3f}")

    if args.out:
        json.dump({"k": int(cfg.manifold_dim), "depth": depth,
                   "full_ctrl": full_ctrl, "full_leak": full_leak,
                   "windows": out}, open(args.out, "w"), indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
