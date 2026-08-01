#!/usr/bin/env python
"""Time-resolved leakage: WHEN along the denoising trajectory does off-target
drift materialize?

The toy model (sec:toy) says leakage is realized at the manifold snap, i.e.
late in the trajectory (t -> 1), while the steer itself is injected from t=0.
Alternative: leakage accumulates linearly from the start with global structure.
This script decides.

Protocol: paired same-noise trajectories. For each fixed noise draw, run the
sampler once with the unsteered code c0 and once with c0 +- alpha*u_perp
(sentiment axis, decontaminated against the other core axes). At every
integration step t record the pooled full-embedding estimate
pool(phi + x_pred_t) projected on the frozen gender direction (off-target) and
the frozen sentiment direction (on-target), steered minus unsteered. Because
noise is paired, the curves are pure steering effects.

Output: leakage_curve[t] and control_curve[t] (calibrated units), per model.

Usage:
  python3 src/eval_leakage_trajectory.py --config <cfg> --checkpoint_path <ckpt> \
      [--alpha 3] [--samples 64] --out paper/traj_<tag>.json
"""

import argparse
import copy
import json
import os
import sys

import jax
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--windows", type=str, default="",
                   help="Comma list of windows 'a:b' during which the steer is "
                        "active (unsteered phi outside). Empty = curve mode only.")
    p.add_argument("--window-mode", choices=["t", "index"], default="t",
                   help="'t': a:b are t-values (careful: the sde grid is "
                        "nonuniform). 'index': a:b are FRACTIONS of the step "
                        "sequence (the final ode step is the last index), which "
                        "decomposes by step order regardless of the t schedule.")
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
    w_g, b_g = _fit_linear_classifier(emb[masks["gender"][0] | masks["gender"][1]],
                                      np.where(masks["gender"][0], 1, -1)[masks["gender"][0] | masks["gender"][1]])
    w_s, b_s = _fit_linear_classifier(emb[masks["sentiment"][0] | masks["sentiment"][1]],
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

    def run_traj(phi_vec, sample_rng, phi_base_vec=None, window=None):
        """Unrolled sampler; returns per-step pooled full-embedding estimates.

        If `window=(a, b)` is given, `phi_vec` conditions only steps whose t is
        in [a, b); `phi_base_vec` conditions the rest. The reconstruction adds
        back whichever phi was active at the final step."""
        n_total = len(t_steps) - 1  # sde/ode pairs + final ode step

        def phi_at(t, i):
            if window is None:
                return phi_vec
            if args.window_mode == "index":
                frac = i / n_total
                active = window[0] <= frac < window[1]
            else:
                active = window[0] <= float(t) < window[1]
            return phi_vec if active else phi_base_vec

        def batched(v):
            return jnp.asarray(np.repeat(v[None, :], M, axis=0))

        def kwargs_at(t, i):
            return dict(model_apply_fn=m0.apply, model_params=m0_params, config=cfg,
                        cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                        phi=batched(phi_at(t, i)))
        z = z0
        x_pred = jnp.zeros_like(z)
        pooled = []
        gamma = getattr(sc, "sde_gamma", 0.0)
        n_pairs = len(t_steps) - 2
        for i in range(n_pairs):
            t, t_next = t_steps[i], t_steps[i + 1]
            if sc.sampling_method == "sde":
                step_rng = jax.random.fold_in(sample_rng, i)
                z, x_pred = _sde_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                                      gamma=gamma, rng=step_rng, **kwargs_at(t, i))
            else:
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                                      **kwargs_at(t, i))
            full = x_pred + batched(phi_at(t, i))[:, None, :]
            pooled.append(np.asarray(full.mean(axis=1)))  # (M, d)
        t_last = t_steps[-2]
        z, x_pred = _ode_step(z=z, t=t_last, t_next=t_steps[-1],
                              x_pred_prev=x_pred, **kwargs_at(t_last, n_pairs))
        full = x_pred + batched(phi_at(t_last, n_pairs))[:, None, :]
        pooled.append(np.asarray(full.mean(axis=1)))
        return np.stack(pooled), np.asarray(t_steps[1:])  # (T, M, d), (T,)

    curves = {}
    base_pooled, tgrid = run_traj(phi_of(c0), rng)
    for sgn in (+1, -1):
        c_st = c0 + sgn * args.alpha * u
        st_pooled, _ = run_traj(phi_of(c_st), rng)  # same rng -> paired noise
        d_pool = st_pooled - base_pooled            # (T, M, d)
        curves[f"gender_{'+' if sgn > 0 else '-'}"] = (d_pool @ w_g).mean(axis=1).tolist()
        curves[f"sentiment_{'+' if sgn > 0 else '-'}"] = (
            sgn * (d_pool @ w_s)).mean(axis=1).tolist()

    gp = np.abs(np.array(curves["gender_+"])); gm = np.abs(np.array(curves["gender_-"]))
    sp = np.array(curves["sentiment_+"]); sm = np.array(curves["sentiment_-"])
    leak = 0.5 * (gp + gm)
    ctrl = 0.5 * (sp + sm)
    print("\nTRAJ  t | on-target (sent) | off-target |gender|")
    for i in range(len(tgrid)):
        print(f"TRAJ_POINT t={float(tgrid[i]):.3f} ctrl={ctrl[i]:+.4f} leak={leak[i]:.4f}")
    def t_at_frac(curve, frac):
        """First t where the |curve| level reaches frac of its final value."""
        tgt = frac * abs(curve[-1])
        for i, v in enumerate(np.abs(curve)):
            if v >= tgt:
                return float(tgrid[i])
        return float(tgrid[-1])
    print(f"TRAJ_SUMMARY k={cfg.manifold_dim} final_ctrl={ctrl[-1]:.4f} "
          f"final_leak={leak[-1]:.4f} "
          f"leak_t50={t_at_frac(leak, 0.5):.3f} ctrl_t50={t_at_frac(ctrl, 0.5):.3f} "
          f"leak_at_t075={float(np.interp(0.75, tgrid, leak)/max(abs(leak[-1]),1e-9)):.3f} "
          f"ctrl_at_t075={float(np.interp(0.75, tgrid, ctrl)/max(abs(ctrl[-1]),1e-9)):.3f}")
    windows_out = {}
    if args.windows:
        phi0 = phi_of(c0)
        for spec in args.windows.split(","):
            a, b = (float(x) for x in spec.split(":"))
            gs, ss = [], []
            for sgn in (+1, -1):
                c_st = c0 + sgn * args.alpha * u
                wp, _ = run_traj(phi_of(c_st), rng, phi_base_vec=phi0, window=(a, b))
                dfin = wp[-1] - base_pooled[-1]      # (M, d) final-state effect
                gs.append(abs(float((dfin @ w_g).mean())))
                ss.append(sgn * float((dfin @ w_s).mean()))
            windows_out[spec] = {"leak": 0.5 * (gs[0] + gs[1]),
                                 "ctrl": 0.5 * (ss[0] + ss[1])}
            print(f"WINDOW_SUMMARY k={cfg.manifold_dim} window={spec} "
                  f"ctrl={windows_out[spec]['ctrl']:+.4f} "
                  f"leak={windows_out[spec]['leak']:.4f}")
    if args.out:
        json.dump({"k": int(cfg.manifold_dim), "t": tgrid.tolist(),
                   "ctrl": ctrl.tolist(), "leak": leak.tolist(), "curves": curves,
                   "windows": windows_out},
                  open(args.out, "w"), indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
