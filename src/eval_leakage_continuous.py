#!/usr/bin/env python
"""Non-saturating off-target leakage for SM-ELF steering.

The fraction-range leakage saturates (female_frac hits 0/1 at small k, losing
resolution exactly where the effect is strongest). This eval adds two
non-saturating, held-out measures while steering the orthogonalized sentiment axis:

  * gender LOGIT-SHIFT: a held-out linear classifier (difference-of-means in the
    frozen T5 *embedding* space, fit on real val stories with lexicon labels and
    applied to the *generated* stories) gives a continuous gender logit per
    generation; leakage = range of the mean logit over the alpha sweep. Unbounded,
    so it does not saturate.
  * gender AUC: how well that classifier's score separates the alpha-extreme
    generations (alpha_min vs alpha_max). 0.5 = no leakage, 1.0 = total leakage.

The classifier is "held out" in the sense that it is trained on real stories and
applied to model generations (out of distribution), and lives in embedding space,
not the lexicon used to build the steering axis. We still report the old
fraction-range for comparison.

Usage mirrors eval_steering:
  python3 src/eval_leakage_continuous.py --config <m2.yml> \
      --checkpoint_path <ckpt> --config_override manifold_dim=<k> [--alphas=-3,..,3]
"""

import argparse
import copy
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
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_steering import (
    lexicon_sentiment, gender_label, attr_scores, _pad_batch, _encode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--samples-per-alpha", type=int, default=24)
    p.add_argument("--alphas", type=str, default="-3,-2,-1,0,1,2,3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5,
                   help="Bootstrap the axis/classifier + vary generation RNG over this many seeds.")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def _auc(scores, pos_mask):
    """AUC of `scores` separating pos_mask (True) from the rest (rank statistic)."""
    s = np.asarray(scores, float)
    y = np.asarray(pos_mask, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.zeros(len(u))
    np.add.at(avg, inv, ranks)
    avg /= cnt
    ranks = avg[inv]
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))  # symmetric: report separation strength


def _embed_texts(texts, tok, L, pad_id, enc_model, enc_params, cfg):
    ids = [tok(t, truncation=True, max_length=L)["input_ids"] for t in texts]
    out = []
    B = 64
    for s in range(0, len(ids), B):
        chunk = ids[s:s + B]
        pi, pv = _pad_batch(chunk, L, pad_id)
        x0 = _encode(pi, pv, enc_model.apply, enc_params, cfg)
        out.append(np.asarray(compute_phi(x0, pv)[:, 0, :]))  # (b, d) pooled
    return np.concatenate(out, axis=0)


def _fit_linear_classifier(emb, lab):
    """Difference-of-means linear classifier with std calibration. Returns (w, b)
    so that w.e - b is a calibrated logit (>0 => positive class)."""
    pos, neg = emb[lab > 0], emb[lab < 0]
    w = pos.mean(0) - neg.mean(0)
    w = w / (np.linalg.norm(w) + 1e-8)
    proj = emb @ w
    mid = 0.5 * (pos @ w).mean() + 0.5 * (neg @ w).mean()
    scale = 1.0 / (proj.std() + 1e-6)
    return w * scale, mid * scale


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        sys.exit("requires an SM-ELF model.")
    is_m2 = cfg.manifold_dim > 0
    alphas = [float(a) for a in args.alphas.split(",")]
    sc = cfg.sampling_configs[0]
    steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
    sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 1
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

    # --- val codes + labels: code-space sentiment axis (orthogonalized) + embedding classifiers ---
    val = load_dataset_split(cfg.eval_data_path)
    N = min(args.label_stories, len(val))
    raw = [val[i]["input_ids"] for i in range(N)]
    texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
    ids, valid = _pad_batch(raw, L, pad_id)
    mus, pools = [], []
    B = 64
    for s in range(0, N, B):
        x0 = _encode(ids[s:s + B], valid[s:s + B], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, valid[s:s + B])[:, 0, :]
        pools.append(np.asarray(pooled))
        if is_m2:
            _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
            mus.append(np.asarray(mu_b))
        else:
            mus.append(np.asarray(pooled))
    mu = np.concatenate(mus, axis=0)
    pooled_emb = np.concatenate(pools, axis=0)
    slab = np.array([lexicon_sentiment(t) for t in texts])
    glab = np.array([gender_label(t) for t in texts])

    M = args.samples_per_alpha
    amin, amax = min(alphas), max(alphas)
    nseeds = max(1, args.seeds)
    boot = np.random.default_rng(args.seed)
    base_rng = jax.random.PRNGKey(args.seed)
    metrics = []  # per seed: (frac_range, logit_shift, auc, ctrl_delta)
    log_for_0(f"k={cfg.manifold_dim} codes {mu.shape} | {nseeds} seeds "
              f"(bootstrap axis+clf, varied generation RNG)")

    for si in range(nseeds):
        # bootstrap the label set for axis + held-out classifier fit
        idx = boot.integers(0, len(mu), size=len(mu)) if nseeds > 1 else np.arange(len(mu))
        mu_s, emb_s, sl_s, gl_s = mu[idx], pooled_emb[idx], slab[idx], glab[idx]
        u = mu_s[sl_s > 0].mean(0) - mu_s[sl_s < 0].mean(0); u /= np.linalg.norm(u) + 1e-8
        ug = mu_s[gl_s > 0].mean(0) - mu_s[gl_s < 0].mean(0); ug /= np.linalg.norm(ug) + 1e-8
        u = u - (u @ ug) * ug; u /= np.linalg.norm(u) + 1e-8
        c0 = mu_s.mean(0)
        wg, bg = _fit_linear_classifier(emb_s, gl_s)
        rng = jax.random.fold_in(base_rng, si)

        per_alpha_logit, per_alpha_frac, per_alpha_pos = [], [], []
        all_logit, all_alpha = [], []
        for a in alphas:
            c = (c0 + a * u).astype(np.float32)
            phi_vec = (c @ U) if is_m2 else c
            phi_lift = jnp.asarray(np.repeat(phi_vec[None, :], M, axis=0))
            rng, nrng, trng = jax.random.split(rng, 3)
            z = jax.random.normal(nrng, (M, L, d)) * cfg.denoiser_noise_scale
            t_steps = get_sampling_steps(trng, n_steps=steps, time_schedule=sc.time_schedule,
                                         P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
            latent = _generate_samples_single_batch(
                model_params=m0_params, model_apply_fn=m0.apply, rng=nrng,
                z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
                config=cfg, sampling_config=sc, cfg_scale=1.0, self_cond_cfg_scale=sccfg, phi=phi_lift,
            )
            pred = np.asarray(mask_after_eos(_dlm_decode_batch(
                z=latent, model_params=m0_params, model_apply_fn=m0.apply,
                t_final_val=float(t_steps[-1]), config=cfg, self_cond_cfg_scale=sccfg, phi=phi_lift,
            ), eos_id, pad_id))
            gtexts = [tok.decode(pred[m], skip_special_tokens=True) for m in range(M)]
            emb_gen = _embed_texts(gtexts, tok, L, pad_id, enc_model, enc_params, cfg)
            logit = emb_gen @ wg - bg
            per_alpha_logit.append(float(logit.mean()))
            per_alpha_frac.append(float(np.mean([attr_scores(g)[1] for g in gtexts])))
            per_alpha_pos.append(float(np.mean([lexicon_sentiment(g) > 0 for g in gtexts])))
            all_logit.extend(logit.tolist()); all_alpha.extend([a] * M)
        all_logit = np.array(all_logit); all_alpha = np.array(all_alpha)
        extreme = (all_alpha == amin) | (all_alpha == amax)
        auc = _auc(all_logit[extreme], all_alpha[extreme] == amax)
        ctrl = per_alpha_pos[-1] - per_alpha_pos[0]   # on-target sentiment control delta
        metrics.append((float(np.ptp(per_alpha_frac)), float(np.ptp(per_alpha_logit)), auc, ctrl))

    arr = np.array(metrics)             # (nseeds, 4)
    mean, std = arr.mean(0), arr.std(0)
    names = ["frac_range(old)", "logit_shift", "gender_AUC", "ctrl_delta"]
    print("\n" + "=" * 64)
    print(f"NON-SATURATING LEAKAGE (orth. sentiment axis)  k={cfg.manifold_dim}  "
          f"N={mu.shape[0]} M={M} seeds={nseeds}")
    print("=" * 64)
    for i, nm in enumerate(names):
        print(f"  {nm:<18} = {mean[i]:.3f} +- {std[i]:.3f}")
    print("=" * 64)
    print(f"LEAKAGE_SUMMARY k={cfg.manifold_dim} "
          f"frac_range={mean[0]:.3f}+-{std[0]:.3f} logit_shift={mean[1]:.3f}+-{std[1]:.3f} "
          f"auc={mean[2]:.3f}+-{std[2]:.3f} ctrl={mean[3]:.3f}+-{std[3]:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
