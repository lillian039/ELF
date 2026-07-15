#!/usr/bin/env python
"""Directed-pair leakage: steer each attribute, score leakage into every other.

Generalizes eval_leakage_continuous (sentiment->gender only) to all directed
attribute pairs among {sentiment, gender, animal, length}. For each SOURCE
attribute we fit a code-space difference-of-means axis, decontaminate it by
projecting out the span of ALL other attribute axes (QR orthogonal complement,
order-independent), sweep alpha, and score the SAME generations against a
held-out embedding-space linear classifier for EVERY other attribute. So the
generation cost is (#sources x #alphas x seeds), not (#pairs x ...).

Per directed pair (a -> b) we report the non-saturating logit-shift and the
extreme-alpha AUC (as in eval_leakage_continuous), plus per-source on-target
control. Continuous attributes (sentiment score, length) are binarized at
terciles for axis/classifier fitting.

Usage:
  python3 src/eval_leakage_pairs.py --config <m2.yml> \
      --checkpoint_path <ckpt> --config_override manifold_dim=<k> \
      [--sources sentiment,gender,animal,length] [--out pairs_k64.json]
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
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_steering import (
    lexicon_sentiment, gender_label, ANIMAL_WORDS, _pad_batch, _encode,
)
from eval_leakage_continuous import _auc, _embed_texts, _fit_linear_classifier

ATTRS = ["sentiment", "gender", "animal", "length"]


def _toks(text):
    return [w.strip(".,!?;:\"'").lower() for w in text.split()]


def attr_value(name, text):
    """Scalar attribute value of a text (continuous for sentiment/length)."""
    if name == "sentiment":
        return float(lexicon_sentiment(text))
    if name == "gender":
        return float(gender_label(text))
    if name == "animal":
        return 1.0 if any(t in ANIMAL_WORDS for t in _toks(text)) else -1.0
    if name == "length":
        return float(len(_toks(text)))
    raise ValueError(name)


def pos_neg_masks(name, vals):
    """Boolean (pos, neg) masks for axis/classifier fitting. Continuous
    attributes use outer terciles; discrete ones use sign (0 = neutral)."""
    vals = np.asarray(vals, float)
    if name in ("sentiment", "length") and len(np.unique(vals)) > 3:
        lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
        return vals > hi, vals < lo
    return vals > 0, vals < 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--samples-per-alpha", type=int, default=24)
    p.add_argument("--alphas", type=str, default="-3,-2,-1,0,1,2,3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5,
                   help="Bootstrap axes/classifiers + vary generation RNG over this many seeds.")
    p.add_argument("--sources", type=str, default=",".join(ATTRS),
                   help="Comma list of source attributes to steer.")
    p.add_argument("--out", type=str, default="",
                   help="Optional path to dump per-pair results as JSON.")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def fit_axis(mu, pos, neg):
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    u = mu[pos].mean(0) - mu[neg].mean(0)
    n = np.linalg.norm(u)
    return u / n if n > 1e-8 else None


def decontaminate(u, others):
    """Project u onto the orthogonal complement of span(others)."""
    others = [o for o in others if o is not None]
    if not others:
        return u
    Q, _ = np.linalg.qr(np.stack(others, axis=1))
    u = u - Q @ (Q.T @ u)
    n = np.linalg.norm(u)
    return u / n if n > 1e-8 else None


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        sys.exit("requires an SM-ELF model.")
    is_m2 = cfg.manifold_dim > 0
    alphas = [float(a) for a in args.alphas.split(",")]
    sources = [s for s in args.sources.split(",") if s]
    assert all(s in ATTRS for s in sources), f"sources must be among {ATTRS}"
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

    # --- val codes + per-attribute labels ---
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
    labels = {a: np.array([attr_value(a, t) for t in texts]) for a in ATTRS}
    len_std = labels["length"].std() + 1e-8  # to normalize on-target length control

    M = args.samples_per_alpha
    amin, amax = min(alphas), max(alphas)
    nseeds = max(1, args.seeds)
    boot = np.random.default_rng(args.seed)
    base_rng = jax.random.PRNGKey(args.seed)
    # results[(src, tgt)] = list over seeds of (logit_shift, auc); ctrl[src] = list of deltas
    results = {(a, b): [] for a in sources for b in ATTRS if b != a}
    ctrl = {a: [] for a in sources}
    log_for_0(f"k={cfg.manifold_dim} codes {mu.shape} | sources={sources} | {nseeds} seeds "
              f"(decontaminated vs ALL other attribute axes)")

    for si in range(nseeds):
        idx = boot.integers(0, len(mu), size=len(mu)) if nseeds > 1 else np.arange(len(mu))
        mu_s, emb_s = mu[idx], pooled_emb[idx]
        lab_s = {a: labels[a][idx] for a in ATTRS}
        masks = {a: pos_neg_masks(a, lab_s[a]) for a in ATTRS}
        axes = {a: fit_axis(mu_s, *masks[a]) for a in ATTRS}
        clfs = {a: _fit_linear_classifier(emb_s[masks[a][0] | masks[a][1]],
                                          np.where(masks[a][0], 1, -1)[masks[a][0] | masks[a][1]])
                for a in ATTRS if masks[a][0].sum() > 0 and masks[a][1].sum() > 0}
        c0 = mu_s.mean(0)
        rng = jax.random.fold_in(base_rng, si)

        for src in sources:
            if axes[src] is None:
                log_for_0(f"  seed {si}: no axis for source '{src}' (empty class), skipping")
                continue
            u = decontaminate(axes[src], [axes[b] for b in ATTRS if b != src])
            if u is None:
                log_for_0(f"  seed {si}: '{src}' axis vanished under decontamination, skipping")
                continue

            per_alpha_logit = {b: [] for b in ATTRS if b != src}
            all_logit = {b: [] for b in ATTRS if b != src}
            per_alpha_src = []
            all_alpha = []
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
                # one generation batch, scored for every off-target attribute
                for b in per_alpha_logit:
                    if b not in clfs:
                        continue
                    wb, bb = clfs[b]
                    logit = emb_gen @ wb - bb
                    per_alpha_logit[b].append(float(logit.mean()))
                    all_logit[b].extend(logit.tolist())
                # on-target raw value of the source attribute
                gvals = np.array([attr_value(src, g) for g in gtexts])
                per_alpha_src.append(float((gvals > 0).mean()) if src != "length"
                                     else float(gvals.mean()))
                all_alpha.extend([a] * M)
            all_alpha_np = np.array(all_alpha)
            extreme = (all_alpha_np == amin) | (all_alpha_np == amax)
            for b in per_alpha_logit:
                if b not in clfs or not per_alpha_logit[b]:
                    continue
                lg = np.array(all_logit[b])
                auc = _auc(lg[extreme], all_alpha_np[extreme] == amax)
                results[(src, b)].append((float(np.ptp(per_alpha_logit[b])), auc))
            delta = per_alpha_src[-1] - per_alpha_src[0]
            ctrl[src].append(delta / len_std if src == "length" else delta)

    # --- report ---
    k = cfg.manifold_dim
    print("\n" + "=" * 78)
    print(f"DIRECTED-PAIR LEAKAGE  k={k}  N={mu.shape[0]} M={M} seeds={nseeds}")
    print("=" * 78)
    print(f"{'pair (src->tgt)':<24} {'logit_shift':>16} {'AUC':>12} {'src ctrl':>14}")
    out = {"k": int(k), "seeds": nseeds, "pairs": {}, "ctrl": {}}
    for src in sources:
        carr = np.array(ctrl[src]) if ctrl[src] else np.array([np.nan])
        out["ctrl"][src] = [float(carr.mean()), float(carr.std())]
        for tgt in ATTRS:
            if tgt == src or not results[(src, tgt)]:
                continue
            arr = np.array(results[(src, tgt)])  # (seeds, 2)
            ls_m, ls_s = arr[:, 0].mean(), arr[:, 0].std()
            auc_m = arr[:, 1].mean()
            print(f"{src+'->'+tgt:<24} {ls_m:>8.3f}+-{ls_s:<6.3f} {auc_m:>12.3f} "
                  f"{carr.mean():>8.3f}+-{carr.std():<5.3f}")
            out["pairs"][f"{src}->{tgt}"] = {
                "logit_shift": [float(ls_m), float(ls_s)],
                "auc": float(auc_m),
            }
            print(f"LEAKPAIR_SUMMARY k={k} pair={src}->{tgt} "
                  f"logit_shift={ls_m:.3f}+-{ls_s:.3f} auc={auc_m:.3f} "
                  f"ctrl={carr.mean():.3f}+-{carr.std():.3f}")
    print("=" * 78)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
