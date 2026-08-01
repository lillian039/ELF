#!/usr/bin/env python
"""Natural-variation hypothesis: leakage flows along the denoiser's own
variability modes.

The paper localizes leakage in the denoiser's conditional response; simple
data-side target scores (insertion movability, class balance, embedding
classifier geometry) fail to explain WHICH targets absorb it. Hypothesis: the
targets that absorb leakage are the attributes the denoiser varies naturally
when resampling noise at a FIXED code; steering merely biases variation the
model already produces.

Measurement: for a handful of fixed codes (the mean code + a few story codes),
generate M samples each by resampling noise, embed the generations, and score
every attribute's held-out linear classifier. Per attribute report the mean
within-code std of the calibrated logit (natural variation) and the pooled
positive rate. No steering anywhere.

Usage:
  python3 src/eval_natural_variation.py --config <cfg> --checkpoint_path <ckpt> \
      [--codes 5] [--samples-per-code 24] --out paper/natvar_<tag>.json
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

from eval_steering import _pad_batch, _encode
from eval_leakage_pairs import ATTRS, EXTRA_ATTRS, EXTRA2_ATTRS, attr_value, pos_neg_masks
from eval_leakage_continuous import _embed_texts, _fit_linear_classifier

TARGETS = ATTRS + EXTRA_ATTRS + EXTRA2_ATTRS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--codes", type=int, default=5,
                   help="number of fixed codes (mean code + codes-1 story codes)")
    p.add_argument("--samples-per-code", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5, help="classifier bootstrap seeds")
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
    labels = {a: np.array([attr_value(a, t) for t in texts]) for a in TARGETS}

    # fixed codes: mean code + (codes-1) individual story codes
    code_rng = np.random.default_rng(args.seed)
    picks = code_rng.choice(len(mu), size=max(args.codes - 1, 0), replace=False)
    code_list = [mu.mean(0)] + [mu[i] for i in picks]

    M = args.samples_per_code
    rng = jax.random.PRNGKey(args.seed + 1)
    gen_texts_per_code = []
    for ci, c in enumerate(code_list):
        phi_vec = (c.astype(np.float32) @ U) if is_m2 else c.astype(np.float32)
        phi_lift = jnp.asarray(np.repeat(phi_vec[None, :], M, axis=0))
        # routing-ablated models saw phi at only one component during
        # training; condition each component at eval exactly as trained
        route = getattr(cfg, "phi_route", "both")
        phi_gen = phi_lift if route in ("both", "denoiser") else None
        phi_dec = phi_lift if route in ("both", "decoder") else None
        rng, nrng, trng = jax.random.split(rng, 3)
        z = jax.random.normal(nrng, (M, L, d)) * cfg.denoiser_noise_scale
        t_steps = get_sampling_steps(trng, n_steps=steps, time_schedule=sc.time_schedule,
                                     P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
        latent = _generate_samples_single_batch(
            model_params=m0_params, model_apply_fn=m0.apply, rng=nrng,
            z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
            config=cfg, sampling_config=sc, cfg_scale=1.0, self_cond_cfg_scale=sccfg, phi=phi_gen,
        )
        pred = np.asarray(mask_after_eos(_dlm_decode_batch(
            z=latent, model_params=m0_params, model_apply_fn=m0.apply,
            t_final_val=float(t_steps[-1]), config=cfg, self_cond_cfg_scale=sccfg, phi=phi_dec,
        ), eos_id, pad_id))
        gen_texts_per_code.append([tok.decode(pred[m], skip_special_tokens=True) for m in range(M)])
        log_for_0(f"code {ci}: generated {M} samples")

    emb_per_code = [
        _embed_texts(g, tok, L, pad_id, enc_model, enc_params, cfg)
        for g in gen_texts_per_code
    ]

    boot = np.random.default_rng(args.seed)
    out = {"k": int(cfg.manifold_dim), "codes": len(code_list), "M": M, "attrs": {}}
    print("\n" + "=" * 70)
    print(f"NATURAL VARIATION  k={cfg.manifold_dim}  codes={len(code_list)} M={M}")
    print("=" * 70)
    print(f"{'attr':<10} {'logit_std':>16} {'pos_rate':>9} {'flip_lex':>9}")
    for a in TARGETS:
        stds, rates, flips = [], [], []
        for si in range(args.seeds):
            idx = boot.integers(0, N, size=N) if args.seeds > 1 else np.arange(N)
            pos, neg = pos_neg_masks(a, labels[a][idx])
            keep = pos | neg
            if pos.sum() == 0 or neg.sum() == 0:
                continue
            w, b = _fit_linear_classifier(pooled_emb[idx][keep], np.where(pos, 1, -1)[keep])
            per_code_std = [float((e @ w - b).std()) for e in emb_per_code]
            stds.append(float(np.mean(per_code_std)))
            allg = np.concatenate([e @ w - b for e in emb_per_code])
            rates.append(float((allg > 0).mean()))
        # lexicon-level flip probability across noise at fixed code (classifier-free)
        for g in gen_texts_per_code:
            v = np.array([attr_value(a, t) for t in g])
            if a in ("sentiment", "length"):
                v = (v > np.median(v)).astype(float)
            else:
                v = (v > 0).astype(float)
            flips.append(float(v.std()))
        sm, ss = float(np.mean(stds)), float(np.std(stds))
        rm = float(np.mean(rates))
        fm = float(np.mean(flips))
        out["attrs"][a] = {"logit_std": [sm, ss], "pos_rate": rm, "flip_lex": fm}
        print(f"{a:<10} {sm:>8.3f}+-{ss:<6.3f} {rm:>9.3f} {fm:>9.3f}")
        print(f"NATVAR_SUMMARY k={cfg.manifold_dim} attr={a} logit_std={sm:.4f} "
              f"pos_rate={rm:.3f} flip_lex={fm:.4f}")
    print("=" * 70)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
