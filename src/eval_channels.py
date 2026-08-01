#!/usr/bin/env python
"""Carrier decomposition: how much of measured leakage travels through the
DENOISER's phi conditioning vs the DECODER HEAD's phi conditioning?

ELF conditions two components on phi: the denoiser (every integration step)
and the decoder head (the latent-to-token readout). The 2x2 here crosses
them at inference: latents generated with phi_gen in {base, steer}, each
decoded with phi_dec in {base, steer}. Readout is text-level (decode,
re-embed with the frozen encoder, apply the held-out classifiers), the same
instrument the leakage protocol uses, so cells are directly comparable to
the paper's leakage numbers.

Cells (per steering sign, averaged over +/-):
  (base , base ) reference
  (steer, steer) standard steering       = total effect
  (steer, base ) denoiser channel only
  (base , steer) decoder channel only

Usage:
  python3 src/eval_channels.py --config <cfg> --checkpoint_path <ckpt> \
      [--samples 48] [--alpha 3] --out paper/channels_<tag>.json
"""

import argparse
import copy
import json
import os
import sys

import jax

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
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_steering import _pad_batch, _encode
from eval_leakage_pairs import ATTRS, attr_value, pos_neg_masks, fit_axis, decontaminate
from eval_leakage_continuous import _embed_texts, _fit_linear_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=400)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--alpha", type=float, default=3.0)
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
    gm = masks["gender"][0] | masks["gender"][1]
    w_g, b_g = _fit_linear_classifier(emb[gm], np.where(masks["gender"][0], 1, -1)[gm])
    sm = masks["sentiment"][0] | masks["sentiment"][1]
    w_s, b_s = _fit_linear_classifier(emb[sm], np.where(masks["sentiment"][0], 1, -1)[sm])
    c0 = mu.mean(0)

    def phi_of(c):
        return ((c.astype(np.float32) @ U) if is_m2 else c.astype(np.float32))

    M = args.samples
    rng, nrng, trng = jax.random.split(jax.random.PRNGKey(args.seed + 1), 3)
    z = jax.random.normal(nrng, (M, L, d)) * cfg.denoiser_noise_scale
    t_steps = get_sampling_steps(trng, n_steps=steps, time_schedule=sc.time_schedule,
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)

    def batched(v):
        return jnp.asarray(np.repeat(v[None, :], M, axis=0))

    def gen_latent(phi_vec, sample_rng):
        return _generate_samples_single_batch(
            model_params=m0_params, model_apply_fn=m0.apply, rng=sample_rng,
            z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
            config=cfg, sampling_config=sc, cfg_scale=1.0,
            self_cond_cfg_scale=sccfg, phi=batched(phi_vec))

    def decode(latent, phi_vec):
        pred = np.asarray(mask_after_eos(_dlm_decode_batch(
            z=latent, model_params=m0_params, model_apply_fn=m0.apply,
            t_final_val=float(t_steps[-1]), config=cfg,
            self_cond_cfg_scale=sccfg, phi=batched(phi_vec)), eos_id, pad_id))
        return [tok.decode(pred[m], skip_special_tokens=True) for m in range(M)]

    def score(txts):
        e = _embed_texts(txts, tok, L, pad_id, enc_model, enc_params, cfg)
        return float((e @ w_g - b_g).mean()), float((e @ w_s - b_s).mean())

    phi0 = phi_of(c0)
    lat_base = gen_latent(phi0, rng)
    ref_g, ref_s = score(decode(lat_base, phi0))

    cells = {}
    for name in ("total", "denoiser_only", "decoder_only"):
        gs, ss = [], []
        for sgn in (+1, -1):
            phi_st = phi_of(c0 + sgn * args.alpha * u)
            if name == "total":
                txts = decode(gen_latent(phi_st, rng), phi_st)
            elif name == "denoiser_only":
                txts = decode(gen_latent(phi_st, rng), phi0)
            else:
                txts = decode(lat_base, phi_st)
            g, s = score(txts)
            gs.append(abs(g - ref_g))
            ss.append(sgn * (s - ref_s))
        cells[name] = {"leak": 0.5 * (gs[0] + gs[1]), "ctrl": 0.5 * (ss[0] + ss[1])}
        print(f"CHANNEL k={cfg.manifold_dim} cell={name} "
              f"ctrl={cells[name]['ctrl']:+.4f} leak={cells[name]['leak']:.4f}")
    tot = cells["total"]
    for name in ("denoiser_only", "decoder_only"):
        print(f"CHANNEL_SHARE k={cfg.manifold_dim} cell={name} "
              f"ctrl_share={cells[name]['ctrl'] / (tot['ctrl'] + 1e-9):+.3f} "
              f"leak_share={cells[name]['leak'] / (tot['leak'] + 1e-9):+.3f}")

    if args.out:
        json.dump({"k": int(cfg.manifold_dim), "cells": cells},
                  open(args.out, "w"), indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
