#!/usr/bin/env python
"""Embedding-space SAE test of the linear-basis caveat.

The inference-time projection result says the injected steering delta carries
almost no off-target content in the LINEAR classifier basis (3.7% of its norm).
Caveat: a richer basis could hide off-target content a linear probe misses.
Test: train a sparse autoencoder on pooled frozen-encoder embeddings of real
stories, identify gender-associated dictionary features (by activation
difference between gender classes), and measure how much of the injected
sentiment-steering delta's energy lands on those features, compared with (a)
its energy on sentiment-associated features and (b) the linear-basis 3.7%.

If the gender-feature share of the delta is comparably small, the null is
basis-robust; if it is large, the linear surgery missed real content.

Encoder-only; one GPU, minutes.

Usage:
  python3 src/train_embedding_sae.py --config <cfg> --checkpoint_path <m2 ckpt> \
      [--dict-size 4096] [--l1 3e-4] [--out paper/emb_sae.json]
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
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_steering import _pad_batch, _encode
from eval_leakage_pairs import ATTRS, attr_value, pos_neg_masks, fit_axis, decontaminate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--stories", type=int, default=20000)
    p.add_argument("--dict-size", type=int, default=4096)
    p.add_argument("--l1", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--topk-assoc", type=int, default=64,
                   help="dictionary features counted as attribute-associated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="paper/emb_sae.json")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    is_m2 = cfg.manifold_dim > 0
    assert is_m2, "needs an M2 model for the code-space steering delta"

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    L = cfg.max_length
    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d = enc_cfg.d_model

    # model params for the manifold (steering delta)
    m2 = ELF_models[cfg.model](
        text_encoder_dim=d, max_length=L,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
        num_phi_tokens=cfg.num_phi_tokens, manifold_dim=cfg.manifold_dim,
        bottleneck_dim=cfg.bottleneck_dim,
    )
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
    U = np.asarray(params["manifold"]["lift"]["kernel"])

    # --- embeddings of training stories (for the SAE) + labels ---
    train_ds = load_dataset_split(cfg.eval_data_path.replace("/val", "/train"))
    N = min(args.stories, len(train_ds))
    log_for_0(f"embedding {N} stories for SAE training")
    embs, texts = [], []
    B = 128
    ids_all = [train_ds[i]["input_ids"] for i in range(N)]
    for s in range(0, N, B):
        raw = ids_all[s:s + B]
        texts.extend(tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw)
        ids, valid = _pad_batch(raw, L, pad_id)
        x0 = _encode(ids, valid, enc_model.apply, enc_params, cfg)
        embs.append(np.asarray(compute_phi(x0, valid)[:, 0, :]))
    E = np.concatenate(embs, 0)  # (N, d)
    mu_e, sd_e = E.mean(0), E.std(0) + 1e-6
    En = (E - mu_e) / sd_e

    # --- SAE: x ~ D f, f = relu(W x + b), L2 + l1*|f| ---
    D_SIZE = args.dict_size
    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key)
    sae = {
        "W": jax.random.normal(k1, (d, D_SIZE)) * 0.05,
        "b": jnp.zeros(D_SIZE),
        "D": jax.random.normal(k2, (D_SIZE, d)) * 0.05,
    }
    opt = optax.adam(1e-3)
    opt_state = opt.init(sae)

    @jax.jit
    def step(sae, opt_state, xb):
        def loss_fn(p):
            f = jax.nn.relu(xb @ p["W"] + p["b"])
            xh = f @ p["D"]
            # standard SAE l1: sum over dictionary units, mean over batch
            return (jnp.mean((xh - xb) ** 2)
                    + args.l1 * jnp.mean(jnp.sum(jnp.abs(f), axis=-1))), f
        (l, f), g = jax.value_and_grad(loss_fn, has_aux=True)(sae)
        up, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(sae, up), opt_state, l, (f > 0).mean()

    Enj = jnp.asarray(En)
    n_batches = len(En) // 512
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(len(En))
        for bi in range(n_batches):
            xb = Enj[perm[bi * 512:(bi + 1) * 512]]
            sae, opt_state, l, sparsity = step(sae, opt_state, xb)
        if ep % 20 == 0:
            log_for_0(f"SAE epoch {ep}: loss {float(l):.4f} active {float(sparsity):.3f}")

    # --- attribute-associated features (top-k by class activation difference) ---
    f_all = np.asarray(jax.nn.relu(Enj @ sae["W"] + sae["b"]))
    labels = {a: np.array([attr_value(a, t) for t in texts]) for a in ATTRS}
    assoc = {}
    for a in ATTRS:
        pos, neg = pos_neg_masks(a, labels[a])
        diff = np.abs(f_all[pos].mean(0) - f_all[neg].mean(0))
        assoc[a] = np.argsort(-diff)[:args.topk_assoc]

    # --- injected steering delta in the SAE basis ---
    # code axes from a val split, decontaminated as in the eval protocol
    val = load_dataset_split(cfg.eval_data_path)
    Nv = min(400, len(val))
    vraw = [val[i]["input_ids"] for i in range(Nv)]
    vtexts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in vraw]
    vids, vvalid = _pad_batch(vraw, L, pad_id)
    mus, vpools = [], []
    for s in range(0, Nv, 64):
        x0 = _encode(vids[s:s + 64], vvalid[s:s + 64], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, vvalid[s:s + 64])[:, 0, :]
        vpools.append(np.asarray(pooled))
        _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
        mus.append(np.asarray(mu_b))
    mu = np.concatenate(mus, 0)
    vemb = np.concatenate(vpools, 0)
    vlabels = {a: np.array([attr_value(a, t) for t in vtexts]) for a in ATTRS}
    vmasks = {a: pos_neg_masks(a, vlabels[a]) for a in ATTRS}
    axes = {a: fit_axis(mu, *vmasks[a]) for a in ATTRS}
    u_dec = decontaminate(axes["sentiment"], [axes[b] for b in ATTRS if b != "sentiment"])
    u_raw = axes["sentiment"]

    # attribution averaged over real story embeddings as base points:
    # df = E_e |f(e + delta) - f(e)|
    base = jnp.asarray(En[:256])

    def shares_of(delta_code):
        delta = (args.alpha * delta_code) @ U
        dn = jnp.asarray(delta / sd_e)
        f0 = jax.nn.relu(base @ sae["W"] + sae["b"])
        f1 = jax.nn.relu((base + dn[None, :]) @ sae["W"] + sae["b"])
        df = np.asarray(jnp.abs(f1 - f0).mean(axis=0))
        total = df.sum() + 1e-12
        sh = {a: float(df[assoc[a]].sum() / total) for a in ATTRS}
        rand = float(np.mean([
            df[np.random.default_rng(s).choice(D_SIZE, args.topk_assoc, replace=False)].sum() / total
            for s in range(20)]))
        return sh, rand

    # instrument check: the embedding-space sentiment difference-of-means
    # direction definitionally carries sentiment content; scale it to the same
    # norm as the raw injected delta.
    sp, sn = pos_neg_masks("sentiment", vlabels["sentiment"])
    w_emb = vemb[sp].mean(0) - vemb[sn].mean(0)
    delta_raw_norm = np.linalg.norm((args.alpha * u_raw) @ U)
    w_emb = w_emb / (np.linalg.norm(w_emb) + 1e-12) * delta_raw_norm

    print("\nEMB_SAE_SUMMARY dict=%d l1=%g active=%.3f" % (D_SIZE, args.l1, float(sparsity)))
    out_json = {"dict": D_SIZE, "l1": args.l1, "topk": args.topk_assoc}

    def shares_of_emb(delta):
        dn = jnp.asarray(delta / sd_e)
        f0 = jax.nn.relu(base @ sae["W"] + sae["b"])
        f1 = jax.nn.relu((base + dn[None, :]) @ sae["W"] + sae["b"])
        df = np.asarray(jnp.abs(f1 - f0).mean(axis=0))
        total = df.sum() + 1e-12
        sh = {a: float(df[assoc[a]].sum() / total) for a in ATTRS}
        rand = float(np.mean([
            df[np.random.default_rng(s).choice(D_SIZE, args.topk_assoc, replace=False)].sum() / total
            for s in range(20)]))
        return sh, rand

    sh_i, rand_i = shares_of_emb(w_emb)
    for a in ATTRS:
        print(f"EMB_SAE_SHARE delta=emb_sent_dir attr={a} share={sh_i[a]:.4f}")
    print(f"EMB_SAE_SHARE delta=emb_sent_dir attr=random{args.topk_assoc} share={rand_i:.4f}")
    out_json["emb_sent_dir"] = {"shares": sh_i, "random": rand_i}
    for name, uc in (("raw", u_raw), ("decon", u_dec)):
        sh, rand = shares_of(uc)
        for a in ATTRS:
            print(f"EMB_SAE_SHARE delta={name} attr={a} share={sh[a]:.4f}")
        print(f"EMB_SAE_SHARE delta={name} attr=random{args.topk_assoc} share={rand:.4f}")
        out_json[name] = {"shares": sh, "random": rand}
    json.dump(out_json, open(args.out, "w"), indent=2)
    log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
