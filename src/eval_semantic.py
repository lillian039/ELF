#!/usr/bin/env python
"""SM-ELF semantic-factorization eval (single device).

Builds a bank of semantic codes phi(s) from held-out stories, then:
  * C1 (phi is used, no collapse): for each phi, generate many samples with the
    SAME phi but DIFFERENT noise. Measure
        - fidelity   : cos(pool(gen), phi_cond)          higher => phi respected
        - within_sim : pairwise cos among same-phi gens   (semantic consistency)
        - across_sim : pairwise cos across different phis  (should be lower)
        - distinct2  : within-group bigram diversity       (not collapsed)
    PASS  if  within_sim > across_sim  AND  distinct2 not collapsed.
  * Dumps all generations to JSONL for inspection / external Gen-PPL (C0).

Usage:
  python3 src/eval_semantic.py \
      --config tinystories_demo/train_tinystories_SM-ELF.yml \
      --checkpoint_path /mnt/faster3/lc2762/elf_tinystories_sm_output/checkpoint_XXXX \
      --num-phi 8 --samples-per-phi 16
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
from modules.model import ELF_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState
from utils.data_utils import load_dataset_split, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.semantic_utils import compute_phi
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml, apply_config_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--num-phi", type=int, default=8, help="Number of distinct phi codes")
    p.add_argument("--samples-per-phi", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="JSONL dump path for generations")
    p.add_argument("--config_override", action="append", default=[],
                   help="Override config values (field=value), e.g. manifold_dim=16.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------------
def _pad_batch(list_of_ids, max_length, pad_id):
    """Pad a list of token-id lists to (B, max_length); return ids + valid mask."""
    B = len(list_of_ids)
    ids = np.full((B, max_length), pad_id, dtype=np.int32)
    valid = np.zeros((B, max_length), dtype=np.float32)
    for i, seq in enumerate(list_of_ids):
        seq = list(seq)[:max_length]
        ids[i, : len(seq)] = seq
        valid[i, : len(seq)] = 1.0
    return jnp.asarray(ids), jnp.asarray(valid)


def _encode(ids, valid, encoder_apply_fn, encoder_params, cfg):
    """T5-encode padded ids -> normalized latent x0 (B, S, C). valid: (B, S)."""
    # Plain (non-cond) self-attention mask: each token attends to all valid tokens.
    enc_mask = jnp.broadcast_to(valid[:, None, :], (valid.shape[0], valid.shape[1], valid.shape[1]))
    x0 = encode_text(
        input_ids=ids, attention_mask=enc_mask,
        encoder_apply_fn=encoder_apply_fn, encoder_params=encoder_params,
        latent_mean=cfg.latent_mean, latent_std=cfg.latent_std,
    )
    return x0


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T


def _mean_pairwise(mat):
    n = mat.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(mat[iu].mean())


def _distinct2(token_id_rows, pad_id, eos_id):
    """Average distinct-2 (unique bigrams / total bigrams) over a list of id rows."""
    vals = []
    for row in token_id_rows:
        toks = [int(t) for t in row if int(t) not in (pad_id, eos_id)]
        bigrams = list(zip(toks[:-1], toks[1:]))
        if not bigrams:
            continue
        vals.append(len(set(bigrams)) / len(bigrams))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        log_for_0("WARNING: config has semantic_factorization=false; this eval expects an SM-ELF model.")
    sc = cfg.sampling_configs[0]
    steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
    sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales
    log_for_0(f"Sampler={sc.sampling_method} steps={steps} sc_cfg={sccfg}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tokenizer, cfg.pad_token)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    max_length = cfg.max_length

    # --- encoder (frozen) ---
    encoder_config, encoder_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d_model = encoder_config.d_model

    # --- model ---
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    input_dim = 2 * d_model if cfg.self_cond_prob > 0 else d_model
    model = ELF_models[cfg.model](
        text_encoder_dim=d_model, max_length=max_length,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size,
        num_model_mode_tokens=cfg.num_model_mode_tokens,
        num_phi_tokens=cfg.num_phi_tokens if cfg.semantic_factorization else 0,
        manifold_dim=cfg.manifold_dim if cfg.semantic_factorization else 0,
        bottleneck_dim=cfg.bottleneck_dim,
    )
    dummy_phi = jnp.ones((1, d_model)) if cfg.semantic_factorization else None
    elf_params = model.init(
        init_rng, x=jnp.ones((1, max_length, input_dim)), t=jnp.ones((1,)),
        deterministic=True,
        self_cond_cfg_scale=jnp.ones((1,)) if cfg.num_self_cond_cfg_tokens > 0 else None,
        phi=dummy_phi,
    )
    state = TrainState.create(
        apply_fn=model.apply, params=elf_params["params"],
        tx=optax.adamw(1e-4), dropout_rng=rng,
        ema_params1=copy.deepcopy(elf_params["params"]),
    )
    state, _ = load_checkpoint(args.checkpoint_path, state)
    params = state.ema_params1 if cfg.eval_use_ema else state.params
    log_for_0("Checkpoint loaded.")

    # --- phi bank from val ---
    val = load_dataset_split(cfg.eval_data_path)
    K = args.num_phi
    raw = [val[i]["input_ids"] for i in range(K)]
    ids, valid = _pad_batch(raw, max_length, pad_id)
    x0 = _encode(ids, valid, encoder_model.apply, encoder_params, cfg)
    phi_bank = np.asarray(compute_phi(x0, valid)[:, 0, :])  # (K, C)
    log_for_0(f"phi bank: {phi_bank.shape}")

    # --- generate M samples per phi ---
    M = args.samples_per_phi
    from utils.sampling_utils import get_sampling_steps
    all_rows = []  # (phi_idx, text, token_ids)
    gen_pooled = []  # pooled phi of each generation
    route = getattr(cfg, "phi_route", "both")
    for k in range(K):
        phi_k = jnp.asarray(np.repeat(phi_bank[k][None, :], M, axis=0))  # (M, C)
        # routing-ablated models saw phi at only one component during
        # training; condition each component at eval exactly as trained
        phi_gen = phi_k if route in ("both", "denoiser") else None
        phi_dec = phi_k if route in ("both", "decoder") else None
        rng, nrng, trng = jax.random.split(rng, 3)
        z = jax.random.normal(nrng, (M, max_length, d_model)) * cfg.denoiser_noise_scale
        t_steps = get_sampling_steps(
            trng, n_steps=steps, time_schedule=sc.time_schedule,
            P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
        )
        latent = _generate_samples_single_batch(
            model_params=params, model_apply_fn=model.apply, rng=nrng,
            z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
            config=cfg, sampling_config=sc, cfg_scale=1.0, self_cond_cfg_scale=sccfg,
            phi=phi_gen,
        )
        pred_ids = _dlm_decode_batch(
            z=latent, model_params=params, model_apply_fn=model.apply,
            t_final_val=float(t_steps[-1]), config=cfg, self_cond_cfg_scale=sccfg, phi=phi_dec,
        )
        pred_ids = np.asarray(mask_after_eos(pred_ids, eos_id, pad_id))
        # re-encode generations to measure their pooled semantic code
        g_valid = (pred_ids != pad_id).astype(np.float32)
        gx0 = _encode(jnp.asarray(pred_ids), jnp.asarray(g_valid), encoder_model.apply, encoder_params, cfg)
        gphi = np.asarray(compute_phi(gx0, jnp.asarray(g_valid))[:, 0, :])  # (M, C)
        gen_pooled.append(gphi)
        for m in range(M):
            txt = tokenizer.decode(pred_ids[m], skip_special_tokens=True)
            all_rows.append((k, txt, pred_ids[m]))

    # --- C1 metrics ---
    gen_pooled = np.stack(gen_pooled, axis=0)  # (K, M, C)
    fid, within, d2 = [], [], []
    for k in range(K):
        g = gen_pooled[k]
        fid.append(float(_cos(g, phi_bank[k][None, :]).mean()))
        within.append(_mean_pairwise(_cos(g, g)))
        d2.append(_distinct2([r[2] for r in all_rows if r[0] == k], pad_id, eos_id))
    # across: centroids of each group, pairwise cos
    centroids = gen_pooled.mean(axis=1)  # (K, C)
    across_sim = _mean_pairwise(_cos(centroids, centroids))

    print("\n" + "=" * 60)
    print("C1 SMOKE TEST (phi is used, no collapse)")
    print("=" * 60)
    print(f"  fidelity   cos(gen, phi_cond) : {np.nanmean(fid):.4f}   (higher => phi respected)")
    print(f"  within_sim same-phi pairwise : {np.nanmean(within):.4f}")
    print(f"  across_sim diff-phi centroids: {across_sim:.4f}")
    print(f"  distinct-2 within group      : {np.nanmean(d2):.4f}   (collapse if ~0)")
    verdict = (np.nanmean(within) > across_sim + 0.02) and (np.nanmean(d2) > 0.3)
    print(f"\n  VERDICT: {'PASS — phi carries semantics & no collapse' if verdict else 'FAIL — inspect (phi ignored or memorized)'}")
    print("=" * 60 + "\n")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint_path.rstrip('/')) or '.', "semantic_eval_samples.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for k, txt, _ in all_rows:
            f.write(json.dumps({"phi_idx": int(k), "generated": txt}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} generations to {out} (use for Gen-PPL / inspection).")


if __name__ == "__main__":
    main()
