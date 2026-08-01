#!/usr/bin/env python
"""SM-ELF C2 steering eval (single device, requires an M2 checkpoint, manifold_dim>0).

Pipeline:
  1. Encode val stories -> pooled latent -> model.reencode -> code mu (=c).
  2. Label each story's sentiment (lexicon by default, or HF classifier) and
     fit a sentiment AXIS u in code space (difference-of-means).
  3. Sweep alpha: c = c0 + alpha*u, lift phi = U c, generate, classify.
  4. Report positive-fraction vs alpha. Monotonic with a large endpoint delta
     => controllability (C2). The cycle loss is what makes it faithful.

Trick: generation feeds a *steered* phi=U c directly. We run it through a
manifold_dim=0 VIEW of the same trained params (phi used as the conditioning
vector as-is, ManifoldCode bypassed), so no extra sampling plumbing is needed.

Usage:
  python3 src/eval_steering.py \
      --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
      --checkpoint_path /mnt/faster3/lc2762/elf_tinystories_sm_m2_output/checkpoint_XXXX \
      --label-stories 200 --samples-per-alpha 24 --alphas -3,-2,-1,0,1,2,3
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
from utils.encoder_utils import encode_text
from utils.semantic_utils import compute_phi
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml, apply_config_overrides


POS_WORDS = set("happy happily smiled smile smiles laugh laughed laughing fun joy joyful "
                "love loved loves kind friend friends friendly played play plays glad excited "
                "wonderful nice good great greatly beautiful proud hug hugged safe won win "
                "yummy delicious magic magical brave cheer cheered cheerful warm gentle".split())
NEG_WORDS = set("sad sadly cried cry crying scared afraid angry mad hurt hurts pain bad lonely "
                "lost fear feared worried worry upset broken sorry fight fought hate hated dark "
                "cold sick fell falling danger dangerous cruel mean scary terrible awful "
                "frightened grumpy nasty trouble".split())


def lexicon_sentiment(text):
    toks = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    p = sum(t in POS_WORDS for t in toks)
    n = sum(t in NEG_WORDS for t in toks)
    return p - n  # >0 positive, <0 negative


# --- off-target attributes (for the disentanglement / leakage test) ---
ANIMAL_WORDS = set("cat cats dog dogs bird birds rabbit bunny fox bear duck frog fish mouse mice "
                   "pig cow horse puppy kitten kitty owl bee butterfly snail turtle squirrel sheep "
                   "lion tiger monkey elephant chick hen goat deer".split())
FEMALE_WORDS = set("she her hers girl woman women mom mommy mother sister aunt grandma queen "
                   "princess lady daughter".split())
MALE_WORDS = set("he him his boy man men dad daddy father brother uncle grandpa king prince guy son".split())


def attr_scores(text):
    """Off-target attributes of a generation: (has_animal, is_female, length)."""
    toks = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    animal = sum(t in ANIMAL_WORDS for t in toks) > 0
    female = sum(t in FEMALE_WORDS for t in toks) > sum(t in MALE_WORDS for t in toks)
    return float(animal), float(female), float(len(toks))


def gender_label(text):
    """+1 female-leaning, -1 male-leaning, 0 neither (for fitting the gender axis)."""
    toks = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    f = sum(t in FEMALE_WORDS for t in toks)
    m = sum(t in MALE_WORDS for t in toks)
    return 1 if f > m else (-1 if m > f else 0)


def _pad_batch(list_of_ids, max_length, pad_id):
    B = len(list_of_ids)
    ids = np.full((B, max_length), pad_id, dtype=np.int32)
    valid = np.zeros((B, max_length), dtype=np.float32)
    for i, seq in enumerate(list_of_ids):
        seq = list(seq)[:max_length]
        ids[i, : len(seq)] = seq
        valid[i, : len(seq)] = 1.0
    return jnp.asarray(ids), jnp.asarray(valid)


def _encode(ids, valid, encoder_apply_fn, encoder_params, cfg):
    enc_mask = jnp.broadcast_to(valid[:, None, :], (valid.shape[0], valid.shape[1], valid.shape[1]))
    return encode_text(
        input_ids=ids, attention_mask=enc_mask,
        encoder_apply_fn=encoder_apply_fn, encoder_params=encoder_params,
        latent_mean=cfg.latent_mean, latent_std=cfg.latent_std,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=200)
    p.add_argument("--samples-per-alpha", type=int, default=24)
    p.add_argument("--alphas", type=str, default="-3,-2,-1,0,1,2,3")
    p.add_argument("--orthogonalize", action="store_true",
                   help="Remove the gender direction from the sentiment axis (disentanglement test).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--config_override", action="append", default=[],
                   help="Override config values (field=value), e.g. manifold_dim=16.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        sys.exit("eval_steering requires an SM-ELF model (semantic_factorization=true).")
    # M2: code c in R^k lifted by U. M1: the "code" is the d-dim mean-pool phi itself
    # (manifold_dim=0), so steering happens directly in phi space with an identity lift.
    is_m2 = cfg.manifold_dim > 0
    alphas = [float(a) for a in args.alphas.split(",")]
    sc = cfg.sampling_configs[0]
    steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
    sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 1
    L, d = cfg.max_length, None

    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d = enc_cfg.d_model

    # Two model views over the SAME params: m2 (manifold on) for reencode,
    # m0 (manifold off) for generation with a directly-supplied phi=U c.
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
    # m0 view: drop the (unused) manifold submodule params to avoid unexpected keys.
    m0_params = {k: v for k, v in params.items() if k != "manifold"}
    if is_m2:
        U = np.asarray(params["manifold"]["lift"]["kernel"])  # (k, d)
        log_for_0(f"Loaded SM checkpoint (M2); lift U: {U.shape}")
    else:
        U = None  # M1: identity lift (phi is the d-dim code itself)
        log_for_0("Loaded SM checkpoint (M1, manifold_dim=0); steering in d-dim phi space")

    # --- codes + sentiment labels on val stories ---
    val = load_dataset_split(cfg.eval_data_path)
    N = min(args.label_stories, len(val))
    raw = [val[i]["input_ids"] for i in range(N)]
    texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
    ids, valid = _pad_batch(raw, L, pad_id)
    # batched encode (chunks) to bound memory
    mus = []
    B = 64
    for s in range(0, N, B):
        x0 = _encode(ids[s:s + B], valid[s:s + B], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, valid[s:s + B])[:, 0, :]
        if is_m2:
            _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
            mus.append(np.asarray(mu_b))
        else:
            mus.append(np.asarray(pooled))  # M1: code = d-dim mean-pool phi

    mu = np.concatenate(mus, axis=0)  # (N, k)
    labels = np.array([lexicon_sentiment(t) for t in texts])
    pos, neg = mu[labels > 0], mu[labels < 0]
    if len(pos) < 5 or len(neg) < 5:
        log_for_0(f"WARNING: few labeled examples (pos={len(pos)}, neg={len(neg)}); axis may be noisy.")
    u = pos.mean(0) - neg.mean(0)
    u = u / (np.linalg.norm(u) + 1e-8)
    c0 = mu.mean(0)

    # Gender axis (for the orthogonalization / disentanglement test).
    glab = np.array([gender_label(t) for t in texts])
    gp, gn = mu[glab > 0], mu[glab < 0]
    u_g = gp.mean(0) - gn.mean(0)
    u_g = u_g / (np.linalg.norm(u_g) + 1e-8)
    cos_sg = float(u @ u_g)  # how much the raw sentiment axis is contaminated by gender
    log_for_0(f"codes mu: {mu.shape} | sent(pos={len(pos)},neg={len(neg)}) "
              f"gender(f={len(gp)},m={len(gn)}) | cos(sent,gender)={cos_sg:+.3f}")
    if args.orthogonalize:
        u = u - (u @ u_g) * u_g
        u = u / (np.linalg.norm(u) + 1e-8)
        log_for_0(f"ORTHOGONALIZED sentiment axis against gender; new cos(sent,gender)={float(u @ u_g):+.4f}")

    # --- steering sweep ---
    M = args.samples_per_alpha
    rows, results = [], []
    for a in alphas:
        c = (c0 + a * u).astype(np.float32)
        phi_vec = (c @ U) if is_m2 else c   # M2 lifts via U; M1 phi == code (d-dim)
        phi_lift = jnp.asarray(np.repeat(phi_vec[None, :], M, axis=0))  # (M, d)
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
        gtexts = [tok.decode(pred[m], skip_special_tokens=True) for m in range(M)]
        scores = [lexicon_sentiment(g) for g in gtexts]
        pos_frac = float(np.mean([s > 0 for s in scores]))
        attrs = np.array([attr_scores(g) for g in gtexts])  # (M, 3): animal, female, len
        animal_frac, female_frac, mean_len = attrs.mean(0)
        results.append((a, pos_frac, animal_frac, female_frac, mean_len))
        for g in gtexts:
            rows.append({"alpha": a, "generated": g})

    print("\n" + "=" * 64)
    print("C2 STEERING (sentiment axis)  [on-target = positive_frac]")
    print("=" * 64)
    print("  alpha  positive  | animal  female  len   (off-target, should stay flat)")
    for a, pf, af, ff, ml in results:
        print(f"  {a:+5.1f}  {pf:.3f}     |  {af:.3f}   {ff:.3f}  {ml:5.1f}")
    fracs = [r[1] for r in results]
    delta = fracs[-1] - fracs[0]
    mono = all(fracs[i + 1] >= fracs[i] - 0.05 for i in range(len(fracs) - 1))
    verdict = (delta >= 0.3) and mono
    # off-target leakage = range (max-min) of each off-target attribute across the sweep
    arr = np.array([[r[2], r[3], r[4]] for r in results])
    leak_animal, leak_female = (arr[:, 0].ptp()), (arr[:, 1].ptp())
    leak_len = arr[:, 2].ptp()
    print("\n  --- OFF-TARGET LEAKAGE (range across alpha; lower = better disentanglement) ---")
    print(f"  animal_frac range = {leak_animal:.3f} | female_frac range = {leak_female:.3f} "
          f"| length range = {leak_len:.1f}")
    print(f"\n  on-target endpoint delta = {delta:+.3f} | monotonic(+/-0.05) = {mono}")
    print(f"  VERDICT: {'PASS — interpretable, monotonic control' if verdict else 'FAIL — flat / non-monotonic'}")
    print("=" * 56 + "\n")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint_path.rstrip('/')) or '.', "steering_samples.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} steered generations to {out}.")


if __name__ == "__main__":
    main()
