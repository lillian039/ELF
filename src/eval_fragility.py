#!/usr/bin/env python
"""Target fragility: an independent, data-side token-locality score per attribute.

Motivation: within a model, which directed pairs leak is set by the TARGET
attribute (paper Sec. superpos: target fixed effects R2 0.24 -> 0.52), with
token-local targets (gender, animal) absorbing leakage that globally
distributed ones (tone, length) resist. This script turns that post-hoc
observation into a PRE-REGISTERED predictor: a fragility score computed only
from real stories and the frozen encoder, never touching any SM-ELF model or
leakage measurement.

Definition. For attribute a, fit the same held-out embedding-space linear
classifier used by the leakage protocol on a fit split. On a disjoint pool of
correctly-classified stories, delete one word at a time, re-embed, and measure
the drop in the signed classification margin. Per story:
    top1_frac = (m_full - min_j m_(-j)) / m_full   (largest single-word drop)
    flip1     = 1[ min_j m_(-j) < 0 ]              (one word flips the label)
fragility(a) = mean over stories (and over bootstrap classifier seeds).
An attribute whose evidence is concentrated in single tokens (a pronoun, an
animal word) scores high; one carried diffusely (tone, length) scores low.

Prediction (registered before the extended leakage eval is run): directed
leakage INTO target b, averaged over models and sources, increases with
fragility(b).

CPU-light, encoder-only; no denoiser checkpoint needed.

Usage:
  python3 src/eval_fragility.py --config <any config.yml> \
      [--fit-stories 400] [--loo-stories 120] [--seeds 5] \
      [--out paper/fragility_scores.json]
"""

import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint
from utils.data_utils import load_dataset_split, get_pad_token_id
from configs.config import load_config_from_yaml, apply_config_overrides

from eval_leakage_pairs import ATTRS, EXTRA_ATTRS, EXTRA2_ATTRS, attr_value, pos_neg_masks
from eval_leakage_continuous import _embed_texts, _fit_linear_classifier

TARGETS = ATTRS + EXTRA_ATTRS + EXTRA2_ATTRS

# Canonical single-word insertions per attribute (word, class_sign): how much
# can ONE incidental token, dropped into an arbitrary story, move this
# attribute's classifier reading? This models what a denoiser does when it
# leaks: it changes a token or two, it does not rewrite the story.
INSERT_WORDS = {
    "sentiment": [("happy", 1), ("sad", -1)],
    "gender": [("she", 1), ("he", -1)],
    "animal": [("dog", 1), ("bird", 1)],
    "length": [("the", 1)],
    "food": [("cake", 1), ("apple", 1)],
    "weather": [("rain", 1), ("sunny", 1)],
    "vehicle": [("car", 1), ("train", 1)],
    "dialogue": [('"', 1), ("said", 1)],
    "color": [("red", 1), ("blue", 1)],
    "plant": [("tree", 1), ("flower", 1)],
    "water": [("water", 1), ("river", 1)],
    "question": [("?", 1)],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fit-stories", type=int, default=400,
                   help="stories used to fit the per-attribute classifiers")
    p.add_argument("--loo-stories", type=int, default=120,
                   help="disjoint stories used for leave-one-word-out scoring")
    p.add_argument("--max-words", type=int, default=150,
                   help="cap on deletions per story (first max-words words)")
    p.add_argument("--seeds", type=int, default=5,
                   help="bootstrap seeds for classifier fitting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="paper/fragility_scores.json")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    L = cfg.max_length
    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)

    val = load_dataset_split(cfg.eval_data_path)
    n_fit, n_loo = args.fit_stories, args.loo_stories
    assert len(val) >= n_fit + n_loo, f"need {n_fit + n_loo} stories, have {len(val)}"
    texts = [tok.decode(np.asarray(val[i]["input_ids"]), skip_special_tokens=True)
             for i in range(n_fit + n_loo)]
    fit_texts, loo_texts = texts[:n_fit], texts[n_fit:]

    log_for_0(f"embedding {n_fit} fit stories")
    fit_emb = _embed_texts(fit_texts, tok, L, pad_id, enc_model, enc_params, cfg)
    fit_lab = {a: np.array([attr_value(a, t) for t in fit_texts]) for a in TARGETS}

    # --- leave-one-word-out variants, shared across all attributes ---
    variants, spans = [], []  # spans[i] = (start, n_variants) into `variants` for story i
    for t in loo_texts:
        words = t.split()
        n = min(len(words), args.max_words)
        spans.append((len(variants), n))
        for j in range(n):
            variants.append(" ".join(words[:j] + words[j + 1:]))
    log_for_0(f"embedding {len(loo_texts)} LOO stories -> {len(variants)} deletion variants")
    loo_full_emb = _embed_texts(loo_texts, tok, L, pad_id, enc_model, enc_params, cfg)
    var_emb = _embed_texts(variants, tok, L, pad_id, enc_model, enc_params, cfg)
    loo_lab = {a: np.array([attr_value(a, t) for t in loo_texts]) for a in TARGETS}

    # --- single-word insertion variants (mid-story) per attribute ---
    ins_emb = {}
    for a in TARGETS:
        for wi, (word, _sgn) in enumerate(INSERT_WORDS[a]):
            txts = []
            for t in loo_texts:
                ws = t.split()
                mid = len(ws) // 2
                txts.append(" ".join(ws[:mid] + [word] + ws[mid:]))
            ins_emb[(a, wi)] = _embed_texts(txts, tok, L, pad_id,
                                            enc_model, enc_params, cfg)

    rng = np.random.default_rng(args.seed)
    out = {}
    print("\n" + "=" * 74)
    print(f"TARGET FRAGILITY  fit={n_fit} loo={len(loo_texts)} seeds={args.seeds}")
    print("  top1_frac/flip1: deletion-LOO (evidence redundancy, relative)")
    print("  del_abs: largest single-deletion margin drop (calibrated units)")
    print("  ins_swing: largest single-word-insertion pull (calibrated units)")
    print("=" * 74)
    print(f"{'attribute':<12} {'top1_frac':>15} {'flip1':>7} {'del_abs':>14} "
          f"{'ins_swing':>14} {'n':>5}")
    for a in TARGETS:
        # pool-story signs from the SAME tercile/sign rule as the leakage protocol
        pos_l, neg_l = pos_neg_masks(a, loo_lab[a])
        sign = np.where(pos_l, 1.0, np.where(neg_l, -1.0, 0.0))
        t1_seeds, fl_seeds, da_seeds, ins_seeds, n_used = [], [], [], [], []
        for si in range(args.seeds):
            idx = (rng.integers(0, n_fit, size=n_fit) if args.seeds > 1
                   else np.arange(n_fit))
            lab_s = fit_lab[a][idx]
            pos, neg = pos_neg_masks(a, lab_s)
            keep = pos | neg
            if pos.sum() == 0 or neg.sum() == 0:
                continue
            w, b = _fit_linear_classifier(fit_emb[idx][keep],
                                          np.where(pos, 1, -1)[keep])
            m_full = sign * (loo_full_emb @ w - b)  # signed margin, >0 = correct
            m_var = var_emb @ w - b
            t1, fl, da = [], [], []
            for i, (s0, nv) in enumerate(spans):
                if sign[i] == 0 or m_full[i] <= 1e-6 or nv == 0:
                    continue
                m_del = sign[i] * m_var[s0:s0 + nv]
                worst = float(m_del.min())
                t1.append(min(max((m_full[i] - worst) / m_full[i], 0.0), 1.5))
                fl.append(1.0 if worst < 0 else 0.0)
                da.append(m_full[i] - worst)
            # insertion pull: every pool story, max over canonical words of the
            # signed movement toward the inserted word's class
            l_orig = loo_full_emb @ w - b
            pulls = []
            for wi, (_word, sgn) in enumerate(INSERT_WORDS[a]):
                l_ins = ins_emb[(a, wi)] @ w - b
                pulls.append(sgn * (l_ins - l_orig))
            ins_seeds.append(float(np.max(np.stack(pulls), axis=0).mean()))
            if t1:
                t1_seeds.append(float(np.mean(t1)))
                fl_seeds.append(float(np.mean(fl)))
                da_seeds.append(float(np.mean(da)))
                n_used.append(len(t1))

        def ms(v):
            return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)
        (t1m, t1s), (flm, _), (dam, das), (inm, ins_) = (
            ms(t1_seeds), ms(fl_seeds), ms(da_seeds), ms(ins_seeds))
        out[a] = {"top1_frac": [t1m, t1s], "flip1_rate": [flm, 0.0],
                  "del_abs": [dam, das], "ins_swing": [inm, ins_],
                  "n_stories": int(np.mean(n_used)) if n_used else 0}
        print(f"{a:<12} {t1m:>7.3f}+-{t1s:<6.3f} {flm:>7.3f} {dam:>7.3f}+-{das:<5.3f} "
              f"{inm:>7.3f}+-{ins_:<5.3f} {out[a]['n_stories']:>5}")
        print(f"FRAGILITY_SUMMARY attr={a} top1_frac={t1m:.4f} flip1={flm:.4f} "
              f"del_abs={dam:.4f} ins_swing={inm:.4f} n={out[a]['n_stories']}")
    print("=" * 74)
    for key in ("top1_frac", "del_abs", "ins_swing"):
        order = sorted((a for a in TARGETS if not np.isnan(out[a][key][0])),
                       key=lambda a: -out[a][key][0])
        print(f"RANKING {key}: " + " > ".join(order))
        out[f"_ranking_{key}"] = order
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        log_for_0(f"wrote {args.out}")


if __name__ == "__main__":
    main()
