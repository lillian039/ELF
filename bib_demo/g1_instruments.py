#!/usr/bin/env python3
"""G1 instrument checks for the Bias in Bios replication (no ELF checkpoint
needed; see paper/bib_registration.md, gate G1 item 2).

1. Lexicon gender vs the dataset's structured gender field (non-tie bios).
2. Lexicon occ-domain (HEALTH/CREATIVE) vs structured profession group,
   restricted to bios whose profession is in one of the two groups.
3. Held-out gender classifier: difference-of-means direction in the frozen
   T5 embedding space (mean-pooled), fit on train_meta bios, AUC on val_meta.

Thresholds: agreements >= 0.90, AUC >= 0.90.

Usage: python3 bib_demo/g1_instruments.py --config bib_demo/train_bib_SM-ELF-M1.yml
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import jax.numpy as jnp  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from modules.t5_encoder import get_encoder  # noqa: E402
from utils.checkpoint_utils import load_encoder_checkpoint  # noqa: E402
from utils.encoder_utils import encode_text  # noqa: E402
from utils.data_utils import get_pad_token_id  # noqa: E402
from utils.semantic_utils import compute_phi  # noqa: E402
from configs.config import load_config_from_yaml  # noqa: E402

from lexicons_bib import bib_lexicon_labels  # noqa: E402

HEALTH_PROFS = {"physician", "nurse", "psychologist", "dentist", "surgeon",
                "dietitian", "chiropractor"}
CREATIVE_PROFS = {"photographer", "journalist", "painter", "model", "poet",
                  "filmmaker", "composer", "comedian", "dj", "rapper",
                  "interior_designer"}


def pooled_embed(texts, tok, enc_model, enc_params, cfg, batch=64):
    outs = []
    for s in range(0, len(texts), batch):
        chunk = texts[s: s + batch]
        enc = tok(chunk, padding="max_length", truncation=True,
                  max_length=cfg.max_length, return_tensors="np",
                  add_special_tokens=False)
        ids = jnp.asarray(enc["input_ids"])
        mask = jnp.asarray(enc["attention_mask"])
        x0 = encode_text(ids, mask, enc_model.apply, enc_params,
                         cfg.latent_mean, cfg.latent_std)
        outs.append(np.asarray(compute_phi(x0, mask)[:, 0, :]))
    return np.concatenate(outs, axis=0)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    npos = labels.sum(); nneg = len(labels) - npos
    return (ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="bib_demo/train_bib_SM-ELF-M1.yml")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=2000)
    args = p.parse_args()

    cfg = load_config_from_yaml(args.config)
    train_meta = load_from_disk(os.path.join(REPO, "bib_demo/data_50k/train_meta"))
    val_meta = load_from_disk(os.path.join(REPO, "bib_demo/data_50k/val_meta"))

    # --- 1. lexicon gender vs structured field ---
    lex = [bib_lexicon_labels(t) for t in train_meta["text"][:20000]]
    gold_g = np.array([1 if g == "female" else -1
                       for g in train_meta["gender"][:20000]])
    lg = np.array([g for _, g in lex])
    nz = lg != 0
    agree_g = float((lg[nz] == gold_g[nz]).mean())
    print(f"[1] lexicon gender vs field: agreement={agree_g:.3f} "
          f"coverage={nz.mean():.3f} (n={nz.sum()})  "
          f"{'PASS' if agree_g >= 0.90 else 'FAIL'} (>=0.90)")

    # --- 2. lexicon occ-domain vs profession group ---
    profs = train_meta["profession"][:20000]
    in_grp = np.array([pr in HEALTH_PROFS or pr in CREATIVE_PROFS for pr in profs])
    gold_o = np.array([1 if pr in HEALTH_PROFS else -1 for pr in profs])
    lo = np.array([o for o, _ in lex])
    m = in_grp & (lo != 0)
    agree_o = float((lo[m] == gold_o[m]).mean())
    print(f"[2] lexicon occ-domain vs profession group: agreement={agree_o:.3f} "
          f"coverage-in-group={(m.sum() / max(in_grp.sum(), 1)):.3f} (n={m.sum()})  "
          f"{'PASS' if agree_o >= 0.90 else 'FAIL'} (>=0.90)")

    # --- 3. frozen-space gender classifier AUC (train -> val) ---
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    get_pad_token_id(tok, cfg.pad_token)  # parity with training tokenization
    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)

    tr_texts = train_meta["text"][: args.n_train]
    tr_lab = np.array([g == "female" for g in train_meta["gender"][: args.n_train]])
    va_texts = val_meta["text"][: args.n_val]
    va_lab = np.array([g == "female" for g in val_meta["gender"][: args.n_val]])

    tr_emb = pooled_embed(tr_texts, tok, enc_model, enc_params, cfg)
    va_emb = pooled_embed(va_texts, tok, enc_model, enc_params, cfg)
    u = tr_emb[tr_lab].mean(0) - tr_emb[~tr_lab].mean(0)
    u /= (np.linalg.norm(u) + 1e-8)
    dom_auc = auc(va_emb @ u, va_lab)
    print(f"[3a] diff-of-means gender classifier: AUC={dom_auc:.3f} "
          f"(train n={len(tr_texts)}, val n={len(va_texts)})  "
          f"{'PASS' if dom_auc >= 0.90 else 'FAIL'} (>=0.90)")
    # Stronger held-out LINEAR readout (still frozen space, still linear):
    # logistic regression. Instrument amendment path when diff-of-means is
    # underpowered on this corpus; see bib_registration.md.
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(tr_emb, tr_lab)
    lr_auc = auc(clf.decision_function(va_emb), va_lab)
    print(f"[3b] logistic-regression gender classifier: AUC={lr_auc:.3f}  "
          f"{'PASS' if lr_auc >= 0.90 else 'FAIL'} (>=0.90)")


if __name__ == "__main__":
    main()
