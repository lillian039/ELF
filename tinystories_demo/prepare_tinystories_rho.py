#!/usr/bin/env python3
"""Prepare TinyStories datasets with CONTROLLED sentiment-gender correlation.

For the k x rho mechanism experiment: build training sets that are identical in
size, source corpus, tokenization, and attribute marginals, differing ONLY in
the joint distribution (phi coefficient rho) of the lexicon sentiment and
protagonist-gender labels.

Method: label a large pool with the shared lexicons (utils/semantic_utils),
split it into the four jointly-labeled sign cells (s,g) in {+,-}^2 plus a
"rest" bucket (either label neutral). Keep the natural labeled/rest ratio and
the natural per-attribute marginals; set the cell quotas to
    p(s,g) = p_s p_g + s*g * t,   t = rho * sqrt(p_s(1-p_s) p_g(1-p_g)),
and quota-sample without replacement. EVERY rho variant (including the natural
one) goes through this same pipeline, so "natural" is a proper control.

Usage:
  # report the pool's natural rho, cell counts, achievable rho range:
  python3 tinystories_demo/prepare_tinystories_rho.py --stats-only
  # build datasets:
  python3 tinystories_demo/prepare_tinystories_rho.py --rhos 0.0,nat,0.5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from utils.semantic_utils import lexicon_labels


def parse_args():
    p = argparse.ArgumentParser(description="rho-controlled TinyStories for the k x rho experiment")
    p.add_argument("--output-root", default="tinystories_demo")
    p.add_argument("--tokenizer-name", default="t5-small")
    p.add_argument("--pool-size", type=int, default=600_000,
                   help="Stories drawn from the raw corpus before quota sampling.")
    p.add_argument("--train-size", type=int, default=50_000)
    p.add_argument("--val-size", type=int, default=2_000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rhos", type=str, default="0.0,nat,0.5",
                   help="Comma list of target phi coefficients; 'nat' = pool's natural rho.")
    p.add_argument("--stats-only", action="store_true",
                   help="Only print pool statistics and the achievable rho range.")
    return p.parse_args()


def label_pool(texts):
    """(s, g) sign labels per story; cells: 0..3 = (++, +-, -+, --), -1 = rest."""
    s = np.empty(len(texts), dtype=np.int8)
    g = np.empty(len(texts), dtype=np.int8)
    for i, t in enumerate(texts):
        si, gi = lexicon_labels(t)
        s[i], g[i] = si, gi
    cell = np.full(len(texts), -1, dtype=np.int8)
    lab = (s != 0) & (g != 0)
    cell[lab] = (s[lab] < 0) * 2 + (g[lab] < 0)
    return s, g, cell


def phi_coefficient(cell_counts):
    """Phi coefficient of the 2x2 table [[n_pp, n_pm], [n_mp, n_mm]]."""
    n_pp, n_pm, n_mp, n_mm = [float(c) for c in cell_counts]
    n = n_pp + n_pm + n_mp + n_mm
    ps, pg = (n_pp + n_pm) / n, (n_pp + n_mp) / n
    denom = np.sqrt(ps * (1 - ps) * pg * (1 - pg))
    return ((n_pp / n) - ps * pg) / (denom + 1e-12), ps, pg


def cell_quotas(n_lab, ps, pg, rho):
    """Cell counts for n_lab labeled stories at target rho, natural marginals."""
    t = rho * np.sqrt(ps * (1 - ps) * pg * (1 - pg))
    probs = np.array([ps * pg + t, ps * (1 - pg) - t, (1 - ps) * pg - t, (1 - ps) * (1 - pg) + t])
    if (probs < 0).any():
        raise ValueError(f"rho={rho} infeasible with marginals ps={ps:.3f} pg={pg:.3f}")
    q = np.floor(probs * n_lab).astype(int)
    q[0] += n_lab - q.sum()  # distribute rounding remainder
    return q


def max_feasible_rho(avail, n_lab, ps, pg, sign=+1, tol=1e-3):
    """Largest |rho| (signed) whose quotas fit the available cell counts."""
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        try:
            q = cell_quotas(n_lab, ps, pg, sign * mid)
            ok = (q <= avail).all() and (q >= 0).all()
        except ValueError:
            ok = False
        lo, hi = (mid, hi) if ok else (lo, mid)
    return sign * lo


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"loading TinyStories pool ({args.pool_size} stories)...")
    raw = load_dataset("roneneldan/TinyStories", split="train")
    raw = raw.shuffle(seed=args.seed)
    pool_n = min(args.pool_size, len(raw))
    texts = [raw[i]["text"] for i in range(pool_n)]
    s, g, cell = label_pool(texts)

    avail = np.array([(cell == c).sum() for c in range(4)])
    n_rest_pool = int((cell == -1).sum())
    rho_nat, ps, pg = phi_coefficient(avail)
    frac_lab = avail.sum() / pool_n
    n_total = args.train_size + args.val_size
    n_lab = int(round(frac_lab * n_total))
    n_rest = n_total - n_lab

    print(f"pool: {pool_n} stories | labeled(s,g both !=0): {avail.sum()} ({frac_lab:.1%}) "
          f"cells(++/+-/-+/--)={avail.tolist()} rest={n_rest_pool}")
    print(f"natural rho={rho_nat:+.3f}  P(s=+)={ps:.3f}  P(g=+)={pg:.3f}")
    lo = max_feasible_rho(avail, n_lab, ps, pg, sign=-1)
    hi = max_feasible_rho(avail, n_lab, ps, pg, sign=+1)
    print(f"achievable rho range for {n_lab} labeled of {n_total} total: [{lo:+.3f}, {hi:+.3f}]")
    if args.stats_only:
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    cell_idx = [rng.permutation(np.flatnonzero(cell == c)) for c in range(4)]
    rest_idx = rng.permutation(np.flatnonzero(cell == -1))
    if n_rest > len(rest_idx):
        sys.exit(f"pool too small: need {n_rest} rest stories, have {len(rest_idx)}")

    for spec in args.rhos.split(","):
        rho = rho_nat if spec.strip() == "nat" else float(spec)
        tag = "nat" if spec.strip() == "nat" else f"{rho:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
        q = cell_quotas(n_lab, ps, pg, rho)
        if (q > avail).any():
            print(f"SKIP rho={rho:+.3f}: quotas {q.tolist()} exceed pool cells {avail.tolist()}")
            continue
        # quota-sample cells + natural rest; each rho uses the same shuffled pools
        # (prefixes), so variants share as many stories as the quotas allow.
        take = np.concatenate([cell_idx[c][: q[c]] for c in range(4)] + [rest_idx[:n_rest]])
        take = rng.permutation(take)
        tr, va = take[: args.train_size], take[args.train_size: args.train_size + args.val_size]

        # verify achieved rho on the train split
        ach = np.array([((s[tr] > 0) & (g[tr] > 0)).sum(), ((s[tr] > 0) & (g[tr] < 0)).sum(),
                        ((s[tr] < 0) & (g[tr] > 0)).sum(), ((s[tr] < 0) & (g[tr] < 0)).sum()])
        rho_ach, _, _ = phi_coefficient(ach)

        out = Path(args.output_root) / f"data_50k_rho_{tag}"
        for name, idxs in (("train", tr), ("val", va)):
            recs = []
            for i in idxs:
                ids = tok(texts[i], add_special_tokens=False)["input_ids"][: args.max_length]
                recs.append({"input_ids": ids})
            Dataset.from_list(recs).save_to_disk(str(out / name))
        print(f"rho={rho:+.3f} (achieved {rho_ach:+.3f}) cells={ach.tolist()} -> {out}")


if __name__ == "__main__":
    main()
