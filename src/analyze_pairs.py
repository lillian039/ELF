#!/usr/bin/env python3
"""Within-k analysis: does code-space interference predict pairwise leakage?

The cross-k correlation (r=0.90) is confounded: any two quantities monotone in k
correlate. This analysis tests the superposition->leakage link WITHIN each fixed
k, across attribute pairs, where k cannot be the driver:

  1. per-k Spearman/Pearson between probe interference(a,b) and behavioral
     leakage(a,b) over the 6 unordered pairs (leakage symmetrized: mean of the
     two directions);
  2. pooled OLS with k fixed effects (leakage ~ interference + C(k)) vs the
     k-only null, F-test on the interference term;
  3. leave-one-out prediction (LOPO: hold out a pair everywhere; LOKO: hold out
     a whole k) vs baselines (k-only, data label-correlation |rho_ab|).

Inputs: paper/pairs_<tag>.json from eval_leakage_pairs.py, and the interference
matrices parsed from tinystories_demo/logs/sup_*.log (Manifold Probe output).

Usage: python3 src/analyze_pairs.py [--pairs-dir paper] [--logs tinystories_demo/logs]
"""

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np

ATTRS = ["sentiment", "gender", "animal", "length"]
PAIRS = list(itertools.combinations(ATTRS, 2))  # 6 unordered

# pairs_<tag>.json / sup log tag -> k value (M1 mean-pool kept separate from learned 512)
TAGS = {"k8": 8, "k16": 16, "k64": 64, "k256": 256, "k512": 512, "m1": "M1", "M1": "M1"}


def parse_interference_log(path):
    """4x4 interference matrix from an eval_superposition log."""
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "mean interference matrix" in ln:
            block = lines[i + 2: i + 2 + len(ATTRS)]
            M = np.array([[float(x) for x in re.findall(r"[\d.]+", row)[:len(ATTRS)]]
                          for row in block])
            return M
    return None


def load_inputs(pairs_dir, logs_dir):
    models = {}
    for pj in glob.glob(os.path.join(pairs_dir, "pairs_*.json")):
        tag = os.path.basename(pj)[len("pairs_"):-len(".json")]
        if tag not in TAGS:
            continue
        with open(pj) as f:
            models.setdefault(TAGS[tag], {})["pairs"] = json.load(f)
    for sl in glob.glob(os.path.join(logs_dir, "sup_*.log*")):
        m = re.match(r"sup_(\w+)\.log", os.path.basename(sl))
        if not m or m.group(1) not in TAGS:
            continue
        M = parse_interference_log(sl)
        if M is not None:
            models.setdefault(TAGS[m.group(1)], {})["interf"] = M
    return {k: v for k, v in models.items() if "pairs" in v and "interf" in v}


def sym_leak(pairs_json, a, b):
    """Symmetrized pairwise leakage: mean of the two directed logit-shifts."""
    vals = []
    for s, t in ((a, b), (b, a)):
        rec = pairs_json["pairs"].get(f"{s}->{t}")
        if rec:
            vals.append(rec["logit_shift"][0])
    return float(np.mean(vals)) if vals else None


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def pearson(x, y):
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float((x * y).mean())


def ols_r2(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / (ss_tot + 1e-12), beta, ss_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default="paper")
    ap.add_argument("--logs", default="tinystories_demo/logs")
    ap.add_argument("--label-corr", default="",
                    help="Optional JSON {\"a-b\": rho_ab} of data label correlations (baseline).")
    args = ap.parse_args()

    models = load_inputs(args.pairs_dir, args.logs)
    if not models:
        raise SystemExit("no model has BOTH pairs_*.json and a sup_*.log interference matrix")

    rows = []  # (model_key, pair_idx, interference, leakage)
    for mk, dat in sorted(models.items(), key=lambda kv: str(kv[0])):
        M = dat["interf"]
        for pi, (a, b) in enumerate(PAIRS):
            lk = sym_leak(dat["pairs"], a, b)
            if lk is None:
                continue
            rows.append((mk, pi, M[ATTRS.index(a), ATTRS.index(b)], lk))
    mks = sorted({r[0] for r in rows}, key=str)
    interf = np.array([r[2] for r in rows])
    leak = np.array([r[3] for r in rows])

    print("=" * 72)
    print(f"WITHIN-k ANALYSIS  models={mks}  n={len(rows)} (pair,model) points")
    print("=" * 72)

    # 1. per-k rank correlation over the 6 pairs
    print("\nper-model correlation (interference vs symmetrized leakage, 6 pairs):")
    per_k = []
    for mk in mks:
        sel = [i for i, r in enumerate(rows) if r[0] == mk]
        if len(sel) < 3:
            continue
        x, y = interf[sel], leak[sel]
        sp, pe = spearman(x, y), pearson(x, y)
        per_k.append(sp)
        pairs_str = "  ".join(f"{PAIRS[rows[i][1]][0][:4]}-{PAIRS[rows[i][1]][1][:4]}"
                              f"={interf[i]:.3f}/{leak[i]:.2f}" for i in sel)
        print(f"  k={mk!s:<4} Spearman={sp:+.3f} Pearson={pe:+.3f}   [{pairs_str}]")
    if per_k:
        print(f"  mean within-k Spearman = {np.mean(per_k):+.3f}"
              f"  (>0 in {sum(s > 0 for s in per_k)}/{len(per_k)} models)")

    # 2. pooled regression with model fixed effects
    def design(with_interf):
        cols = [np.ones(len(rows))]
        for mk in mks[1:]:
            cols.append(np.array([1.0 if r[0] == mk else 0.0 for r in rows]))
        if with_interf:
            cols.append(interf)
        return np.stack(cols, axis=1)

    r2_full, beta_full, ss_full = ols_r2(design(True), leak)
    r2_null, _, ss_null = ols_r2(design(False), leak)
    n, p_full = len(rows), len(mks) + 1
    f_stat = (ss_null - ss_full) / (ss_full / max(1, n - p_full))
    print(f"\npooled OLS with model fixed effects:")
    print(f"  leakage ~ C(model)                R2={r2_null:.3f}")
    print(f"  leakage ~ C(model) + interference R2={r2_full:.3f}"
          f"  beta_interf={beta_full[-1]:+.3f}  F(1,{n - p_full})={f_stat:.2f}")

    # 2b. directed-level regression with target fixed effects (absorbs per-target
    # classifier scale) and a no-length sensitivity check (length steering/readout
    # is the noisiest attribute).
    drows = []  # (model, src, tgt, interference, leakage)
    for mk, dat in sorted(models.items(), key=lambda kv: str(kv[0])):
        M = dat["interf"]
        for key, rec in dat["pairs"]["pairs"].items():
            s, t = key.split("->")
            drows.append((mk, s, t, M[ATTRS.index(s), ATTRS.index(t)],
                          rec["logit_shift"][0]))

    def directed_fit(rows_d, label):
        mks_d = sorted({r[0] for r in rows_d}, key=str)
        tgts = sorted({r[2] for r in rows_d})
        y = np.array([r[4] for r in rows_d])
        x = np.array([r[3] for r in rows_d])
        cols = [np.ones(len(rows_d))]
        for mk in mks_d[1:]:
            cols.append(np.array([1.0 if r[0] == mk else 0.0 for r in rows_d]))
        for t in tgts[1:]:
            cols.append(np.array([1.0 if r[2] == t else 0.0 for r in rows_d]))
        Xn = np.stack(cols, axis=1)
        Xf = np.stack(cols + [x], axis=1)
        r2n, _, ssn = ols_r2(Xn, y)
        r2f, bf, ssf = ols_r2(Xf, y)
        dof = len(rows_d) - Xf.shape[1]
        f = (ssn - ssf) / (ssf / max(1, dof))
        print(f"  {label:<26} R2 {r2n:.3f} -> {r2f:.3f}"
              f"  beta_interf={bf[-1]:+.3f}  F(1,{dof})={f:.2f}")

    print("\ndirected-level OLS, fixed effects C(model)+C(target):")
    directed_fit(drows, "all 12 directed pairs")
    nolen = [r for r in drows if "length" not in (r[1], r[2])]
    directed_fit(nolen, "excluding length pairs")

    # 3. held-out prediction: LOPO (leave one pair out) and LOKO (leave one model out)
    def cv_rmse(groups):
        errs_model, errs_konly = [], []
        for g in sorted(set(groups)):
            tr = [i for i, gg in enumerate(groups) if gg != g]
            te = [i for i, gg in enumerate(groups) if gg == g]
            for te_i in te:
                # model: per-model intercept + global interference slope (fit on train)
                Xtr = np.stack([np.ones(len(tr)), interf[tr]], axis=1)
                _, b, _ = ols_r2(Xtr, leak[tr])
                errs_model.append(leak[te_i] - (b[0] + b[1] * interf[te_i]))
                errs_konly.append(leak[te_i] - leak[tr].mean())
        return (float(np.sqrt(np.mean(np.square(errs_model)))),
                float(np.sqrt(np.mean(np.square(errs_konly)))))

    lopo = cv_rmse([r[1] for r in rows])
    loko = cv_rmse([str(r[0]) for r in rows])
    print(f"\nheld-out prediction RMSE (interference model vs grand-mean baseline):")
    print(f"  leave-one-PAIR-out : {lopo[0]:.3f} vs baseline {lopo[1]:.3f}")
    print(f"  leave-one-MODEL-out: {loko[0]:.3f} vs baseline {loko[1]:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
