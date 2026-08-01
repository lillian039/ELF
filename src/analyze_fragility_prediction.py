#!/usr/bin/env python
"""Test the registered fragility predictions (paper/fragility_registration.md).

Reads paper/fragility_scores.json (ins_swing per attribute) and the extended
directed-pair results paper/pairs8_*.json (4 core sources x 7 extended targets
per model), and evaluates:

  P1: ordering of mean leakage INTO the 4 new targets vs registered order
      weather > food > dialogue > vehicle (Spearman).
  P2: absorb/resist split: each new target's mean leakage > leakage into
      sentiment and into length within the same models.
  P3: pooled regression of target-demeaned leakage on ins_swing across all 8
      targets (slope, R^2 added over model fixed effects), mirroring the
      original 72-cell variance decomposition.

CPU-only. Run any time; uses whichever pairs8_*.json exist.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = ["food", "weather", "vehicle", "dialogue"]
CORE = ["sentiment", "gender", "animal", "length"]
REGISTERED = ["weather", "food", "dialogue", "vehicle"]  # predicted descending


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12))


def main():
    frag = json.load(open(os.path.join(REPO, "paper/fragility_scores.json")))
    ins = {a: frag[a]["ins_swing"][0] for a in CORE + NEW}

    files = sorted(glob.glob(os.path.join(REPO, "paper/pairs8_*.json")))
    files = [f for f in files if "deconcore" not in f]
    if not files:
        sys.exit("no pairs8_*.json yet")
    # leak[model][tgt] = list over sources of logit_shift
    leak = {}
    for f in files:
        d = json.load(open(f))
        tag = os.path.basename(f)[len("pairs8_"):-len(".json")]
        m = {}
        for pair, v in d["pairs"].items():
            src, tgt = pair.split("->")
            m.setdefault(tgt, []).append(v["logit_shift"][0])
        leak[tag] = {t: float(np.mean(v)) for t, v in m.items()}
        print(f"loaded {tag}: {len(d['pairs'])} directed cells")

    models = sorted(leak)
    targets = [t for t in CORE + NEW if all(t in leak[m] for m in models)]
    print(f"\nmodels: {models}\ntargets with full coverage: {targets}")

    # mean leakage into each target, per model and pooled (model-demeaned)
    print(f"\n{'target':<10} {'ins_swing':>10} {'mean leak-into (demeaned)':>26}")
    demeaned = {}
    for t in targets:
        vals = []
        for m in models:
            mm = np.mean([leak[m][u] for u in targets])
            vals.append(leak[m][t] - mm)
        demeaned[t] = float(np.mean(vals))
    for t in sorted(targets, key=lambda t: -demeaned[t]):
        print(f"{t:<10} {ins[t]:>10.4f} {demeaned[t]:>26.4f}")

    newt = [t for t in NEW if t in targets]
    if len(newt) == 4:
        obs_order = sorted(newt, key=lambda t: -demeaned[t])
        rho1 = spearman([REGISTERED.index(t) for t in newt],
                        [obs_order.index(t) for t in newt])
        print(f"\nP1 registered {' > '.join(REGISTERED)}")
        print(f"P1 observed   {' > '.join(obs_order)}   Spearman = {rho1:.2f}")
        p2 = all(demeaned[t] > demeaned["sentiment"] and demeaned[t] > demeaned["length"]
                 for t in newt)
        print(f"P2 all new targets absorb more than sentiment & length: {p2}")

    # P3: variance decomposition, directed cells, model FE vs model FE + ins_swing
    rows = []
    for f in files:
        d = json.load(open(f))
        tag = os.path.basename(f)[len("pairs8_"):-len(".json")]
        for pair, v in d["pairs"].items():
            src, tgt = pair.split("->")
            if tgt in targets:
                rows.append((tag, tgt, v["logit_shift"][0]))
    y = np.array([r[2] for r in rows])
    Xm = np.stack([np.array([r[0] == m for r in rows], float) for m in models], 1)
    xf = np.array([ins[r[1]] for r in rows])

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_m = r2(Xm)
    r2_mf = r2(np.column_stack([Xm, xf]))
    # target fixed effects ceiling
    Xt = np.stack([np.array([r[1] == t for r in rows], float) for t in targets], 1)
    r2_mt = r2(np.column_stack([Xm, Xt[:, :-1]]))
    beta, *_ = np.linalg.lstsq(np.column_stack([Xm, xf]), y, rcond=None)
    print(f"\nP3 directed cells n={len(rows)}")
    print(f"   R2 model FE only         = {r2_m:.3f}")
    print(f"   + ins_swing (1 dof)      = {r2_mf:.3f}   slope = {beta[-1]:+.2f}")
    print(f"   + target FE ({len(targets)-1} dof) ceiling = {r2_mt:.3f}")
    print(f"   fraction of target-FE gain captured by ins_swing: "
          f"{(r2_mf - r2_m) / max(r2_mt - r2_m, 1e-9):.2f}")


if __name__ == "__main__":
    main()
