#!/usr/bin/env python
"""Score the round-2 registered predictions (paper/fragility_registration_round2.md)
against the pairs12_*.json leakage measurements.

P1: predicted ordering water > plant > color > question vs observed (Spearman).
P2: cell-level correlation between the 28 predicted (model, new-target) values
    in round2_predictions.json and the observed leak-into.
P3: share of the target-FE gain captured by the 3-dof predictor on the full
    12-target directed-cell set.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = ["color", "plant", "water", "question"]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12))


def main():
    reg = json.load(open(os.path.join(REPO, "paper/round2_predictions.json")))
    frag = json.load(open(os.path.join(REPO, "paper/fragility_scores.json")))
    nat = {os.path.basename(f)[7:-5]: json.load(open(f))["attrs"]
           for f in glob.glob(os.path.join(REPO, "paper/natvar_*.json"))}
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "paper/pairs12_*.json"))):
        tag = os.path.basename(f)[len("pairs12_"):-len(".json")]
        d = json.load(open(f))
        for pair, v in d["pairs"].items():
            s, t = pair.split("->")
            rows.append((tag, s, t, v["logit_shift"][0]))
    if not rows:
        sys.exit("no pairs12_*.json yet")
    models = sorted({r[0] for r in rows})
    targets = sorted({r[2] for r in rows})
    print(f"models: {models} ({len(rows)} cells, {len(targets)} targets)")

    # observed mean leak-into per target (model-demeaned)
    demeaned = {}
    for t in targets:
        vals = []
        for m in models:
            mm = np.mean([r[3] for r in rows if r[0] == m])
            vals.append(np.mean([r[3] for r in rows if r[0] == m and r[2] == t]) - mm)
        demeaned[t] = float(np.mean(vals))
    print("\nobserved mean leak-into (demeaned), all 12 targets:")
    for t in sorted(targets, key=lambda t: -demeaned[t]):
        mark = " <-- new" if t in NEW else ""
        print(f"  {t:<10} {demeaned[t]:+7.3f}{mark}")

    obs_new = sorted(NEW, key=lambda t: -demeaned[t])
    pred_new = reg["predicted_order"]
    rho = spearman([pred_new.index(t) for t in NEW], [obs_new.index(t) for t in NEW])
    print(f"\nP1 predicted {' > '.join(pred_new)}")
    print(f"P1 observed  {' > '.join(obs_new)}   Spearman = {rho:.2f}")

    # P2: cell level
    po, oo = [], []
    for m in models:
        for t in NEW:
            key = f"{m}|{t}"
            if key in reg["predicted_cells"]:
                po.append(reg["predicted_cells"][key])
                oo.append(np.mean([r[3] for r in rows if r[0] == m and r[2] == t]))
    po, oo = np.array(po), np.array(oo)
    print(f"\nP2 cells n={len(po)}: Pearson {np.corrcoef(po, oo)[0, 1]:.2f}, "
          f"Spearman {spearman(po, oo):.2f}")

    # P3: variance decomposition on the full 12-target set
    y = np.array([r[3] for r in rows])
    Xm = np.stack([np.array([r[0] == m for r in rows], float) for m in models], 1)
    Xt = np.stack([np.array([r[2] == t for r in rows], float) for t in targets], 1)

    def feats(m, t):
        n = nat[m][t]["logit_std"][0]; i = frag[t]["ins_swing"][0]
        return [n, i, n * i]
    Xf = np.array([feats(r[0], r[2]) for r in rows])

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2m = r2(Xm)
    r2mf = r2(np.column_stack([Xm, Xf]))
    r2mt = r2(np.column_stack([Xm, Xt[:, :-1]]))
    print(f"\nP3 (12-target set): R2 model FE {r2m:.3f} | +natvar/ins/interaction "
          f"{r2mf:.3f} | +target FE ceiling {r2mt:.3f}")
    print(f"   predictor captures {(r2mf - r2m) / max(r2mt - r2m, 1e-9):.0%} of target-FE gain")


if __name__ == "__main__":
    main()
