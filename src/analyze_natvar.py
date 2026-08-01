#!/usr/bin/env python
"""Natural-variation hypothesis test: does the denoiser's own per-attribute
variability at FIXED code (natvar_*.json) predict which targets absorb
directed leakage (pairs8_*.json)?

Unlike target fixed effects (one constant per attribute), natural variation is
measured PER MODEL, so it can also explain model-specific deviations. Compare:
  R2(model FE)  ->  + natvar (1 dof)  ->  + target FE ceiling  ->  target FE + natvar
and per-model Spearman(natvar, leak-into) across the 8 targets.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12))


def main():
    nat = {}
    for f in sorted(glob.glob(os.path.join(REPO, "paper/natvar_*.json"))):
        tag = os.path.basename(f)[len("natvar_"):-len(".json")]
        nat[tag] = json.load(open(f))["attrs"]
    if not nat:
        sys.exit("no natvar_*.json yet")

    rows = []  # (model, src, tgt, leak)
    for f in sorted(glob.glob(os.path.join(REPO, "paper/pairs8_*.json"))):
        if "deconcore" in f:
            continue
        tag = os.path.basename(f)[len("pairs8_"):-len(".json")]
        if tag not in nat:
            continue
        d = json.load(open(f))
        for pair, v in d["pairs"].items():
            s, t = pair.split("->")
            rows.append((tag, s, t, v["logit_shift"][0]))
    models = sorted({r[0] for r in rows})
    targets = sorted({r[2] for r in rows})
    print(f"models: {models}")

    for key in ("logit_std", "flip_lex"):
        get = (lambda m, t: nat[m][t][key][0] if isinstance(nat[m][t][key], list)
               else nat[m][t][key])
        y = np.array([r[3] for r in rows])
        Xm = np.stack([np.array([r[0] == m for r in rows], float) for m in models], 1)
        xv = np.array([get(r[0], r[2]) for r in rows])
        Xt = np.stack([np.array([r[2] == t for r in rows], float) for t in targets], 1)

        def r2(X):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(), beta
        r2m, _ = r2(Xm)
        r2mv, bv = r2(np.column_stack([Xm, xv]))
        r2mt, _ = r2(np.column_stack([Xm, Xt[:, :-1]]))
        r2mtv, _ = r2(np.column_stack([Xm, Xt[:, :-1], xv]))
        print(f"\n=== predictor: {key} ===")
        print(f"R2 model FE            = {r2m:.3f}")
        print(f"+ natvar (1 dof)       = {r2mv:.3f}   slope = {bv[-1]:+.2f}   "
              f"captures {(r2mv - r2m) / max(r2mt - r2m, 1e-9):.0%} of target-FE gain")
        print(f"+ target FE ceiling    = {r2mt:.3f}")
        print(f"target FE + natvar     = {r2mtv:.3f}  (model-specific signal beyond identity)")
        print("per-model Spearman(natvar, mean leak-into):")
        for m in models:
            leak_t = {t: np.mean([r[3] for r in rows if r[0] == m and r[2] == t])
                      for t in targets}
            nv = [get(m, t) for t in targets]
            lk = [leak_t[t] for t in targets]
            print(f"  {m:<10} {spearman(nv, lk):+.2f}")


if __name__ == "__main__":
    main()
