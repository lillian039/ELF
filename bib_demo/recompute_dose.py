#!/usr/bin/env python3
"""Recompute the per-alpha dose-response curves from dose_*.jsonl generations.

The hogfather run's stdout table was lost to a grep filter; the metrics are
pure lexicon counts (eval_steering.lexicon_sentiment / attr_scores with the
BiB lexicons swapped in), so the curves are exactly reproducible from the
saved generations. Writes paper/bib_dose_response.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from lexicons_bib import HEALTH, CREATIVE, FEM, MAL  # noqa: E402
import eval_steering as es  # noqa: E402

es.POS_WORDS, es.NEG_WORDS = HEALTH, CREATIVE
es.FEMALE_WORDS, es.MALE_WORDS = FEM, MAL

out = {}
for tag in ["bib_m1_s1", "bib_k16_s1", "bib_k64_s1", "bib_k256_s1"]:
    with open(os.path.join(HERE, "logs", f"dose_{tag}.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    by_alpha = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r["generated"])
    curve = []
    for a in sorted(by_alpha):
        gtexts = by_alpha[a]
        pos_frac = float(np.mean([es.lexicon_sentiment(g) > 0 for g in gtexts]))
        attrs = np.array([es.attr_scores(g) for g in gtexts])
        curve.append({"alpha": a, "health_frac": pos_frac,
                      "female_frac": float(attrs[:, 1].mean()),
                      "mean_len": float(attrs[:, 2].mean()), "n": len(gtexts)})
    fracs = [c["health_frac"] for c in curve]
    delta = fracs[-1] - fracs[0]
    mono = all(fracs[i + 1] >= fracs[i] - 0.05 for i in range(len(fracs) - 1))
    fem = [c["female_frac"] for c in curve]
    out[tag] = {"curve": curve, "endpoint_delta": delta, "monotonic": mono,
                "verdict": "PASS" if (delta >= 0.3 and mono) else "FAIL",
                "female_range": float(np.ptp(fem)),
                "len_range": float(np.ptp([c["mean_len"] for c in curve]))}
    line = "  ".join(f"{c['alpha']:+.0f}:{c['health_frac']:.2f}" for c in curve)
    print(f"{tag:13s} delta={delta:+.3f} mono={mono} fem_range={out[tag]['female_range']:.3f}")
    print(f"   health_frac  {line}")
    fline = "  ".join(f"{c['alpha']:+.0f}:{c['female_frac']:.2f}" for c in curve)
    print(f"   female_frac  {fline}")

dst = os.path.join(os.path.dirname(HERE), "paper", "bib_dose_response.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", dst)
