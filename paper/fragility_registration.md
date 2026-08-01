# Registered prediction: target fragility -> directed leakage (2026-07-20 01:50 UTC)

Written BEFORE any extended-target leakage number existed. The extended
directed-pair evals (paper/pairs8_*.json, logs tinystories_demo/logs/pairs8_*)
were launched ~01:37-01:45 and take hours per model; at registration time none
had produced a LEAKPAIR_SUMMARY line. Fragility scores: paper/fragility_scores.json
(eval_fragility.py, mtime 01:42), computed only from real stories + the frozen
encoder, touching no SM-ELF model and no leakage measurement.

## Candidate selection (in-sample, 4 core attributes)

Known target effects from the paper's 72-cell analysis: leakage flows INTO
animal (+0.21) and gender (+0.19), AWAY from length (-0.14) and sentiment (-0.26).

| candidate | definition | core-4 ordering | vs known (Spearman) |
|---|---|---|---|
| top1_frac | largest single-deletion margin drop / margin | sent > animal > length > gender | -0.4 (rejected) |
| del_abs | largest single-deletion drop, calibrated units | sent > animal > length > gender | -0.4 (rejected) |
| **ins_swing** | largest single-word-insertion pull, calibrated units | **animal .098 > gender .084 > sent .059 > length -.000** | **+0.8, absorb/resist split exact (selected)** |

Interpretation: leakage lands on targets whose classifier reading one incidental
token can move (insertion movability), not on targets whose evidence is
redundant (deletion robustness measures the opposite thing; gender is the most
redundant attribute and the deletion scores anti-predict).

## Registered out-of-sample prediction (4 new attributes)

ins_swing: weather 0.165 > food 0.133 > dialogue 0.116 > vehicle 0.096.

P1 (ordering): mean directed leakage INTO the new targets (averaged over the 4
core sources and over models, logit-shift) decreases in the order
  weather > food > dialogue > vehicle.

P2 (absorb/resist split): all four new targets score in or above the
gender/animal fragility band (>= 0.096 vs gender 0.084), so all four should be
ABSORBERS: mean leakage into each of them should exceed mean leakage into
sentiment and into length in the same models.

P3 (quantitative): pooled over all 8 targets x models, target-mean leakage
regressed on ins_swing should be positive and should recover a substantial part
of the R^2 = 0.52 - 0.24 = 0.28 that target fixed effects add over model fixed
effects in the original analysis.

Falsifiers: vehicle leaking most among the new four (P1), any new target
absorbing less than sentiment/length (P2), or a flat/negative ins_swing slope (P3).
