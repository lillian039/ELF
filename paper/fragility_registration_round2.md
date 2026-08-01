# Round-2 registered predictions: natvar x movability -> leak-into (2026-07-20 02:44 UTC)

Written BEFORE any leakage number for {color, plant, water, question} existed
(their first pairs8 eval launches after this file; scores in
paper/round2_predictions.json, model fit R2=0.459 on the 196 known cells).

Predictors, both computed with no access to the new targets' leakage:
- natvar logit_std: per-model denoiser natural variation at fixed code
  (eval_natural_variation.py, 12-attribute rerun, mtime 02:38)
- ins_swing: single-word insertion movability (eval_fragility.py v3, 02:30)
- fitted model: leak ~ model FE + a*natvar + b*ins + c*natvar*ins
  (a=+4.84, b=+4.33, c=-26.5)

## Registered predictions

P1 (ordering among new targets, mean over models):
  water > plant > color > question
  (identical under natvar-only ranking and under the full interaction model)

P2 (cell-level): across the 28 new (model, target) cells, predicted values in
round2_predictions.json should correlate positively with observed leak-into
(report Pearson & Spearman; success = clearly positive, e.g. r >= 0.4).

P3 (variance capture): adding the fitted 3-dof predictor to model FE should
capture a similar share of the target-FE gain on the 12-target dataset as it
did in-sample on 8 targets (~45%); report the exact number.

Falsifiers: question absorbing most; flat or negative cell-level correlation;
predictor share collapsing toward 0 on the enlarged target set.

## VERDICT (2026-07-20 03:55 UTC, after pairs12 evals)

P1: observed plant > water > color > question (top-two adjacent swap), Spearman 0.80. PASS.
P2: n=28 cells, Pearson 0.87, Spearman 0.82 (registered line: >= 0.4). PASS.
P3: 41% of target-FE gain on the 12-target set (45% in-sample). PASS.
Full numbers: src/analyze_round2.py output; observed plant +0.137 and water +0.117
absorb at gender-level (+0.128).
