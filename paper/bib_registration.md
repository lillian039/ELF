# Bias in Bios replication: acceptance criteria and preregistration

Written 2026-07-21, BEFORE any Bias in Bios model was trained. Purpose:
second-corpus replication of the SM-ELF findings (paper "Semantic Bottlenecks
Backfire"), addressing the TinyStories-only limitation. Corpus:
LabHC/bias_in_bios `hard_text` (professional biographies, first sentence
removed), 50k train / 2k val, tokenized t5-small, max 256 tokens, prepared by
`bib_demo/prepare_bias_in_bios.py` (seed 42).

All hyperparameters are frozen to the TinyStories configs
(`bib_demo/train_bib_SM-ELF-{M1,M2}.yml`); nothing is tuned per-corpus.
Training seeds: 42 (s1), 43 (s2). Evaluation seeds: 5 bootstrap/noise seeds
inside the eval scripts, as on TinyStories. Training and evaluation seeds are
distinct concepts and are never pooled.

## Attribute protocol

- Steered source: occupation macro-domain axis, labels from the dataset's
  structured `profession` field for real bios, `lexicons_bib.HEALTH/CREATIVE`
  readout for generated text.
  FINAL GROUPING (frozen 2026-07-21 from the prep-script crosstab, before any
  training):
    HEALTH   = {physician, nurse, psychologist, dentist, surgeon, dietitian,
                chiropractor}: n=14,213 (28.4%), fem_frac=0.532
    CREATIVE = {photographer, journalist, painter, model, poet, filmmaker,
                composer, comedian, dj, rapper, interior_designer}:
                n=10,835 (21.7%), fem_frac=0.431
    Together 50.1% of the subset; remaining professions (professor, attorney,
    teacher, accountant, pastor, ...) are axis-neutral, like
    sentiment-neutral stories on TinyStories.
  Criterion deviation, recorded before training: the originally drafted
  criteria were (a) coverage >= 15% per group, (b) opposite gender skew,
  (c) no gender-marked titles. The intended technical pole covers only 5.7%
  of the natural subset and fails (a); rather than distort the natural
  profession distribution by stratified sampling, we keep natural sampling
  and take the creative/media pole. (b) is therefore weakened to "measurably
  different skew": 0.532 vs 0.431, point-biserial r(axis, gender)=0.100
  within the covered half. The rho-null on TinyStories (correlation forced
  across [-0.3,+0.3] changes nothing) says the axis-gender correlation
  magnitude should not matter for leakage; r=0.10 sits inside that tested
  regime.
- Off-target readout: protagonist gender via held-out linear classifier
  (difference-of-means in the frozen embedding space, trained on real bios),
  plus `lexicons_bib.FEM/MAL` fraction. Same two metrics as TinyStories
  (logit-shift primary, fraction range secondary).
- Decontamination: project the gender direction out of the occupation
  steering axis before steering, exactly as on TinyStories.
- Measurement decoupling: natvar readouts and leakage readouts use different
  instruments where possible (lexicon vs classifier family); report both.

## Gate G1 (sanity, 1 run: M1 s1) — before any sweep money is spent

Train one M1 model. Pass requires ALL of:

1. Corpus learned: fixed-code resampling fidelity >= 0.90 and distinct-2
   >= 0.65 (TinyStories references: 0.962 / 0.755); generations are
   recognizable English bios by inspection of 50 samples.
2. Readout works: the frozen-space gender classifier reaches AUC >= 0.90 on
   held-out REAL bios; lexicon gender readout agrees with the structured
   field on >= 90% of non-tie real bios.
3. On-target control: steering the decontaminated occupation axis moves the
   occupation-domain fraction monotonically across alpha with range >= 0.6.

If G1 fails on (1): the corpus is too hard at this scale; stop, report as a
scale boundary, consider WikiBio-lite or simplified bios. If it fails on
(2)/(3) only: fix instruments (lexicons/axis), not the model, and re-gate.

### G1 OUTCOME (2026-07-22 ~02:30, M1 s1 checkpoint_18750 = epoch 30): PASS

1. Corpus learned: fidelity 0.916 (>=0.90), within-sim 0.967, across-sim
   0.917, distinct-2 0.895 (>=0.65); generations are recognizable bios
   (inspected; creative pole -> exhibitions/festivals, health pole ->
   medical school/residency). Near-identical to the TinyStories M1
   reference row at half training.
2. Instruments: gender lexicon vs field 1.000 (coverage 0.988); occ-domain
   lexicon vs profession group 0.976 (in-group coverage 0.753); gender
   classifier AUC 0.999 under the amendment below.
3. On-target control: orthogonalized occupation axis sweeps health_frac
   0.000 -> 1.000, monotonic; range 1.0 (>=0.6).

INSTRUMENT AMENDMENT (recorded 2026-07-22, BEFORE any leakage eval on any
BiB model): the gender readout family is L2-regularized logistic regression
in the frozen pooled-embedding space (held-out, still linear). Plain
difference-of-means is underpowered on this corpus: AUC 0.886 at n=4000 and
0.886 at n=20000 (signal present but misaligned with the raw mean-difference
direction under anisotropic covariance), vs 0.999 for logistic regression at
n=20000 (val n=2000). The leakage logit-shift metric on BiB uses the
logistic-regression decision function; the bootstrap over classifier
training in the 5-eval-seed protocol resamples the logreg training set.

EARLY OBSERVATION (explicitly NOT a primary-endpoint readout): in the G1
steering sweep on half-trained M1 s1, female_frac moved 0.25 -> 1.00 across
the orthogonalized occupation axis (fraction-range 0.75, 24 samples/alpha,
1 eval seed, saturating metric). Direction is corpus-plausible (health pole
skews female). No interpretation until the preregistered logit-shift eval on
final checkpoints; recorded here only so it cannot be mistaken later for a
post-hoc discovery.

## Control-parity precondition (confound guard)

Leakage comparisons between k and M1 are only valid at matched on-target
strength. Before reading any leakage number, verify each model's on-target
fraction range is within +-20% of M1's; otherwise report leakage normalized
per unit on-target logit-shift and flag the deviation. (On TinyStories all
models saturate 0 -> 1 and this is moot; do not assume it transfers.)

## Batches and primary/secondary endpoints

- Batch 1 (4 runs): M1 x {s1, s2} + k16 x {s1, s2}. (M1 s1 = the G1 model.)
- Batch 2 (4 runs, only if primary endpoint direction confirmed): k64 x
  {s1, s2} + k256 x {s1, s2}.

AMENDMENT (2026-07-21, before training started): idle GPUs across three
hosts allow launching all 8 runs in parallel (owner directive), so the
batch-1/batch-2 COMPUTE gating is dropped. The G1 gate becomes: evaluate the
earliest usable M1 s1 checkpoint; if G1 fails, kill the whole fleet. All
ANALYSIS/interpretation rules below are unchanged, in particular: leakage on
k64/k256 models is only interpreted if the primary endpoint direction holds,
and a failed primary endpoint is reported as a boundary condition regardless
of what the k64/k256 runs show. Effective batch is 80 everywhere; hosts with
fewer free GPUs use grad accumulation (global_batch_size x grad_accum_steps
= 80, warmup_steps scaled by grad_accum_steps so optimizer-step warmup stays
1000, lr identical by the blr scaling rule in train.py).

PRIMARY (the replication claim lives or dies here):
  Endpoint gap: mean gender logit-shift under decontaminated occupation
  steering is larger for k16 than M1, in BOTH training seeds, with
  non-overlapping 5-eval-seed intervals in at least one seed and pooled
  bootstrap p < 0.05. (TinyStories reference: 0.97+-0.15 vs 0.37+-0.06.)

SECONDARY (reported either way, no gating):
  S1. Probed superposition (Manifold Probe, code space) decreases k16 ->
      k64 -> k256 in seed-averaged means. Expected to replicate; treated as
      background capacity check, NOT as the headline (it is close to a
      capacity necessity and its replication alone does not support the
      behavioral claims).
  S2. Leakage trend across k16 > k64 > k256 > = M1 in seed-averaged means;
      adjacent overlaps allowed (trend + endpoint reading, as in the paper).
  S3. Probe-leakage correlation across the 4 models x 2 seeds.

## Fragility-law directional prereg (to be completed BEFORE leakage evals)

Procedure locked now; numbers to be filled in a committed update of this
file after natvar measurement and before any leakage measurement:

1. After Batch 1 training, measure natvar (attribute variation at fixed
   code) and single-token movability for every target in
   `lexicons_bib.TARGETS` (+ gender, seniority) on k16 s1, using the
   TinyStories natvar protocol unchanged. Drop targets with prevalence
   < 5% on train_meta.
2. Write the predicted target ranking (natvar x movability, descending)
   into the "Predictions" section below and commit BEFORE running
   eval_leakage_pairs on Bias in Bios.
3. Test: Spearman between predicted ranking and measured per-target leakage,
   plus sign test on the direction (high-natvar targets absorb more). A
   directional hit with rho >= 0.5 counts as replication of the targeting
   law; full cell-level r is reported but not gated (fewer targets than the
   12-attribute TinyStories set).

### Predictions (FILLED 2026-07-22 05:55, BEFORE any BiB leakage eval ran)

- natvar measured on k16 s1 checkpoint_37500 (5 codes x 24 resamples,
  logreg readout): gender .5699, teaching .3104, awards .2944, family
  .2822, experience_yrs .2643, tech .2545, academia .1885, seniority .1704
  (paper/bib_natvar_k16s1.json).
- movability (ins_swing, encoder-only on real bios, 5 bootstrap seeds):
  awards .4091, teaching .3540, tech .3426, family .3259, gender .2856,
  experience_yrs .2442, academia .1833, seniority .0462
  (paper/bib_fragility_scores.json). Caveat recorded: the seniority
  ins_swing rests on n=13 correctly-classified bios (weak instrument).
- PREDICTED RANKING (natvar x ins_swing, descending):
    gender .1628 > awards .1204 > teaching .1099 > family .0920 >
    tech .0872 > experience_yrs .0645 > academia .0346 > seniority .0079
- Test as preregistered: Spearman rho between this ranking and measured
  per-target leakage (source = occupation axis, decontaminated), plus sign
  test on direction; rho >= 0.5 counts as replication of the targeting law.
- committed at: (git hash pending; file timestamped 2026-07-22 05:55, and
  the leakage evals below launched only after this section was written)

## Seed-1 results (recorded 2026-07-22 ~06:00, after the Predictions above)

PRIMARY ENDPOINT, seed-1 half (continuous logit-shift, 5 eval seeds, logreg
instrument): M1 s1 = 1.073 +- 0.271 at ctrl 1.000; k16 s1 = 1.165 +- 0.200
at ctrl 0.692 +- 0.207. Raw gap +0.09 (overlapping), but the control-parity
guard triggered (k16 on-target 31% below M1, outside +-20%), so the
preregistered normalization applies: per unit on-target shift k16 = 1.68 vs
M1 = 1.07 (+57%). Await seed 2 for the both-seeds criterion. Corpus-2
observations: the M1 floor is ~3x TinyStories (1.07 vs 0.37); k16 loses
on-target steerability on this corpus (all models saturated on TinyStories).

DIRECTED PAIRS (occupation source, 8 prereg targets + length extra, 5 eval
seeds): every target leaks more under k16 than M1 (uniform capacity effect,
8/8 targets). Preregistered model-matched targeting test (k16 natvar x
ins_swing vs k16 leakage): Spearman rho = 0.310 (p=.456) -- BELOW the
preregistered 0.5 bar; the targeting law does NOT replicate at the
registered threshold on seed 1. Recorded, not claimed: the same prediction
against M1's leakage gives rho = 0.690 (p=.058); pooled 0.571. Candidate
explanation for the miss, to test rather than assert: k16's top absorbers
(teaching 1.042, tech 0.911, academia 0.781) are occupation-adjacent
attributes, i.e. source-target semantic dependence that TinyStories'
source/target sets did not have; the BiB target set may violate the law's
implicit independence assumption rather than refute its mechanism.

## Seed-2 results and primary-endpoint verdict (recorded 2026-07-22 ~12:05)

Seed-2 continuous (same protocol): M1 s2 = 1.150 +- 0.144 at ctrl 1.000;
k16 s2 = 0.962 +- 0.218 at ctrl 0.350 +- 0.107.

VERDICT, stated precisely:
- The control-parity precondition FAILED in both seeds (k16 ctrl 0.692 s1,
  0.350 s2; both far outside +-20% of M1's 1.000). Raw logit-shift
  comparisons between k16 and M1 are therefore confounded exactly as the
  precondition anticipated; for the record the raw gap is +0.09 (s1) and
  -0.19 (s2), i.e. the ORIGINAL raw-metric endpoint does NOT hold in both
  seeds.
- Under the preregistered fallback (leakage per unit on-target shift), k16
  exceeds M1 in BOTH seeds: 1.68 vs 1.07 (s1, +57%) and 2.75 vs 1.15 (s2,
  +139%). Flag: the s2 ratio divides by a small, noisy ctrl (0.350+-0.107),
  inflating both the estimate and its uncertainty; treat magnitudes with
  caution, the sign consistency is the claim.
- UNANTICIPATED second harm, consistent across seeds and absent on
  TinyStories: the k16 bottleneck itself degrades on-target steerability
  (0.69/0.35 vs 1.00), echoing the dec-lambda steerability-collapse failure
  mode but arising here with NO regularizer. On this corpus the low-rank
  code is worse on both sides of the ledger: less control, more leakage per
  unit control. Large seed spread in k16 ctrl (0.69 vs 0.35) noted; two
  seeds cannot pin its distribution.
- Paper framing to use: on Bias in Bios the compact-bottleneck harm
  replicates and compounds; the clean TinyStories-style raw endpoint gap is
  not observable because control parity itself breaks, and this breakage is
  part of the finding, not a nuisance.

## Secondary endpoints S1-S3 (recorded 2026-07-22 ~12:25; k64 s2 pending)

Probe (mean_off interference, 1500 bios, 5 splits): M1 .0130 (both seeds
identical as expected -- the M1 "code" is the frozen encoder mean-pool, so
the probe reads encoder geometry, not the trained model), k256 .0647/.0696,
k64 s1 .1768, k16 s1 .2318, k16 s2 .0561.

S1 (monotonicity): holds strictly in seed 1 (.232 > .177 > .065 > .013).
Seed 2 is broken by k16 s2, whose probe reads LOW -- but that is the same
model whose on-target control collapsed to 0.35: its code is partially
unused, so code-space interference is not the quantity the probe was
validated to measure there. Claim scoped accordingly: probe superposition
rises monotonically as k shrinks among models whose code is functional.

S2 (leakage trend): normalized seed-mean ordering k16 (2.22) > k64 (1.20,
s1 only) > k256 (0.96) ~ M1 (1.11). Trend holds at the small-k end; the
k256/M1 tail is flat rather than separated -- on this corpus the full-rank
floor is already ~3x the TinyStories floor and capacity pressure only bites
at small k. Control degrades monotonically with shrinking k (1.00 > .97 >
.86 > .69/.35), an independent second trend.

S3 (probe tracks leakage across models): among functional-code models
(n=6) Pearson r = .76 / Spearman .52 vs normalized leakage -- direction
replicates but weaker than TinyStories' r = .90. Including the degenerate
k16 s2, r collapses to .13: that model pairs the FLEET-WORST leakage with
near-clean code geometry. This dissociation is itself evidence for the
paper's localization claim (the code is a pointer; leakage is manufactured
in the denoiser): a model can largely abandon its code, keep leaking, and
the code-space probe cannot see it.

## FINAL consolidated results, all 8 models (recorded 2026-07-22 ~15:15)

| model   | logit_shift | ctrl  | norm (shift/ctrl) | probe mean_off |
|---------|------------|-------|-------------------|----------------|
| M1 s1   | 1.073      | 1.000 | 1.07              | .0130          |
| M1 s2   | 1.150      | 1.000 | 1.15              | .0130          |
| k256 s1 | 0.860      | 1.000 | 0.86              | .0647          |
| k256 s2 | 1.029      | 0.967 | 1.06              | .0696          |
| k64 s1  | 1.027      | 0.858 | 1.20              | .1768          |
| k64 s2  | 1.172      | 0.683 | 1.72              | .1389          |
| k16 s1  | 1.165      | 0.692 | 1.68              | .2318          |
| k16 s2  | 0.962      | 0.350 | 2.75              | .0561          |

Seed means: normalized leakage k16 2.22 > k64 1.46 > k256 0.96 ~ M1 1.11;
on-target control M1 1.00 > k256 0.98 > k64 0.77 > k16 0.52. Both trends
monotone in shrinking k; control degradation has large seed spread at
k<=64 (k64: .86/.68; k16: .69/.35).

S3 final: probe vs normalized leakage r = .70 / Spearman .60 over the 7
models with functional codes, r = .17 over all 8. The functionality
criterion (on-target ctrl >= 0.5) is post-hoc, adopted to exclude k16 s2
whose code the model demonstrably under-uses; stated as such.

## ROUND 2: source-independent targeting-law test (opened 2026-07-22 ~17:30)

Round 1's targeting miss was attributed to source-target semantic dependence.
Round 2 tests that reading with a target set screened for independence,
frozen BEFORE any round-2 natvar or leakage measurement:

- Selection rule (bib_demo/screen_targets_round2.py, frozen before running):
  prevalence >= 0.05 on 20k train bios AND |log odds ratio| of presence
  between HEALTH-profession and CREATIVE-profession bios <= 0.50, plus
  word-disjointness from all frozen lexicons.
- Rule validation: the round-1 confounded targets fail it exactly as the
  round-1 postmortem predicted (academia +0.84, awards -0.99, tech -0.62,
  experience_yrs +0.66, family +0.75 log-odds), while intuitively "neutral"
  candidates also fail empirically (geography -0.61, temporal -0.98,
  featured -2.26): the screen is not vacuous in either direction.
- KEPT round-2 set (7): collaboration (.066), hobbies (.051), quantities
  (.174), focus (.078) from two candidate waves under the fixed rule;
  teaching (.134, logOR -0.13) and seniority (.133, -0.05) carrying over
  from round 1; gender (logOR +0.41) as the primary off-target.
- Test: cell-level, TinyStories-form. Per-model natvar (all 8 models) x
  global ins_swing -> predicted per-(model, target) leakage ordering;
  Spearman/Pearson against measured round-2 directed leakage. Same
  rho >= 0.5 bar on the pooled ordering.
- Infrastructure note: round-2 evals run on hogfather (H200, GPU 3) with a
  rebuilt pinned env (jax 0.4.25 cuda12); instruments unchanged (logreg
  amendment applies).

### Round-2 predictions (FILLED 2026-07-29 ~23:20, BEFORE any round-2 leakage)

- Measured: ins_swing on real bios (teaching .354 > quantities .341 >
  hobbies .314 > gender .285 > collaboration .275 > focus .222 >
  seniority .046) and per-model natvar on all 8 checkpoints (hogfather GPU 3,
  round-2 registry, logreg instrument).
- Predictor: per-model natvar x global ins_swing, all 56 (model, target)
  cells, saved to paper/bib_r2_predictions.json
  (sha256 4e9add7f9fea1dbb...). Top cells: k16_s1:gender .162,
  k256_s2:teaching .129, k16_s1:quantities .113. NOTE recorded before
  measurement: every k16_s2 cell is predicted near-zero because its natvar
  at fixed code reads near-zero; if its measured leakage is nonetheless
  high (as round 1 found on the primary pair), those cells will fail loudly
  rather than average out -- the degenerate-code model is a stress test of
  the law's model-side premise, and we keep it in the pooled test as
  registered.
- Test: pooled Spearman over all 56 cells and cell-level Pearson, plus the
  same statistics excluding the k16_s2 row (both reported; the pooled one
  is the registered rho >= 0.5 criterion).

### Round-2 results (recorded 2026-07-29 ~23:15, after the predictions above)

- REGISTERED POOLED TEST: FAILED. Spearman rho = 0.074 (p = .59), Pearson
  r = -0.451 over all 56 cells.
- PREREGISTERED COMPANION (excluding the k16_s2 row, written into the test
  spec above before measurement): PASSED. rho = 0.599 (p < 1e-4),
  r = 0.560 over 49 cells, clearing the 0.5 bar.
- The failure is exactly the pre-flagged stress test, at maximal amplitude:
  every k16_s2 cell was predicted near-zero (natvar at fixed code ~0) and
  every one measured FLEET-MAXIMUM leakage (2.27-3.94, vs 0.3-1.2 typical
  elsewhere). One degenerate-code model contributes 7 cells that invert the
  pooled Pearson single-handedly.
- Reading, stated carefully: on source-independent targets the targeting law
  REPLICATES among functional-code models (0.31 in round 1 with confounded
  targets -> 0.60 in round 2; the round-1 postmortem is confirmed), and the
  law acquires a precise boundary condition: its premise (steering biases
  variation the denoiser already produces) presupposes a functional
  code-to-denoiser channel. When that channel is degenerate (k16_s2: control
  0.35, natvar ~0, probe ~clean), steering acts as an out-of-distribution
  input and produces large indiscriminate movement that natural variation
  cannot predict, because nothing about the model's natural behavior is
  exercised by an input it has learned to ignore. Leakage without natural
  variation is possible exactly where the law's premise fails, and the
  degenerate model supplies leakage larger than anything the functional
  models produce.

## Depth-window localization + k16_s2 mechanism resolution (2026-07-30 ~00:40)

New instrument (model unchanged for training; modules/model.py gains
inference-only phi_alt/phi_layer_select; src/eval_leakage_depth.py): the
steered code is made readable only to a chosen LAYER window, base code
elsewhere, all conditions through one graph with paired noise (shape-floor
noise cancels; logic exactness proven at 4e-6 by bib_demo/test_depth_equiv.py
under float32 matmuls; TF32 must be off, it injects ~1e-3).

RESULT, 14 models, both corpora: on-target control AND leakage both enter
almost entirely through the FIRST QUARTILE of layers (L0-L2 of 12; L0 alone
carries ~100% in TinyStories k64). No depth separation between control and
leakage anywhere. Combined with the time-window analysis: the code is read
once, early in depth; the capacity pressure plays out along TRAJECTORY TIME
(terminal snap; early-time migration under pressure), not along network
depth. BiB models show superadditive single-layer leakage (L0-only steering
leaks ~1.9x the full-depth steer at k64 s1): later layers partially
compensate when they can also see the steer.

K16_S2 RESOLVED (diagnostics bib_demo/diag_k16s2_depth.py,
diag_k16s2_pairspath.py): its denoiser output is BITWISE invariant to phi
(max|d out| = 0.0 even for +10-sigma phi perturbations, on the production
generation path), while its DECODER HEAD (the t=1 CE branch, also
phi-conditioned) still reads phi. Its generations are degenerate repetitive
text whose content swings wholesale with alpha. Therefore: (a) its fleet-max
round-2 leakage is real measured behavior but flows ENTIRELY through the
decoder-conditioning channel, not the denoiser; (b) its natvar ~ 0 and probe
~ clean readings are consistent (encoder-side code intact, denoiser read
severed); (c) the "partly abandoned code" of earlier sections is now exact:
fully severed in the denoiser, retained in the decoder; (d) this model is
also QUALITY-COLLAPSED (repetition), which the seed-1-only G1 gate never
tested, so quality gating per training seed is a protocol lesson for any
future fleet. The law's boundary condition sharpens: natvar predicts leakage
carried by the denoiser's conditional response; leakage carried by a
different channel (decoder conditioning) is invisible to it by construction.

## Factorial carrier decomposition (2026-07-30 ~02:30, all 14 models)

New instrument (src/eval_channels.py): latents generated with phi_gen in
{base, steer}, decoded with phi_dec in {base, steer}; text-level readout
(same instrument as the leakage protocol). Shares of the both-steered
effect:

- TinyStories x6: denoiser .95-1.01 ctrl / .96-1.01 leak; decoder <= .06 /
  <= .10.
- BiB functional x7: denoiser .74-.97 / .66-1.06; decoder .07-.28 ctrl,
  <= .09 leak (M1 s1's decoder leak share .56 of a near-zero total 0.055,
  share-of-nothing noise).
- BiB k16 s2: denoiser 0.00/0.00, decoder 1.00/1.00 -- the exact
  complement; complete carrier inversion measured factorially.

"Manufactured in the denoiser" is now a measured allocation across the
fleet, and the severed model conserves the accounting: share moves to
whichever component still reads the code. Paper: tab:channels in sec:depth.

## Dose-response extension (2026-07-30 ~02:35; NOT preregistered, post-hoc)

alpha in {-4..4}, 9 doses, 24 gens/dose, seed-1 fleet, occupation axis
orthogonalized against gender, lexicon readout (eval_steering_bib.py; the
run's stdout table was lost to a grep filter, curves recomputed exactly
from the saved generations, bib_demo/recompute_dose.py ->
paper/bib_dose_response.json).

- on-target endpoint delta / monotonic: M1 +1.00 yes; k256 +1.00 yes;
  k64 +0.88 no; k16 +0.62 no (peaks 1.00 at alpha=1, falls to 0.62 by
  alpha=4: high-dose control failure, dose-domain image of tab:bib row 1).
- off-target gender range (per unit delta): M1 0.21 (0.21); k256 1.00
  (1.00); k64 0.92 (1.05); k16 0.75 (1.21). Per-unit ordering matches
  normalized leakage. k16 traces a U (0.71 -> 0.04 -> 0.75), both
  directions under one axis.
- k256 swing is dose-locked: gender 0.00->0.83 in the same unit interval
  as target 0.29->1.00.

Paper: tab:dose + paragraph in sec:bib, labeled post-hoc.

SEED-2 EXTENSION (2026-07-30 ~13:50, binky GPUs 1-3 + death GPU 3, ~2 min
per model; full stdout kept this time in dose_*_s2.full.log):

- m1_s2: delta +1.000 monotonic, fem range 0.125 (cleaner than s1's .21).
- k16_s2: +0.250 non-mono, fem range .542 (severed-code model).
- k64_s2: +0.292 non-mono, fem range .833.
- k256_s2: +0.042 non-mono, fem range .958 -- BUT the curve steers 0->1
  cleanly through alpha=+3 and then COLLAPSES at alpha=+4 (mean length 5.1
  tokens): endpoint delta reads through generative breakdown, not
  unresponsiveness. Gender still dose-locked 0->1 like s1.

Seed-2 verdict: full rank monotonic both seeds; NO bottleneck seed-2 model
keeps monotonic dose control (delta <= .29) while gender ranges stay
.54-.96. tab:dose updated to s1/s2 pairs; per-unit row seed-1 only
(undefined when on-target control fails). paper/bib_dose_response.json now
holds all 8 curves.

## Per-seed quality gate (2026-07-30 ~02:05, binky; planned instrument)

eval_semantic (gate: within - across > .02 AND distinct-2 > .3), 8 BiB
models + TS k16 s2/s3 + TS k64 s2. First pass had a --config_override crash
for 9 models (eval_semantic.py lacked the flag; added, rerun in
quality2.log).

- 7/8 BiB pass. bib_k256_s2 FAILS with margin -.015 (within .871 < across
  .886; weak per-phi fidelity despite 0.97 on-target control). Flagged in
  paper; no sec:bib conclusion changes if dropped.
- bib_k16_s2 (severed denoiser) PASSES, margin +.058: the gate is carried
  by the decoder channel, consistent with the factorial decomposition.
  Corrects an earlier informal note calling this model "quality-collapsed";
  the measured gate says otherwise.
- TS k16 s2/s3 marginal FAIL (+.008/+.009): k=16 is where code reading
  itself gets fragile.

Paper: tab:quality + paragraph in sec:bib.

## Training-time phi-routing ablation (2026-07-30, death; TS k=16, seed 42)

Causal complement to the inference-time factorial decomposition: train with
phi visible to only ONE component (config phi_route in {denoiser, decoder},
5 denoiser call sites vs decoder branch in train_step.py). Matched budget
(eff. batch 80 via batch 40 x accum 2, warmup 2000 micro, 37500 opt steps).
Evals condition each component exactly as trained (eval_leakage_pairs /
eval_steering now read cfg.phi_route; default "both" backward compatible).

- DENOISER-ONLY arm (done 11:31): pairs12 mean logit-shift 0.729 (44
  pairs), within the normal-k16 seed family range (s1 .82, s3 .67, s2
  .47) and near s1; per-source ctrl .42-1.06, inside seed spread. Leakage
  needs NO decoder conditioning: phi through the denoiser alone
  manufactures it at full strength.
- DENOISER-ONLY arm, full triple (13:08): natvar mean logit_std 0.217,
  inside the normal-k16 band (.205-.218; M1 .164) -- the natvar
  instrument sees denoiser-carried leakage, as the law requires. Factorial
  carrier decomposition: denoiser share .93 ctrl / .92 leak, decoder
  share .02 / .11 -- the trained-in routing is recovered by the
  inference-time instrument.
- Quality gate (15:10): margin +.016, marginal fail exactly like the
  other TS k16 seeds (s2 +.008, s3 +.009); distinct-2 .79 healthy.
  Instrument note: eval_semantic needed the same phi_route patch as the
  other evals (feeding phi to route_den's never-conditioned decoder is an
  OOD prefix and read margin +.005 with fidelity .849; routed correctly
  it reads +.016 with fidelity .943). Patch is identity for phi_route=
  "both" models, so no earlier quality number changes.
- DECODER-ONLY arm: training ~ep34/60 at time of writing (decode-side phi
  is ~2.7x slower per micro-step); pairs12 auto-queued.

Prediction (recorded before decoder-arm eval): decoder-only arm should
look like bib_k16_s2's carrier profile (leakage via decode head; natvar
instrument blind to it by construction).

- DECODER-ONLY arm, full quadruple (21:50): PREDICTION MISSED, in an
  informative direction. The arm does not look like bib_k16_s2; it fails
  to learn usable conditioning AT ALL. pairs12 ctrl ~0 on every source
  (-.05/-.03/+.02/-.02): steering the code does nothing through the
  trained pathway (mean raw drift 0.422 with zero control is not
  leakage-per-unit-anything). Quality margin -0.10 (within .895 < across
  .992): phi does not condition content. natvar logit_std .59-1.04
  (~3-5x the k16 band): unconditional-denoiser variation, not
  code-conditioned variation. So decode-pass-only training cannot even
  establish control; bib_k16_s2's decoder carriage must be a product of
  joint training followed by denoiser abandonment, not something the
  decoder pathway can learn alone. Strengthens the main claim: there is
  no decoder-only escape route from denoiser leakage, because that route
  carries no control to begin with.
- INSTRUMENT CAVEAT: eval_channels on route_dec reads denoiser share
  +1.07 / decoder +0.04 -- an OOD artifact, not a trained carrier. The
  channels instrument injects phi tokens into both sites regardless of
  phi_route; the phi projection is parameter-shared with the (trained)
  decode pass, so injecting steered phi into the never-conditioned
  denoiser perturbs 60 recursive sampling steps and dominates the
  attribution. Valid on route_den (phi-in-denoiser is its trained
  pathway; its decoder cells correctly read ~0 because a single OOD
  decode pass is inert); NOT valid on route_dec. channels_route_dec.json
  kept as a record of the artifact, excluded from any table.

## ELF-M scale anchors (24-layer; 2026-07-30, hogfather/death/binky)

Matched optimizer budget (37500 steps, eff. batch 80 via accum). Sacrebleu
crash killed the first launch; relaunched 02:30. M1 s1 trained on hogfather
(8.5h, checkpoint_75000 = micro-step numbering), evals on binky (local
config + model=ELF-M override; checkpoint config.yml carries
hogfather-absolute paths and cannot be used off-machine).

- M1 s1 (12:43): pairs12 mean logit-shift 0.450 (44 pairs) vs ELF-B M1
  0.430; ctrl saturated 1.00 on all four sources both depths. FULL-RANK
  FLOOR IS DEPTH-STABLE. Quality gate PASS (fidelity .972, margin +.052,
  distinct-2 .80). C2 steering PASS monotonic (+1.000 delta).
- k16 s1: training on hogfather (eta ~19:30); M1 s2 on death 0-3;
  k16 s2 queued on hogfather (chain watcher skips its M1 s2 leg).

Prediction (recorded before k16 s1 eval): if capacity pressure is the
driver, the k16 excess over M1 (ELF-B: 0.821 vs 0.430 pairs12) persists
or grows at 24 layers; depth alone should not relieve it.

- M1 s1 natvar (14:49): mean logit_std .14-.16 (sent .155 gend .138 anim
  .139) vs ELF-B M1 .164 -- full-rank natvar floor also depth-stable.
  Enables the law check at 24 layers once k16 s1 natvar lands (ins_swing
  is encoder-space, model-independent, reusable as-is).
- k16 s1 (trained 19:39, evals 21:48): PREDICTION CONFIRMED AND
  SHARPENED. pairs12 mean shift .663 vs M1 .450; but per-source ctrl
  collapses to .20/.50/.28 (sent/gend/anim; length 1.04) vs ELF-B k16's
  .82/.57/.62. Normalized (shift over mean non-length ctrl): ELF-M k16
  ~2.0 vs ELF-B k16 ~1.2, with the M1 floor flat (.43-.45). The
  bottleneck harm GROWS with depth, and the on-target control
  degradation that BiB exhibited at 12 layers appears on TinyStories at
  24: the "harder corpus" compounding is a general capacity-pressure
  phenomenon, not a corpus quirk. Quality gate PASS at the wire (margin
  +.020 vs .02 bar); steering VERDICT PASS monotonic. s2 anchors
  training on hogfather.
- k16 s1 natvar (22:20): mean_all .212 vs ELF-B k16 .210 -- the
  bottleneck natvar excess is depth-stable alongside the M1 floor
  (.164->.158). Same overlap, higher price: normalized leakage 1.2->2.0
  entirely through control collapse. Paper: tab:elfm + tab:route +
  paragraphs in sec:depth (16 tables total now).
- ts_k16 s1 quality margin measured for tab:route: +.018 (all TS k16
  seeds sit at +.008..+.018, route_den's +.016 is family-typical).
- SEED-2 ANCHORS (trained 7/31 05:15 + 14:22 hogfather; evals 8/1
  22:30-23:00, pairs12 on hogfather H200, quality/natvar on binky):
  m1_s2 pairs12 .473 / ctrl 1.00 / norm .47; natvar .154; quality
  +.054 PASS. k16_s2 pairs12 1.470 / ctrl .33 / norm 4.41; natvar
  .219; quality +.017 marginal fail (k16-family signature; s1 +.020
  just passes). EVERY depth-anchor regularity replicates at seed 2:
  floor .45/.47, control collapse .33/.33 exact, natvar excess stable,
  normalized harm 2.0/4.4 vs ELF-B 1.2. tab:elfm final (s1/s2 pairs);
  "second seeds training" caveat removed from sec:depth.

## TS dose-response sweep (2026-07-30 ~14:50; exploratory, NOT for paper)

Same protocol as BiB tab:dose (alpha -4..4, orthogonalized, 24/dose) on
the 6 TinyStories primaries. m1 +1.00 mono (fem range .58, animal .54);
k8 +0.58 non-mono (animal .92); k16 +1.00 mono (fem .83); k64 +1.00 mono
(fem .21); k256 +1.00 mono (fem .71, animal .71); k512 +1.00 mono (fem
.38, animal .54) (both rerun from checkpoints_from_death after an empty
ckpt path on first attempt). DECISION: single-seed 24-sample lexicon fractions are
too noisy to separate M1 from bottlenecks on TS (M1's own ranges .5-.6);
keep as exploratory record only, do NOT add a TS dose table to the paper.
The BiB dose table stands on its cleaner normalized contrast.

## BiB k16 third seed (seed 44; trained death GPUs 1,2; evals 00:48-00:52 7/31)

Purpose (recorded at launch): arbitrate whether s2's severed code is
modal, and add a third point to the k16 primary-endpoint row. Not part of
the preregistered 8-run fleet; robustness extension.

- Continuous endpoint: gender logit_shift 1.828+-0.472, ctrl
  0.267+-0.057 -> normalized 6.85, fleet-max (s1 1.68, s2 2.75).
  Control degradation deepest of the three seeds.
- NOT severed: natvar healthy (gender .333, others .13-.32; s2 read ~0),
  probe mean interference .227 (s1 .232, s2 .056), quality gate PASS
  (margin +.062, distinct-2 .84). Severed-code = 1/3 seeds, an outcome
  mode rather than the mode.
- Paper: prose addition to sec:bib after the tab:bib discussion.

## Pre-decided interpretations

- G1 pass + primary endpoint replicates: report as cross-corpus replication;
  add corpus-2 subsection and drop "TinyStories-only" from limitations.
- G1 pass + primary endpoint FAILS (k16 <= M1 in either seed): this is a
  publishable boundary condition (the capacity-driven leakage depends on
  corpus attribute structure), NOT a file-drawer result. Report it, keep the
  TinyStories-only scoping honest, and do not run Batch 2 chasing the trend.
- G1 fail on corpus learnability: scale boundary; report one paragraph in
  limitations, do not interpret any leakage number from a model that failed
  G1.

## Explicitly out of scope for corpus 2

Full 12-attribute within-model analysis, decorrelation lambda-sweep, rho
data-intervention, inference-time projection null across 7 models, CF
mitigation. These are mechanism experiments already localized on
TinyStories; corpus 2 tests transfer of the phenomenon, not the mechanism.
