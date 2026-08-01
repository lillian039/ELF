# Model-seed-2 numbers (2026-07-20, for table updates once k64_s2 lands)

k16_seed2 (checkpoint_37500):
- probe: mean_off=0.1348 (s1: 0.122), s-g=0.0460 (s1: 0.083), capacity=0.3875 (s1: 0.388)
- leak s->g: 0.61+-0.22 (s1: 0.97+-0.15); AUC 0.68
- two-seed k16 mean: leak 0.79 (seed half-range 0.18); capacity replicates to 3 digits
- reading: aggregate probe metrics highly seed-stable; pair-level (s-g) noisier,
  consistent with the paper's aggregate-thermometer framing. Endpoint gap vs M1
  (0.37+-0.06) holds in both seeds.

k64_seed2 (checkpoint_37500):
- probe: capacity=0.100 (s1: 0.100, exact), s-g=0.0298 (s1: 0.069), mean_off=0.111 (s1: 0.055)
- leak s->g matched core protocol: 1.04+-0.31 (s1: 0.76+-0.30); ctrl 0.91
- k16_seed2 matched core protocol: leak 1.04+-0.44 (s1: 0.97+-0.15); ctrl 0.47 (!)
  -> capacity-collapse boundary is seed-dependent (partial control loss at k16 s2)
- law pool with both seed2 models: 19/19 positive Spearman, predictor share 61%
- paper updated: sec:leak seed sentence (final numbers), Limitations rewritten
