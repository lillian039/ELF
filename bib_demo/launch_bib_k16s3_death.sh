#!/usr/bin/env bash
# BiB k=16 third seed (44) on death GPUs 1,2: robustness seed for the
# primary-endpoint row (s1 functional / s2 severed-code; s3 arbitrates).
# Effective batch 80 via global 20 x accum 4 (per-dev micro 10, fits 3090).
# Usage: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_bib_k16s3_death.sh'
set -euo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

: > bib_demo/logs/train_bib_k16_s3.log
CUDA_VISIBLE_DEVICES=1,2 setsid nohup bash "${REPO}/bib_demo/run_train_bib.sh" M2 \
  --config_override "manifold_dim=16" \
  --config_override "seed=44" \
  --config_override "save_freq=10" \
  --config_override "global_batch_size=20" \
  --config_override "grad_accum_steps=4" \
  --config_override "warmup_steps=4000" \
  --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k16_s3" \
  > bib_demo/logs/train_bib_k16_s3.log 2>&1 < /dev/null &
disown
echo "launched bib_k16_s3 on GPUs 1,2, pid $!"
