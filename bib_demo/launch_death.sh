#!/usr/bin/env bash
# Launch the death-side BiB runs: k16 s1 on GPUs 0-3 now, k256 s2 queued
# behind it. Run via: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_death.sh'
set -euo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

: > bib_demo/logs/train_k16_s1.log
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup bash "${REPO}/bib_demo/run_train_bib.sh" M2 \
  --config_override "manifold_dim=16" \
  --config_override "seed=42" \
  --config_override "save_freq=10" \
  --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k16_s1" \
  > bib_demo/logs/train_k16_s1.log 2>&1 < /dev/null &
disown
echo "launched k16_s1 pid $!"

: > bib_demo/logs/queue_k256_s2.log
setsid nohup bash "${REPO}/bib_demo/queue_bib.sh" M2 0,1,2,3 37500 \
  --config_override "manifold_dim=256" \
  --config_override "seed=43" \
  --config_override "save_freq=10" \
  --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k256_s2" \
  > bib_demo/logs/queue_k256_s2.log 2>&1 < /dev/null &
disown
echo "queued k256_s2 pid $!"
