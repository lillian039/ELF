#!/usr/bin/env bash
# Serial queue for death GPUs 0-3 behind the running k16 s1:
#   k256 s2, then k64 s2 (moved off binky 5,6 -- user: no shared GPUs).
# Replaces the earlier single-run k256 s2 queue (kill it before starting this).
# Run via: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_death_chain.sh'
set -euo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

: > bib_demo/logs/chain_k256s2_k64s2.log
setsid nohup bash -c '
  cd '"${REPO}"'
  bash bib_demo/queue_bib.sh M2 0,1,2,3 37500 \
    --config_override "manifold_dim=256" \
    --config_override "seed=43" \
    --config_override "save_freq=10" \
    --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k256_s2"
  echo "=== chain: k256 s2 done, starting k64 s2 ($(date "+%F %T")) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash bib_demo/run_train_bib.sh M2 \
    --config_override "manifold_dim=64" \
    --config_override "seed=43" \
    --config_override "save_freq=10" \
    --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k64_s2" \
    || echo "!!! k64 s2 FAILED ($(date "+%F %T"))"
  echo "=== DEATH CHAIN DONE ($(date "+%F %T")) ==="
' > bib_demo/logs/chain_k256s2_k64s2.log 2>&1 < /dev/null &
disown
echo "death chain queued (k256 s2 -> k64 s2), pid $!"
