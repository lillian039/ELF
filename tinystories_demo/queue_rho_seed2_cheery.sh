#!/usr/bin/env bash
# Second-seed replicates of the rho-controlled runs (error bars for the
# dose-response curve). Same matched budget (5 GPUs x 16 = global batch 80),
# seed 7 instead of 42, output dirs get an _s2 suffix.
#
# Usage (on cheery): bash tinystories_demo/queue_rho_seed2_cheery.sh p000 p030
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/faster1/lc2762/venv/bin/activate

export CUDA_VISIBLE_DEVICES="${RHO_GPUS:-1,2,3,4,5}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster1/lc2762/hf_cache

for TAG in "$@"; do
  OUT="/mnt/faster1/lc2762/elf_rho_${TAG}_k64_s2"
  echo "=== $(date '+%F %T') rho ${TAG} k=64 seed2 -> ${OUT} ==="
  python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
    --config_override manifold_dim=64 \
    --config_override seed=7 \
    --config_override data_path=tinystories_demo/data_50k_rho_${TAG}/train \
    --config_override eval_data_path=tinystories_demo/data_50k_rho_${TAG}/val \
    --config_override save_freq=10 \
    --config_override output_dir=${OUT} \
    || echo "!!! rho ${TAG} seed2 FAILED ($(date '+%F %T'))"
done
echo "=== SEED2 QUEUE DONE $(date '+%F %T') ==="
