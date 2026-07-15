#!/usr/bin/env bash
# Sequential k x rho training queue for the mechanism experiment, on cheery.
# One 5-GPU matched-budget run at a time (global batch 80 = 16/GPU, same as the
# k-sweep), over the rho-controlled datasets from prepare_tinystories_rho.py.
# Order: max-contrast first (+0.30), then the rho=0 pipeline control, then the
# intermediate/negative points.
#
# Usage (on cheery): nohup bash tinystories_demo/queue_rho_cheery.sh > rho_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

# cheery has no python3.10; use the faster1 venv (same jax/flax versions as binky).
source /mnt/faster1/lc2762/venv/bin/activate

export CUDA_VISIBLE_DEVICES="${RHO_GPUS:-0,1,3,6,7}"   # override with RHO_GPUS=...
export HF_HOME=/mnt/faster1/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster1/lc2762/hf_cache
# NOTE: we call train.py directly (not run_train_sm_m2.sh) because that wrapper's
# ptxas discovery is a binky-specific old-driver hack that breaks on cuda12 wheels.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
K="${RHO_K:-64}"

for TAG in ${@:-p030 p000 m030 p015}; do   # tags as args, default all four
  OUT="/mnt/faster1/lc2762/elf_rho_${TAG}_k${K}"
  if [ -f "${OUT}/checkpoint_37500/checkpoint" ] || [ -d "${OUT}/checkpoint_37500" ]; then
    echo "=== rho ${TAG}: already complete, skipping ==="
    continue
  fi
  echo "=== $(date '+%F %T') rho ${TAG} k=${K} -> ${OUT} ==="
  python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
    --config_override "manifold_dim=${K}" \
    --config_override "data_path=tinystories_demo/data_50k_rho_${TAG}/train" \
    --config_override "eval_data_path=tinystories_demo/data_50k_rho_${TAG}/val" \
    --config_override "save_freq=10" \
    --config_override "output_dir=${OUT}" \
    || echo "!!! rho ${TAG} FAILED ($(date '+%F %T'))"
  sleep 60   # let the GPUs drain before the next run (a crashed run holds memory briefly)
done
echo "=== RHO QUEUE DONE $(date '+%F %T') ==="
