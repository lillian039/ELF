#!/usr/bin/env bash
# CF mitigation training on angua (10x A5000, cuda12 venv at /mnt/faster2).
# k=16 variant: does the generation-pathway penalty hold up under higher
# capacity pressure? v3 recipe (constant alpha + sg-manifold + warmup 15k).
# Usage (on angua): nohup bash /mnt/faster2/lc2762/ELF/tinystories_demo/queue_cf_angua.sh > /mnt/faster2/lc2762/cf_queue_angua.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/faster2/lc2762/venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CF_GPUS:-1,6,7,9}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster2/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster2/lc2762/hf_cache

# Pareto point: gentle weight at k=64 (lambda=1 collapsed control; 0.3 runs on
# death; this maps the low end of the leakage-control tradeoff).
LCF="${CF_LAMBDA:-0.1}"
TAG="${CF_TAG:-cf01}"
OUT="/mnt/faster2/lc2762/elf_${TAG}_k64"
if [ -d "${OUT}/checkpoint_75000" ]; then
  echo "=== ${TAG}_k64: already complete ==="
  exit 0
fi
echo "=== $(date '+%F %T') lambda_cf=${LCF} k=64 -> ${OUT} ==="
python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --config_override "manifold_dim=64" \
  --config_override "manifold_cf_weight=${LCF}" \
  --config_override "manifold_cf_warmup_steps=15000" \
  --config_override "global_batch_size=40" \
  --config_override "grad_accum_steps=2" \
  --config_override "save_freq=10" \
  --config_override "output_dir=${OUT}" \
  || echo "!!! ${TAG}_k64 FAILED ($(date '+%F %T'))"
echo "=== CF ANGUA DONE $(date '+%F %T') ==="
