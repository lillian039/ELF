#!/usr/bin/env bash
# Counterfactual-consistency (generation-pathway) mitigation queue, on death.
# k=64, manifold_cf_weight in {1, 10}, sequential 5-GPU matched-budget runs
# (global batch 80 = 16/GPU, identical to the k-sweep).
#
# death env notes (2026-07-20): the old jax.devices() segfault was the TPU
# probe tripping over the libtpu/libtpu_nightly packages in the NFS-shared
# ~/.local; JAX_PLATFORMS=cuda skips it. LD_LIBRARY_PATH must point at the pip
# nvidia libs (system cuBLAS 11.10 is older than the 11.11 jaxlib needs).
#
# Usage (on death): nohup bash tinystories_demo/queue_cf_death.sh > cf_queue_death.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CF_GPUS:-0,1,2,3,4}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

# The CF second forward doubles activation memory: 16/device OOMs a 24GB 3090,
# so run microbatch 8/device with 2-step gradient accumulation (effective batch
# 80, identical lr scaling and optimizer trajectory; 75000 microsteps = 37500
# optimizer steps).
# v3: constant-scale alpha + sg-manifold + 15000-microstep warmup (12 epochs
# of base training before CF pressure); lambda 10 dropped in favor of 0.3.
for TAG in 1 03; do
  case "${TAG}" in 03) LCF="0.3";; *) LCF="${TAG}";; esac
  OUT="/mnt/faster3/lc2762/elf_cf${TAG}_k64"
  if [ -d "${OUT}/checkpoint_75000" ]; then
    echo "=== cf${TAG}: already complete, skipping ==="
    continue
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
    || echo "!!! cf${TAG} FAILED ($(date '+%F %T'))"
  sleep 60
done
echo "=== CF QUEUE (death) DONE $(date '+%F %T') ==="
