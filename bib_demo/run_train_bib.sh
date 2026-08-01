#!/usr/bin/env bash
# Train SM-ELF on Bias in Bios (second-corpus replication, see
# paper/bib_registration.md). Host-aware: angua uses the cuda12 venv on
# /mnt/faster2; binky/death use system python3 + pip nvidia CUDA libs.
# Caller sets CUDA_VISIBLE_DEVICES. Effective batch must stay 80:
#   4 GPUs -> defaults (global_batch_size=80)
#   2 GPUs -> global_batch_size=40 grad_accum_steps=2 warmup_steps=2000
#   1 GPU  -> global_batch_size=20 grad_accum_steps=4 warmup_steps=4000
# (warmup_steps is counted in micro-steps; scale it by grad_accum_steps to
# keep optimizer-step warmup at 1000. lr is invariant: blr * batch*accum/256.)
#
# Usage: run_train_bib.sh <M1|M2> [--config_override ...]...
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:?M1 or M2}"; shift

if [ "$(hostname)" = "angua" ]; then
  source /mnt/faster2/lc2762/venv/bin/activate
  export HF_HOME="${HF_HOME:-/mnt/faster2/lc2762/hf_cache}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster2/lc2762/hf_cache}"
else
  # CUDA 11.8 ptxas for the old driver; pip nvidia libs ahead of system CUDA
  # (death has empty LD_LIBRARY_PATH over non-interactive ssh).
  NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
  export PATH="${NVCC_DIR}/bin:${PATH}"
  NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
  export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
  export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"
fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec python3 src/train.py --config "bib_demo/train_bib_SM-ELF-${VARIANT}.yml" "$@"
