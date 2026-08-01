#!/usr/bin/env bash
# ELF-M k16 seed-2 scale anchor on death GPUs 0-3 (3090 24GB).
# Micro-batch 2/device (global 8 x accum 10 = effective 80): ELF-M statics
# (params+opt+ema ~13GB) leave no room for the micro-5 activations that
# OOM'd the M1 s2 attempt; micro-2 fits.
# Usage: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_elfm_k16s2_death.sh'
set -euo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

: > bib_demo/logs/train_elfm_k16_s2.log
setsid nohup python3 src/train.py \
  --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --config_override "model=ELF-M" \
  --config_override "manifold_dim=16" \
  --config_override "seed=43" \
  --config_override "save_freq=10" \
  --config_override "global_batch_size=8" \
  --config_override "grad_accum_steps=10" \
  --config_override "warmup_steps=10000" \
  --config_override "output_dir=/mnt/faster3/lc2762/elfm_ts_k16_s2" \
  > bib_demo/logs/train_elfm_k16_s2.log 2>&1 < /dev/null &
disown
echo "launched elfm_k16_s2 on GPUs 0-3, pid $!"
