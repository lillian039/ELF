#!/usr/bin/env bash
# Model-seed-2 runs for the k-sweep (the paper's stated rigor gap: one model
# per k, adjacent-k separation resting on eval-seed error bars only).
# Waits for GPUs 0-3 to free up (pairs8 eval chains), then trains k=16 and
# k=64 with seed=2 sequentially (global batch 80 = 20/GPU on 4 GPUs).
#
# Usage: nohup bash tinystories_demo/queue_kseed2_binky.sh > kseed2_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

echo "=== $(date '+%F %T') waiting for GPUs 0-3 to be free ==="
while true; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$1<=3 && $2>500 {n++} END {print n+0}')
  [ "${busy}" -eq 0 ] && break
  sleep 300
done
echo "=== $(date '+%F %T') GPUs free, starting k-seed2 queue ==="

for K in 16 64; do
  OUT="/mnt/faster3/lc2762/elf_tinystories_k${K}_seed2"
  if [ -d "${OUT}/checkpoint_37500" ]; then
    echo "=== k${K} seed2: already complete, skipping ==="
    continue
  fi
  echo "=== $(date '+%F %T') k=${K} seed=2 -> ${OUT} ==="
  python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
    --config_override "manifold_dim=${K}" \
    --config_override "seed=2" \
    --config_override "save_freq=10" \
    --config_override "output_dir=${OUT}" \
    || echo "!!! k${K} seed2 FAILED ($(date '+%F %T'))"
  sleep 60
done
echo "=== K-SEED2 QUEUE DONE $(date '+%F %T') ==="
