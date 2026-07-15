#!/usr/bin/env bash
# Superposition + directed-pair evals for the decorrelation models, on binky
# (death's jax-GPU env is currently broken — segfaults at device init).
# Runs a list of (dir, step, tag) triples sequentially on one GPU.
#
# Usage: bash tinystories_demo/dec_evals_binky.sh <gpu> <triple> [<triple> ...]
#   triple = dir:step:tag, e.g. /mnt/.../elf_tinystories_k64_dec1:70000:dec1
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:?gpu}"; shift

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

mkdir -p tinystories_demo/logs dec_results
for TRIPLE in "$@"; do
  DIR="${TRIPLE%%:*}"; REST="${TRIPLE#*:}"; STEP="${REST%%:*}"; TAG="${REST#*:}"
  CKPT="${DIR}/checkpoint_${STEP}"
  echo "=== $(date '+%F %T') ${TAG} superposition @${STEP} ==="
  python3 src/eval_superposition.py \
    --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
    2>&1 | tee "tinystories_demo/logs/dec_${TAG}_sup.log" | grep -E "SUPERPOSITION_SUMMARY" || true
  echo "=== $(date '+%F %T') ${TAG} directed pairs @${STEP} ==="
  python3 src/eval_leakage_pairs.py \
    --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
    --seeds 5 --samples-per-alpha 24 \
    --out "dec_results/pairs_${TAG}.json" \
    2>&1 | tee "tinystories_demo/logs/dec_${TAG}_pairs.log" | grep -E "LEAKPAIR_SUMMARY" || true
done
echo "=== DEC EVALS DONE $(date '+%F %T') ==="
