#!/usr/bin/env bash
# Round-2 directed-pair leakage eval: extended2 targets (12 attributes), same
# 4 core sources. Results -> paper/pairs12_<tag>.json.
# Usage: bash tinystories_demo/pairs12_eval.sh <gpu> <tag:dir> [...]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="$1"; shift

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

mkdir -p tinystories_demo/logs
for spec in "$@"; do
  tag="${spec%%:*}"; dir="${spec#*:}"
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  echo "=== pairs12 eval ${tag}: ${ckpt} ==="
  python3 src/eval_leakage_pairs.py \
    --config "${dir}/config.yml" \
    --checkpoint_path "${ckpt}" \
    --seeds 5 --samples-per-alpha 24 \
    --targets extended2 --decon all \
    --out "paper/pairs12_${tag}.json" \
    2>&1 | tee "tinystories_demo/logs/pairs12_${tag}.log" \
    | grep -E "LEAKPAIR_SUMMARY|label balance|===" || true
done
echo "PAIRS12 CHAIN (gpu ${GPU}) DONE"
