#!/usr/bin/env bash
# Run the full directed-pair leakage eval (eval_leakage_pairs.py) over a list of
# checkpoints, sequentially on one GPU. Results -> paper/pairs_<tag>.json.
#
# Usage: bash tinystories_demo/chain_pairs_eval.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

mkdir -p tinystories_demo/logs

run_one () {  # tag ckpt_dir
  local tag="$1" dir="$2"
  local ckpt
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  echo "=== pairs eval ${tag}: ${ckpt} ==="
  python3 src/eval_leakage_pairs.py \
    --config "${dir}/config.yml" \
    --checkpoint_path "${ckpt}" \
    --seeds 5 --samples-per-alpha 24 \
    --out "paper/pairs_${tag}.json" \
    2>&1 | tee "tinystories_demo/logs/pairs_${tag}.log" | grep -E "LEAKPAIR_SUMMARY|===" || true
}

run_one k8  /mnt/faster3/lc2762/elf_tinystories_k8
run_one k16 /mnt/faster3/lc2762/elf_tinystories_k16
run_one m1  /mnt/faster3/lc2762/elf_tinystories_sm_output
echo "ALL PAIRS EVALS DONE"
