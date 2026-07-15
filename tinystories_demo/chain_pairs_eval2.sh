#!/usr/bin/env bash
# Directed-pair leakage eval for the checkpoints copied from death
# (k64 baseline = sm_m2_output, k256, k512-learned). Companion to
# chain_pairs_eval.sh, meant for a second GPU.
#
# Usage: bash tinystories_demo/chain_pairs_eval2.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-7}"

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

mkdir -p tinystories_demo/logs
FD=/mnt/faster3/lc2762/from_death

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

run_one k64  "${FD}/elf_tinystories_sm_m2_output"
run_one k256 "${FD}/elf_tinystories_k256"
run_one k512 "${FD}/elf_tinystories_k512"
echo "ALL PAIRS EVALS (death models) DONE"
