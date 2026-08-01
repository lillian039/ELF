#!/usr/bin/env bash
# Time-resolved leakage trajectory eval, binky env.
# Usage: bash tinystories_demo/traj_eval_binky.sh <gpu> <tag:dir> [...]
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
  echo "=== traj ${tag}: ${ckpt} ==="
  python3 src/eval_leakage_trajectory.py \
    --config "${dir}/config.yml" --checkpoint_path "${ckpt}" \
    --out "paper/traj_${tag}.json" ${TRAJ_ARGS:-} \
    2>&1 | tee "tinystories_demo/logs/traj_${tag}.log" | grep -aE "TRAJ_SUMMARY|WINDOW_SUMMARY|===" || true
done
echo "TRAJ CHAIN binky (gpu ${GPU}) DONE"
