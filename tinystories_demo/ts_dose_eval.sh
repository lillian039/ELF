#!/usr/bin/env bash
# TinyStories alpha dose-response sweep (eval_steering, alphas -4..4,
# gender-orthogonalized sentiment axis), mirroring the BiB tab:dose protocol.
# Usage: bash tinystories_demo/ts_dose_eval.sh <gpu> <tag:dir> [...]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="$1"; shift

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

mkdir -p tinystories_demo/logs
for spec in "$@"; do
  tag="${spec%%:*}"; dir="${spec#*:}"
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  echo "=== ts dose ${tag}: ${ckpt} $(date '+%F %T') ==="
  python3 src/eval_steering.py --config "${dir}/config.yml" \
    --checkpoint_path "${ckpt}" \
    --label-stories 400 --samples-per-alpha 24 \
    --alphas=-4,-3,-2,-1,0,1,2,3,4 --orthogonalize \
    --out "tinystories_demo/logs/tsdose_${tag}.jsonl" \
    > "tinystories_demo/logs/tsdose_${tag}.full.log" 2>&1
  grep -aE "endpoint delta|VERDICT|female_frac range" "tinystories_demo/logs/tsdose_${tag}.full.log" | head -3
done
echo "TS DOSE CHAIN (gpu ${GPU}) DONE"
