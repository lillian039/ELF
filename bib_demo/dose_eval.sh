#!/usr/bin/env bash
# BiB alpha dose-response sweep (eval_steering_bib, alphas -4..4, gender-
# orthogonalized axis), one model per call. Host-agnostic (binky/death env).
# Usage: bash bib_demo/dose_eval.sh <gpu> <tag:ckpt_dir:k> [...]   (k empty = M1)
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

mkdir -p bib_demo/logs
for spec in "$@"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; dir="${rest%:*}"; k="${rest##*:}"
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  cfg=bib_demo/train_bib_SM-ELF-M1.yml
  [ -n "${k}" ] && cfg=bib_demo/train_bib_SM-ELF-M2.yml
  echo "=== dose ${tag}: ${ckpt} $(date '+%F %T') ==="
  python3 bib_demo/eval_steering_bib.py --config "${cfg}" \
    ${k:+--config_override "manifold_dim=${k}"} \
    --checkpoint_path "${ckpt}" \
    --label-stories 400 --samples-per-alpha 24 \
    --alphas=-4,-3,-2,-1,0,1,2,3,4 --orthogonalize \
    --out "bib_demo/logs/dose_${tag}.jsonl" \
    > "bib_demo/logs/dose_${tag}.full.log" 2>&1
  grep -aE "endpoint delta|VERDICT" "bib_demo/logs/dose_${tag}.full.log" | head -2
done
echo "DOSE CHAIN (gpu ${GPU}) DONE"
