#!/usr/bin/env bash
# Extended-target directed-pair leakage eval (fragility-prediction experiment).
# Sources stay the 4 core attributes (same generation cost as pairs_*.json);
# every generation is additionally scored against food/weather/vehicle/dialogue
# classifiers, and source axes are decontaminated against all 8 attribute axes
# (--decon all). Results -> paper/pairs8_<tag>.json.
#
# Usage: bash tinystories_demo/pairs8_eval.sh <gpu> <tag:dir> [<tag:dir> ...]
#   e.g. bash tinystories_demo/pairs8_eval.sh 2 k256:/mnt/faster3/lc2762/from_death/elf_tinystories_k256
# A tag suffixed with '@core' runs the --decon core control instead.
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
  decon="all"
  if [[ "${tag}" == *"@core" ]]; then decon="core"; tag="${tag%@core}_deconcore"; fi
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  echo "=== pairs8 eval ${tag} (decon=${decon}): ${ckpt} ==="
  python3 src/eval_leakage_pairs.py \
    --config "${dir}/config.yml" \
    --checkpoint_path "${ckpt}" \
    --seeds 5 --samples-per-alpha 24 \
    --targets extended --decon "${decon}" \
    --out "paper/pairs8_${tag}.json" \
    2>&1 | tee "tinystories_demo/logs/pairs8_${tag}.log" \
    | grep -E "LEAKPAIR_SUMMARY|label balance|===" || true
done
echo "PAIRS8 CHAIN (gpu ${GPU}) DONE"
