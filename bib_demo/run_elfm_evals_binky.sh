#!/usr/bin/env bash
# ELF-M scale-anchor evals on binky, one GPU each (local config + overrides;
# the checkpoint's own config.yml carries hogfather-absolute paths).
# Usage: bash bib_demo/run_elfm_evals_binky.sh <tag> <ckpt_dir> <gpu_pairs> <gpu_quality> <gpu_steer> [k]
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="$1"; DIR="$2"; G1="$3"; G2="$4"; G3="$5"; K="${6:-}"

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

CKPT=$(ls -d "${DIR}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
CFG=tinystories_demo/train_tinystories_SM-ELF.yml
[ -n "${K}" ] && CFG=tinystories_demo/train_tinystories_SM-ELF-M2.yml
OV=(--config_override "model=ELF-M")
[ -n "${K}" ] && OV+=(--config_override "manifold_dim=${K}")
mkdir -p bib_demo/logs

CUDA_VISIBLE_DEVICES="${G1}" setsid nohup python3 src/eval_leakage_pairs.py \
  --config "${CFG}" "${OV[@]}" --checkpoint_path "${CKPT}" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out "paper/pairs12_${TAG}.json" \
  > "bib_demo/logs/pairs12_${TAG}.log" 2>&1 < /dev/null &
disown; echo "pairs12 on GPU ${G1}, pid $!"

CUDA_VISIBLE_DEVICES="${G2}" setsid nohup python3 src/eval_semantic.py \
  --config "${CFG}" "${OV[@]}" --checkpoint_path "${CKPT}" \
  --num-phi 8 --samples-per-phi 16 \
  --out "bib_demo/logs/quality_${TAG}.jsonl" \
  > "bib_demo/logs/quality_${TAG}.log" 2>&1 < /dev/null &
disown; echo "quality on GPU ${G2}, pid $!"

CUDA_VISIBLE_DEVICES="${G3}" setsid nohup python3 src/eval_steering.py \
  --config "${CFG}" "${OV[@]}" --checkpoint_path "${CKPT}" \
  --label-stories 400 --samples-per-alpha 24 --orthogonalize \
  --out "bib_demo/logs/steer_${TAG}.jsonl" \
  > "bib_demo/logs/steer_${TAG}.log" 2>&1 < /dev/null &
disown; echo "steering on GPU ${G3}, pid $!"
