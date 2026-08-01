#!/usr/bin/env bash
# BiB leakage eval chain (run ONLY after the Predictions section of
# paper/bib_registration.md is filled -- prereg ordering):
#   [1] continuous logit-shift, M1 s1   (primary endpoint, half 1)
#   [2] continuous logit-shift, k16 s1  (primary endpoint, half 2)
#   [3] directed pairs k16 s1, occupation source -> all targets (fragility test)
#   [4] directed pairs M1 s1 (secondary)
# Usage: bash bib_demo/run_leakage_binky.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-5}"

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"

M1_CKPT=/mnt/faster3/lc2762/elf_bib_m1_s1/checkpoint_37500
K16_CKPT=/mnt/faster3/lc2762/elf_bib_k16_s1/checkpoint_37500

echo "=== [1/4] continuous M1 s1 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_continuous_bib.py \
  --config bib_demo/train_bib_SM-ELF-M1.yml \
  --checkpoint_path "${M1_CKPT}" --seeds 5 \
  || echo "!!! continuous M1 FAILED"

echo "=== [2/4] continuous k16 s1 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_continuous_bib.py \
  --config bib_demo/train_bib_SM-ELF-M2.yml \
  --config_override "manifold_dim=16" \
  --checkpoint_path "${K16_CKPT}" --seeds 5 \
  || echo "!!! continuous k16 FAILED"

echo "=== [3/4] pairs k16 s1 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_pairs_bib.py \
  --config bib_demo/train_bib_SM-ELF-M2.yml \
  --config_override "manifold_dim=16" \
  --checkpoint_path "${K16_CKPT}" --seeds 5 \
  --sources sentiment --targets extended \
  --out paper/bib_pairs_k16s1.json \
  || echo "!!! pairs k16 FAILED"

echo "=== [4/4] pairs M1 s1 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_pairs_bib.py \
  --config bib_demo/train_bib_SM-ELF-M1.yml \
  --checkpoint_path "${M1_CKPT}" --seeds 5 \
  --sources sentiment --targets extended \
  --out paper/bib_pairs_m1s1.json \
  || echo "!!! pairs M1 FAILED"

echo "=== LEAKAGE CHAIN DONE $(date '+%F %T') ==="
