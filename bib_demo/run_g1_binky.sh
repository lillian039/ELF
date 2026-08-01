#!/usr/bin/env bash
# G1 gate eval chain for the BiB replication (paper/bib_registration.md):
#   [1] instrument checks (lexicon agreements + frozen-space gender AUC)
#   [2] eval_semantic fidelity/distinct-2 on the M1 s1 epoch-30 checkpoint
#   [3] orthogonalized occupation-axis steering sweep (on-target control)
# Usage: bash bib_demo/run_g1_binky.sh [gpu] [checkpoint]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-5}"
CKPT="${2:-/mnt/faster3/lc2762/elf_bib_m1_s1/checkpoint_18750}"
CFG=bib_demo/train_bib_SM-ELF-M1.yml

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

echo "=== G1 [1/3] instruments $(date '+%F %T') ==="
python3 bib_demo/g1_instruments.py --config "${CFG}" --n-train 20000 \
  || echo "!!! instruments FAILED"

echo "=== G1 [2/3] fidelity (eval_semantic) $(date '+%F %T') ==="
# eval_use_ema=false in config (EMA undertrained at epoch 30 anyway).
python3 src/eval_semantic.py --config "${CFG}" \
  --checkpoint_path "${CKPT}" \
  --num-phi 8 --samples-per-phi 16 \
  --out bib_demo/logs/g1_semantic_samples.jsonl \
  || echo "!!! eval_semantic FAILED"

echo "=== G1 [3/3] steering sweep (orthogonalized occ axis) $(date '+%F %T') ==="
python3 bib_demo/eval_steering_bib.py --config "${CFG}" \
  --checkpoint_path "${CKPT}" \
  --label-stories 400 --samples-per-alpha 24 \
  --alphas=-3,-2,-1,0,1,2,3 --orthogonalize \
  --out bib_demo/logs/g1_steering_samples.jsonl \
  || echo "!!! steering FAILED"

echo "=== G1 CHAIN DONE $(date '+%F %T') ==="
