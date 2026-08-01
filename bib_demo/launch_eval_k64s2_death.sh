#!/usr/bin/env bash
# Final fleet eval: k64 s2 continuous + probe on death GPU 0.
# Run via: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_eval_k64s2_death.sh'
set -uo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

CKPT=/mnt/faster3/lc2762/elf_bib_k64_s2/checkpoint_37500

echo "=== continuous k64_s2 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_continuous_bib.py \
  --config bib_demo/train_bib_SM-ELF-M2.yml \
  --config_override "manifold_dim=64" \
  --checkpoint_path "${CKPT}" --seeds 5 2>&1 \
  | grep -aE "LEAKAGE_SUMMARY|Traceback|Error" | tail -3

echo "=== probe k64_s2 $(date '+%F %T') ==="
python3 bib_demo/eval_superposition_bib.py \
  --config bib_demo/train_bib_SM-ELF-M2.yml \
  --config_override "manifold_dim=64" \
  --checkpoint_path "${CKPT}" --seeds 5 2>&1 \
  | grep -aE "SUPERPOSITION_SUMMARY|Traceback|Error" | tail -3

echo "=== K64S2 EVAL DONE $(date '+%F %T') ==="
