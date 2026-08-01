#!/usr/bin/env bash
# BiB k16 seed-3 arbitration chain: waits for the final checkpoint, then on
# one GPU runs [1] continuous gender endpoint, [2] quality gate, [3] natvar
# (severed-code determinant), [4] probe interference.
# Usage: ssh death 'setsid nohup bash .../run_k16s3_evals_death.sh <gpu> > .../logs/k16s3_evals.log 2>&1 &'
set -uo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
GPU="${1:-1}"
DIR=/mnt/faster3/lc2762/elf_bib_k16_s3
CKPT=${DIR}/checkpoint_150000

until [ -d "${CKPT}" ]; do sleep 120; done
sleep 60

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

CFG=bib_demo/train_bib_SM-ELF-M2.yml
OV=(--config_override "manifold_dim=16")

echo "=== continuous k16_s3 $(date '+%F %T') ==="
python3 bib_demo/eval_leakage_continuous_bib.py --config "${CFG}" "${OV[@]}" \
  --checkpoint_path "${CKPT}" --seeds 5 2>&1 \
  | grep -aE "LEAKAGE_SUMMARY|CTRL|control|Traceback" | tail -6

echo "=== quality k16_s3 $(date '+%F %T') ==="
python3 src/eval_semantic.py --config "${CFG}" "${OV[@]}" \
  --checkpoint_path "${CKPT}" --num-phi 8 --samples-per-phi 16 \
  --out bib_demo/logs/quality_bib_k16_s3.jsonl 2>&1 \
  | grep -aE "fidelity|within|across|distinct|VERDICT" | head -6

echo "=== natvar k16_s3 $(date '+%F %T') ==="
python3 bib_demo/eval_natvar_bib.py --config "${CFG}" "${OV[@]}" \
  --checkpoint_path "${CKPT}" \
  --out paper/bib_natvar_k16s3.json 2>&1 \
  | grep -aE "NATVAR_SUMMARY|Traceback" | head -6

echo "=== probe k16_s3 $(date '+%F %T') ==="
python3 bib_demo/eval_superposition_bib.py --config "${CFG}" "${OV[@]}" \
  --checkpoint_path "${CKPT}" --seeds 5 2>&1 \
  | grep -aE "SUPERPOSITION_SUMMARY|Traceback" | tail -3

echo "=== K16S3 EVAL CHAIN DONE $(date '+%F %T') ==="
