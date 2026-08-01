#!/usr/bin/env bash
# CF model evaluation chain, death: waits for elf_cf1_k64/checkpoint_75000,
# then runs probe + natvar + pairs12 + terminal-window trajectory on one GPU.
# Usage (on death): nohup bash .../cf_eval_death.sh <gpu> > log 2>&1 &
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

D=/mnt/faster3/lc2762/elf_cf1_k64
until [ -d "${D}/checkpoint_75000" ]; do sleep 120; done
# let the queue's next run start and the trajectory jobs drain
sleep 60
mkdir -p tinystories_demo/logs

echo "=== $(date '+%F %T') CF1 probe ==="
python3 src/eval_superposition.py --config ${D}/config.yml \
  --checkpoint_path ${D}/checkpoint_75000 \
  2>&1 | tee tinystories_demo/logs/cf1_sup.log | grep -aE "SUPERPOSITION_SUMMARY" || true

echo "=== $(date '+%F %T') CF1 natvar ==="
python3 src/eval_natural_variation.py --config ${D}/config.yml \
  --checkpoint_path ${D}/checkpoint_75000 --out paper/natvar_cf1.json \
  2>&1 | tee tinystories_demo/logs/natvar_cf1.log | grep -aE "NATVAR_SUMMARY" || true

echo "=== $(date '+%F %T') CF1 pairs12 ==="
python3 src/eval_leakage_pairs.py --config ${D}/config.yml \
  --checkpoint_path ${D}/checkpoint_75000 \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_cf1.json \
  2>&1 | tee tinystories_demo/logs/pairs12_cf1.log | grep -aE "LEAKPAIR_SUMMARY" || true

echo "=== $(date '+%F %T') CF1 terminal-window trajectory ==="
python3 src/eval_leakage_trajectory.py --config ${D}/config.yml \
  --checkpoint_path ${D}/checkpoint_75000 \
  --windows 0:0.25,0.25:0.5,0.5:0.75,0.75:0.96,0.96:1.01 --window-mode index \
  --out paper/traj_cf1.json \
  2>&1 | tee tinystories_demo/logs/traj_cf1.log | grep -aE "TRAJ_SUMMARY|WINDOW_SUMMARY" || true

echo "CF1 EVAL CHAIN DONE $(date '+%F %T')"
