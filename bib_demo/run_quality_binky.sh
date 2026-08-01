#!/usr/bin/env bash
# Per-seed quality gate table (fidelity / within / across / distinct-2) for
# every model whose numbers appear in the paper: all 8 BiB + the extra
# TinyStories seeds. Runs eval_semantic on one binky GPU.
# Usage: bash bib_demo/run_quality_binky.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-2}"

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

CK=/mnt/faster3/lc2762
TS_M2=tinystories_demo/train_tinystories_SM-ELF-M2.yml
BB_M2=bib_demo/train_bib_SM-ELF-M2.yml
BB_M1=bib_demo/train_bib_SM-ELF-M1.yml

run_q() { # tag cfg k ckpt
  echo "=== quality $1 (k=${3:-0}) $(date '+%F %T') ==="
  python3 src/eval_semantic.py --config "$2" \
    ${3:+--config_override "manifold_dim=$3"} \
    --checkpoint_path "$4" \
    --num-phi 8 --samples-per-phi 16 \
    --out "bib_demo/logs/quality_$1.jsonl" 2>&1 \
    | grep -aE "fidelity|within|across|distinct|VERDICT|Traceback" | head -6
}

# bib_m1_s1 / bib_m1_s2 already completed in the first pass (see quality.log)
run_q bib_k16_s1  "${BB_M2}" 16  "${CK}/elf_bib_k16_s1/checkpoint_37500"
run_q bib_k16_s2  "${BB_M2}" 16  "${CK}/elf_bib_k16_s2/checkpoint_150000"
run_q bib_k64_s1  "${BB_M2}" 64  "${CK}/elf_bib_k64_s1/checkpoint_150000"
run_q bib_k64_s2  "${BB_M2}" 64  "${CK}/elf_bib_k64_s2/checkpoint_37500"
run_q bib_k256_s1 "${BB_M2}" 256 "${CK}/elf_bib_k256_s1/checkpoint_37500"
run_q bib_k256_s2 "${BB_M2}" 256 "${CK}/elf_bib_k256_s2/checkpoint_37500"
run_q ts_k16_s2   "${TS_M2}" 16  "${CK}/elf_tinystories_k16_seed2/checkpoint_37500"
run_q ts_k16_s3   "${TS_M2}" 16  "${CK}/elf_tinystories_k16_seed3/checkpoint_37500"
run_q ts_k64_s2   "${TS_M2}" 64  "${CK}/elf_tinystories_k64_seed2/checkpoint_37500"

echo "=== QUALITY TABLE DONE $(date '+%F %T') ==="
