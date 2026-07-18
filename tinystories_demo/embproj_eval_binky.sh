#!/usr/bin/env bash
# Inference-time generation-pathway decontamination test on the k=64 baseline:
# same directed-pair protocol, but the injected embedding delta U(alpha*u) is
# projected off the off-target attributes' embedding classifier directions.
# Runs the treated condition and a same-code-rerun control back to back.
set -uo pipefail
cd "$(dirname "$0")/.."

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${1:?gpu}"
export HF_HOME=/mnt/faster3/lc2762/hf_cache

D=/mnt/faster3/lc2762/from_death/elf_tinystories_sm_m2_output
mkdir -p dec_results tinystories_demo/logs

echo "=== $(date '+%F %T') k64 baseline, emb-project=offtargets ==="
python3 src/eval_leakage_pairs.py \
  --config $D/config.yml --checkpoint_path $D/checkpoint_37500 \
  --seeds 5 --samples-per-alpha 24 --emb-project offtargets \
  --out dec_results/pairs_k64_embproj.json \
  2>&1 | tee tinystories_demo/logs/embproj_k64.log | grep -E "LEAKPAIR_SUMMARY" || true

echo "=== $(date '+%F %T') k64 baseline, emb-project=none (same-codebase control) ==="
python3 src/eval_leakage_pairs.py \
  --config $D/config.yml --checkpoint_path $D/checkpoint_37500 \
  --seeds 5 --samples-per-alpha 24 --emb-project none \
  --out dec_results/pairs_k64_ctrlrerun.json \
  2>&1 | tee tinystories_demo/logs/embproj_k64_ctrl.log | grep -E "LEAKPAIR_SUMMARY" || true
echo "=== EMBPROJ DONE $(date '+%F %T') ==="
