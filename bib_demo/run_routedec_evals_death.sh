#!/usr/bin/env bash
# Decoder-arm (phi_route=decoder) instrument quadruple, one GPU each:
#   pairs12 (GPU A), natvar (GPU B), channels (GPU C), quality (GPU D).
# All evals are phi_route-aware (condition components exactly as trained);
# eval_channels crosses phi sites itself by design.
# Usage: bash run_routedec_evals_death.sh <gpuA> <gpuB> <gpuC> <gpuD>
set -uo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
A="$1"; B="$2"; C="$3"; D="$4"
DIR=/mnt/faster3/lc2762/elf_route_decoder_k16_s1
CKPT=${DIR}/checkpoint_75000

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache
mkdir -p bib_demo/logs

CUDA_VISIBLE_DEVICES="${A}" setsid nohup python3 src/eval_leakage_pairs.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_route_dec.json \
  > bib_demo/logs/pairs12_route_dec.log 2>&1 < /dev/null &
disown; echo "pairs12 GPU ${A} pid $!"

CUDA_VISIBLE_DEVICES="${B}" setsid nohup python3 src/eval_natural_variation.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  --out paper/natvar_route_dec.json \
  > bib_demo/logs/natvar_route_dec.log 2>&1 < /dev/null &
disown; echo "natvar GPU ${B} pid $!"

CUDA_VISIBLE_DEVICES="${C}" setsid nohup python3 src/eval_channels.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  --out paper/channels_route_dec.json \
  > bib_demo/logs/channels_route_dec.log 2>&1 < /dev/null &
disown; echo "channels GPU ${C} pid $!"

CUDA_VISIBLE_DEVICES="${D}" setsid nohup python3 src/eval_semantic.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  --num-phi 8 --samples-per-phi 16 \
  --out bib_demo/logs/quality_route_dec.jsonl \
  > bib_demo/logs/quality_route_dec.log 2>&1 < /dev/null &
disown; echo "quality GPU ${D} pid $!"
