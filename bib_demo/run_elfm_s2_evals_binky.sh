#!/usr/bin/env bash
# ELF-M seed-2 anchor evals (M1 s2 + k16 s2), binky GPUs 0-3:
#   GPU A: pairs12 m1_s2      GPU B: pairs12 k16_s2
#   GPU C: quality->natvar m1_s2   GPU D: quality->natvar k16_s2
# Local configs + overrides (checkpoint config.yml has hogfather paths).
set -uo pipefail
cd "$(dirname "$0")/.."
A="$1"; B="$2"; C="$3"; D="$4"

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

M1CK=/mnt/faster3/lc2762/elfm_ts_m1_s2/checkpoint_75000
K16CK=/mnt/faster3/lc2762/elfm_ts_k16_s2/checkpoint_75000
M1CFG=tinystories_demo/train_tinystories_SM-ELF.yml
M2CFG=tinystories_demo/train_tinystories_SM-ELF-M2.yml

CUDA_VISIBLE_DEVICES="${A}" setsid nohup python3 src/eval_leakage_pairs.py \
  --config "${M1CFG}" --config_override model=ELF-M --checkpoint_path "${M1CK}" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_elfm_m1_s2.json > bib_demo/logs/pairs12_elfm_m1_s2.log 2>&1 < /dev/null &
disown; echo "pairs12 m1_s2 GPU ${A} pid $!"

CUDA_VISIBLE_DEVICES="${B}" setsid nohup python3 src/eval_leakage_pairs.py \
  --config "${M2CFG}" --config_override model=ELF-M --config_override manifold_dim=16 \
  --checkpoint_path "${K16CK}" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_elfm_k16_s2.json > bib_demo/logs/pairs12_elfm_k16_s2.log 2>&1 < /dev/null &
disown; echo "pairs12 k16_s2 GPU ${B} pid $!"

CUDA_VISIBLE_DEVICES="${C}" setsid nohup bash -c "
python3 src/eval_semantic.py --config ${M1CFG} --config_override model=ELF-M \
  --checkpoint_path ${M1CK} --num-phi 8 --samples-per-phi 16 \
  --out bib_demo/logs/quality_elfm_m1_s2.jsonl > bib_demo/logs/quality_elfm_m1_s2.log 2>&1
python3 src/eval_natural_variation.py --config ${M1CFG} --config_override model=ELF-M \
  --checkpoint_path ${M1CK} --out paper/natvar_elfm_m1_s2.json \
  > bib_demo/logs/natvar_elfm_m1_s2.log 2>&1
" > /dev/null 2>&1 < /dev/null &
disown; echo "quality+natvar m1_s2 GPU ${C} pid $!"

CUDA_VISIBLE_DEVICES="${D}" setsid nohup bash -c "
python3 src/eval_semantic.py --config ${M2CFG} --config_override model=ELF-M --config_override manifold_dim=16 \
  --checkpoint_path ${K16CK} --num-phi 8 --samples-per-phi 16 \
  --out bib_demo/logs/quality_elfm_k16_s2.jsonl > bib_demo/logs/quality_elfm_k16_s2.log 2>&1
python3 src/eval_natural_variation.py --config ${M2CFG} --config_override model=ELF-M --config_override manifold_dim=16 \
  --checkpoint_path ${K16CK} --out paper/natvar_elfm_k16_s2.json \
  > bib_demo/logs/natvar_elfm_k16_s2.log 2>&1
" > /dev/null 2>&1 < /dev/null &
disown; echo "quality+natvar k16_s2 GPU ${D} pid $!"
