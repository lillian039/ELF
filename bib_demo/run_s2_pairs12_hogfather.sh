#!/usr/bin/env bash
# ELF-M seed-2 pairs12 on hogfather GPU 3 (H200), serial: m1_s2 then k16_s2.
# Checkpoints are local to hogfather; results -> paper/pairs12_elfm_*_s2.json.
set -uo pipefail
BASE=/mnt/fast0/lc2762/elf
cd "${BASE}/ELF"
source "${BASE}/venv312/bin/activate"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="${BASE}/hf_cache"
export HF_DATASETS_CACHE="${BASE}/hf_cache"
mkdir -p bib_demo/logs

echo "=== pairs12 elfm_m1_s2 $(date '+%F %T') ==="
python3 src/eval_leakage_pairs.py \
  --config tinystories_demo/train_tinystories_SM-ELF.yml \
  --config_override model=ELF-M \
  --checkpoint_path "${BASE}/ckpts/elfm_ts_m1_s2/checkpoint_75000" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_elfm_m1_s2.json 2>&1 | tail -3

echo "=== pairs12 elfm_k16_s2 $(date '+%F %T') ==="
python3 src/eval_leakage_pairs.py \
  --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --config_override model=ELF-M --config_override manifold_dim=16 \
  --checkpoint_path "${BASE}/ckpts/elfm_ts_k16_s2/checkpoint_75000" \
  --seeds 5 --samples-per-alpha 24 --targets extended2 --decon all \
  --out paper/pairs12_elfm_k16_s2.json 2>&1 | tail -3

echo "=== S2 PAIRS12 DONE $(date '+%F %T') ==="
