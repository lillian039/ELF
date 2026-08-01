#!/usr/bin/env bash
# Full depth-window localization sweep on hogfather GPU 3.
# TinyStories k-sweep (comparable to the time-window analysis) then BiB 8.
# eval_leakage_depth.py pins float32 matmuls internally (TF32 breaks its
# equivalence checks); windows default to quartiles + per-layer singletons.
# Usage: ssh hogfather 'bash /mnt/fast0/lc2762/elf/ELF/bib_demo/run_depth_sweep_hogfather.sh'
set -uo pipefail
BASE=/mnt/fast0/lc2762/elf
cd "${BASE}/ELF"
mkdir -p paper bib_demo/logs

source "${BASE}/venv312/bin/activate"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="${BASE}/hf_cache"
export HF_DATASETS_CACHE="${BASE}/hf_cache"

CK="${BASE}/ckpts"
TS_M2=tinystories_demo/train_tinystories_SM-ELF-M2.yml
TS_M1=tinystories_demo/train_tinystories_SM-ELF.yml
BB_M2=bib_demo/train_bib_SM-ELF-M2.yml
BB_M1=bib_demo/train_bib_SM-ELF-M1.yml

run_depth() { # tag cfg k ckpt
  echo "=== depth $1 (k=${3:-0}) $(date '+%F %T') ==="
  python3 src/eval_leakage_depth.py --config "$2" \
    ${3:+--config_override "manifold_dim=$3"} \
    --checkpoint_path "$4" --samples 64 \
    --out "paper/depth_$1.json" 2>&1 \
    | grep -aE "SHAPE_FLOOR|DEPTH_REF|DEPTH_WINDOW|Traceback" | head -22
}

# TinyStories
run_depth ts_m1   "${TS_M1}" ""  "${CK}/elf_tinystories_sm_output/checkpoint_37500"
run_depth ts_k8   "${TS_M2}" 8   "${CK}/elf_tinystories_k8/checkpoint_37500"
run_depth ts_k16  "${TS_M2}" 16  "${CK}/elf_tinystories_k16/checkpoint_37500"
run_depth ts_k64  "${TS_M2}" 64  "${CK}/elf_tinystories_k64_seed2/checkpoint_37500"
run_depth ts_k256 "${TS_M2}" 256 "${CK}/elf_tinystories_k256/checkpoint_37500"
run_depth ts_k512 "${TS_M2}" 512 "${CK}/elf_tinystories_k512/checkpoint_37500"

# Bias in Bios: shim wrapper swaps axes to occupation/gender (round-1 registry)
run_depth_bib() { # tag cfg k ckpt
  echo "=== depth $1 (k=${3:-0}) $(date '+%F %T') ==="
  python3 bib_demo/eval_leakage_depth_bib.py --config "$2" \
    ${3:+--config_override "manifold_dim=$3"} \
    --checkpoint_path "$4" --samples 64 \
    --out "paper/depth_$1.json" 2>&1 \
    | grep -aE "SHAPE_FLOOR|DEPTH_REF|DEPTH_WINDOW|Traceback" | head -22
}
run_depth_bib bib_m1_s1  "${BB_M1}" ""  "${CK}/elf_bib_m1_s1/checkpoint_37500"
run_depth_bib bib_m1_s2  "${BB_M1}" ""  "${CK}/elf_bib_m1_s2/checkpoint_150000"
run_depth_bib bib_k16_s1 "${BB_M2}" 16  "${CK}/elf_bib_k16_s1/checkpoint_37500"
run_depth_bib bib_k16_s2 "${BB_M2}" 16  "${CK}/elf_bib_k16_s2/checkpoint_150000"
run_depth_bib bib_k64_s1 "${BB_M2}" 64  "${CK}/elf_bib_k64_s1/checkpoint_150000"
run_depth_bib bib_k64_s2 "${BB_M2}" 64  "${CK}/elf_bib_k64_s2/checkpoint_37500"
run_depth_bib bib_k256_s1 "${BB_M2}" 256 "${CK}/elf_bib_k256_s1/checkpoint_37500"
run_depth_bib bib_k256_s2 "${BB_M2}" 256 "${CK}/elf_bib_k256_s2/checkpoint_37500"

echo "=== DEPTH SWEEP DONE $(date '+%F %T') ==="
