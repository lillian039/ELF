#!/usr/bin/env bash
# Overnight chain on hogfather GPU 3:
#   Phase 1: carrier (denoiser vs decoder) 2x2 decomposition, all 14 models.
#   Phase 2: ELF-M scale anchor, TinyStories M1 then k16 (seed 1), matched
#            budget (60 epochs, effective batch 80, single H200).
# Usage: ssh hogfather 'bash /mnt/fast0/lc2762/elf/ELF/bib_demo/run_night_hogfather.sh'
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

run_ch() { # script tag cfg k ckpt
  echo "=== channels $2 (k=${4:-0}) $(date '+%F %T') ==="
  python3 "$1" --config "$3" \
    ${4:+--config_override "manifold_dim=$4"} \
    --checkpoint_path "$5" \
    --out "paper/channels_$2.json" 2>&1 \
    | grep -aE "CHANNEL|Traceback" | head -8
}

echo "===== PHASE 1: carrier decomposition $(date '+%F %T') ====="
run_ch src/eval_channels.py ts_m1   "${TS_M1}" ""  "${CK}/elf_tinystories_sm_output/checkpoint_37500"
run_ch src/eval_channels.py ts_k8   "${TS_M2}" 8   "${CK}/elf_tinystories_k8/checkpoint_37500"
run_ch src/eval_channels.py ts_k16  "${TS_M2}" 16  "${CK}/elf_tinystories_k16/checkpoint_37500"
run_ch src/eval_channels.py ts_k64  "${TS_M2}" 64  "${CK}/elf_tinystories_k64_seed2/checkpoint_37500"
run_ch src/eval_channels.py ts_k256 "${TS_M2}" 256 "${CK}/elf_tinystories_k256/checkpoint_37500"
run_ch src/eval_channels.py ts_k512 "${TS_M2}" 512 "${CK}/elf_tinystories_k512/checkpoint_37500"
run_ch bib_demo/eval_channels_bib.py bib_m1_s1   "${BB_M1}" ""  "${CK}/elf_bib_m1_s1/checkpoint_37500"
run_ch bib_demo/eval_channels_bib.py bib_m1_s2   "${BB_M1}" ""  "${CK}/elf_bib_m1_s2/checkpoint_150000"
run_ch bib_demo/eval_channels_bib.py bib_k16_s1  "${BB_M2}" 16  "${CK}/elf_bib_k16_s1/checkpoint_37500"
run_ch bib_demo/eval_channels_bib.py bib_k16_s2  "${BB_M2}" 16  "${CK}/elf_bib_k16_s2/checkpoint_150000"
run_ch bib_demo/eval_channels_bib.py bib_k64_s1  "${BB_M2}" 64  "${CK}/elf_bib_k64_s1/checkpoint_150000"
run_ch bib_demo/eval_channels_bib.py bib_k64_s2  "${BB_M2}" 64  "${CK}/elf_bib_k64_s2/checkpoint_37500"
run_ch bib_demo/eval_channels_bib.py bib_k256_s1 "${BB_M2}" 256 "${CK}/elf_bib_k256_s1/checkpoint_37500"
run_ch bib_demo/eval_channels_bib.py bib_k256_s2 "${BB_M2}" 256 "${CK}/elf_bib_k256_s2/checkpoint_37500"
echo "===== PHASE 1 DONE $(date '+%F %T') ====="

echo "===== PHASE 2: ELF-M anchors $(date '+%F %T') ====="
python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF.yml \
  --config_override "model=ELF-M" \
  --config_override "seed=42" \
  --config_override "save_freq=10" \
  --config_override "global_batch_size=40" \
  --config_override "grad_accum_steps=2" \
  --config_override "warmup_steps=2000" \
  --config_override "output_dir=${BASE}/ckpts/elfm_ts_m1_s1" \
  2>&1 | tail -3
echo "=== ELF-M M1 done, starting k16 $(date '+%F %T') ==="
python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --config_override "model=ELF-M" \
  --config_override "manifold_dim=16" \
  --config_override "seed=42" \
  --config_override "save_freq=10" \
  --config_override "global_batch_size=40" \
  --config_override "grad_accum_steps=2" \
  --config_override "warmup_steps=2000" \
  --config_override "output_dir=${BASE}/ckpts/elfm_ts_k16_s1" \
  2>&1 | tail -3
echo "===== NIGHT CHAIN DONE $(date '+%F %T') ====="
