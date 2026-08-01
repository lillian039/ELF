#!/usr/bin/env bash
# Phase 3, queued behind run_night_hogfather.sh (polls for its DONE marker):
#   3a. alpha dose-response steering sweeps (BiB seed-1 fleet, alpha -4..4)
#   3b. ELF-M seed-2 anchors (M1 s2, k16 s2), matched budget
# Usage: ssh hogfather 'nohup bash .../run_night_phase3_hogfather.sh > phase3.log 2>&1 &'
set -uo pipefail
BASE=/mnt/fast0/lc2762/elf
cd "${BASE}/ELF"
mkdir -p paper bib_demo/logs

until grep -aq "NIGHT CHAIN DONE" "${BASE}/ELF/bib_demo/logs/night.log" 2>/dev/null; do
  sleep 600
done

source "${BASE}/venv312/bin/activate"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="${BASE}/hf_cache"
export HF_DATASETS_CACHE="${BASE}/hf_cache"

CK="${BASE}/ckpts"
BB_M2=bib_demo/train_bib_SM-ELF-M2.yml
BB_M1=bib_demo/train_bib_SM-ELF-M1.yml

echo "===== PHASE 3a: alpha dose-response $(date '+%F %T') ====="
run_dose() { # tag cfg k ckpt
  echo "=== dose $1 $(date '+%F %T') ==="
  python3 bib_demo/eval_steering_bib.py --config "$2" \
    ${3:+--config_override "manifold_dim=$3"} \
    --checkpoint_path "$4" \
    --label-stories 400 --samples-per-alpha 24 \
    --alphas=-4,-3,-2,-1,0,1,2,3,4 --orthogonalize \
    --out "bib_demo/logs/dose_$1.jsonl" 2>&1 \
    | grep -aE "^  [+-]|VERDICT|Traceback" | head -12
}
run_dose bib_m1_s1   "${BB_M1}" ""  "${CK}/elf_bib_m1_s1/checkpoint_37500"
run_dose bib_k16_s1  "${BB_M2}" 16  "${CK}/elf_bib_k16_s1/checkpoint_37500"
run_dose bib_k64_s1  "${BB_M2}" 64  "${CK}/elf_bib_k64_s1/checkpoint_150000"
run_dose bib_k256_s1 "${BB_M2}" 256 "${CK}/elf_bib_k256_s1/checkpoint_37500"

echo "===== PHASE 3b: ELF-M seed-2 anchors $(date '+%F %T') ====="
for spec in "elfm_ts_m1_s2::43" "elfm_ts_k16_s2:16:43"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; k="${rest%%:*}"; sd="${rest#*:}"
  cfg=tinystories_demo/train_tinystories_SM-ELF.yml
  [ -n "${k}" ] && cfg=tinystories_demo/train_tinystories_SM-ELF-M2.yml
  echo "=== train ${tag} $(date '+%F %T') ==="
  python3 src/train.py --config "${cfg}" \
    --config_override "model=ELF-M" \
    ${k:+--config_override "manifold_dim=${k}"} \
    --config_override "seed=${sd}" \
    --config_override "save_freq=10" \
    --config_override "global_batch_size=40" \
    --config_override "grad_accum_steps=2" \
    --config_override "warmup_steps=2000" \
    --config_override "output_dir=${BASE}/ckpts/${tag}" \
    2>&1 | tail -2
done
echo "===== PHASE 3 DONE $(date '+%F %T') ====="
