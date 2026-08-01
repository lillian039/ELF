#!/usr/bin/env bash
# Round-2 leakage phase on hogfather GPU 3. Run ONLY after the round-2
# predictions block in paper/bib_registration.md is filled (prereg ordering).
# Directed pairs, occupation source, round-2 target registry, all 8 models.
# Usage: ssh hogfather 'bash /mnt/fast0/lc2762/elf/ELF/bib_demo/run_r2_leakage_hogfather.sh'
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
export BIB_TARGET_ROUND=2

CK="${BASE}/ckpts"
M1CFG=bib_demo/train_bib_SM-ELF-M1.yml
M2CFG=bib_demo/train_bib_SM-ELF-M2.yml

run_pairs() { # name cfg k ckpt
  echo "=== r2 pairs $1 $(date '+%F %T') ==="
  python3 bib_demo/eval_leakage_pairs_bib.py --config "$2" \
    ${3:+--config_override "manifold_dim=$3"} \
    --checkpoint_path "$4" --seeds 5 \
    --sources sentiment --targets extended \
    --out "paper/bib_pairs_r2_$1.json" 2>&1 \
    | grep -aE "LEAKPAIR_SUMMARY|Traceback" | tail -12
}
run_pairs m1_s1   "${M1CFG}" ""  "${CK}/elf_bib_m1_s1/checkpoint_37500"
run_pairs m1_s2   "${M1CFG}" ""  "${CK}/elf_bib_m1_s2/checkpoint_150000"
run_pairs k16_s1  "${M2CFG}" 16  "${CK}/elf_bib_k16_s1/checkpoint_37500"
run_pairs k16_s2  "${M2CFG}" 16  "${CK}/elf_bib_k16_s2/checkpoint_150000"
run_pairs k64_s1  "${M2CFG}" 64  "${CK}/elf_bib_k64_s1/checkpoint_150000"
run_pairs k64_s2  "${M2CFG}" 64  "${CK}/elf_bib_k64_s2/checkpoint_37500"
run_pairs k256_s1 "${M2CFG}" 256 "${CK}/elf_bib_k256_s1/checkpoint_37500"
run_pairs k256_s2 "${M2CFG}" 256 "${CK}/elf_bib_k256_s2/checkpoint_37500"

echo "=== R2 LEAKAGE DONE $(date '+%F %T') ==="
