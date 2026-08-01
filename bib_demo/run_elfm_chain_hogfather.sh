#!/usr/bin/env bash
# ELF-M (24-layer) scale-anchor chain on hogfather GPU 3, sequential:
#   M1 s1 -> k16 s1 -> M1 s2 -> k16 s2   (matched budget, eff. batch 80 via accum)
# Replaces the crashed Phase 2 / Phase 3b (sacrebleu missing; now installed).
# Usage: ssh hogfather 'nohup bash /mnt/fast0/lc2762/elf/ELF/bib_demo/run_elfm_chain_hogfather.sh > /mnt/fast0/lc2762/elf/ELF/bib_demo/logs/elfm_chain.log 2>&1 &'
set -uo pipefail
BASE=/mnt/fast0/lc2762/elf
cd "${BASE}/ELF"
source "${BASE}/venv312/bin/activate"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="${BASE}/hf_cache"
export HF_DATASETS_CACHE="${BASE}/hf_cache"

for spec in "elfm_ts_m1_s1::42" "elfm_ts_k16_s1:16:42" "elfm_ts_m1_s2::43" "elfm_ts_k16_s2:16:43"; do
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
    2>&1 | tail -3
  echo "=== done ${tag} $(date '+%F %T') ==="
done
echo "===== ELFM CHAIN DONE $(date '+%F %T') ====="
