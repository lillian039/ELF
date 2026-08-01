#!/usr/bin/env bash
# Launch the angua-side BiB runs: three single-GPU trainings (A5000) with
# grad accumulation keeping effective batch 80 and optimizer warmup 1000.
#   GPU 1: M1  s2   GPU 6: k16 s2   GPU 7: k64 s1
# Run via: ssh angua 'bash /mnt/faster2/lc2762/ELF/bib_demo/launch_angua.sh'
set -euo pipefail
REPO=/mnt/faster2/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

launch_one() {
  local gpu="$1" name="$2" variant="$3"; shift 3
  : > "bib_demo/logs/train_${name}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" setsid nohup bash "${REPO}/bib_demo/run_train_bib.sh" "${variant}" \
    "$@" \
    --config_override "global_batch_size=20" \
    --config_override "grad_accum_steps=4" \
    --config_override "warmup_steps=4000" \
    --config_override "save_freq=10" \
    --config_override "output_dir=/mnt/faster2/lc2762/elf_bib_${name}" \
    > "bib_demo/logs/train_${name}.log" 2>&1 < /dev/null &
  disown
  echo "launched ${name} on GPU ${gpu}, pid $!"
}

launch_one 1 m1_s2  M1 --config_override "seed=43"
launch_one 6 k16_s2 M2 --config_override "manifold_dim=16" --config_override "seed=43"
launch_one 7 k64_s1 M2 --config_override "manifold_dim=64" --config_override "seed=42"
