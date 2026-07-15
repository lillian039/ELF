#!/usr/bin/env bash
# Sequential k x rho training queue on binky (fallback after cheery's idle GPUs
# were taken). Same matched budget as the original k-sweep: 5 GPUs, global
# batch 80 = 16/GPU. Native binky env (run_train_sm_m2.sh handles ptxas etc.).
# Order: max-contrast first (+0.30), then the rho=0 control, then the rest.
#
# Usage: setsid nohup bash tinystories_demo/queue_rho_binky.sh \
#          > tinystories_demo/logs/rho_queue_binky.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

# 1,3,4,7 fully idle; 0 has a light tenant (2.8GB/4%) - 16/GPU (~18.5GB) still fits.
export CUDA_VISIBLE_DEVICES="0,1,3,4,7"
K="${RHO_K:-64}"

for TAG in ${@:-p030 p000 m030 p015}; do   # tags as args, default all four
  OUT="/mnt/faster3/lc2762/elf_rho_${TAG}_k${K}"
  if [ -d "${OUT}/checkpoint_37500" ]; then
    echo "=== rho ${TAG}: already complete, skipping ==="
    continue
  fi
  echo "=== $(date '+%F %T') rho ${TAG} k=${K} -> ${OUT} ==="
  bash tinystories_demo/run_train_sm_m2.sh \
    --config_override "manifold_dim=${K}" \
    --config_override "data_path=tinystories_demo/data_50k_rho_${TAG}/train" \
    --config_override "eval_data_path=tinystories_demo/data_50k_rho_${TAG}/val" \
    --config_override "save_freq=10" \
    --config_override "output_dir=${OUT}" \
    || echo "!!! rho ${TAG} FAILED ($(date '+%F %T'))"
  sleep 60   # let GPUs drain before the next run
done
echo "=== RHO QUEUE (binky) DONE $(date '+%F %T') ==="
