#!/usr/bin/env bash
# Queue one BiB run behind whatever currently occupies the given GPUs
# (kseed2 pattern): poll until every listed GPU is <500MiB used, then train.
# Skips if the final checkpoint already exists.
#
# Usage: nohup bash bib_demo/queue_bib.sh <M1|M2> <gpus> <final_step> \
#          [--config_override ...]... > log 2>&1 &
#   e.g.  bib_demo/queue_bib.sh M2 0,1,2,3 37500 \
#           --config_override "manifold_dim=256" \
#           --config_override "seed=42" \
#           --config_override "output_dir=/mnt/faster3/lc2762/elf_bib_k256_s1"
set -uo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:?M1 or M2}"; GPUS="${2:?gpu list}"; FINAL="${3:?final step}"
shift 3

OUT=""
for arg in "$@"; do
  case "$arg" in output_dir=*) OUT="${arg#output_dir=}";; esac
done
if [ -n "${OUT}" ] && [ -d "${OUT}/checkpoint_${FINAL}" ]; then
  echo "=== already complete (${OUT}/checkpoint_${FINAL}), skipping ==="
  exit 0
fi

echo "=== $(date '+%F %T') waiting for GPUs ${GPUS} to free up ==="
while true; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g=",${GPUS}," 'index(g, ","$1",") && $2>500 {n++} END {print n+0}')
  [ "${busy}" -eq 0 ] && break
  sleep 300
done
echo "=== $(date '+%F %T') GPUs ${GPUS} free, starting ==="

export CUDA_VISIBLE_DEVICES="${GPUS}"
bash bib_demo/run_train_bib.sh "${VARIANT}" "$@" \
  || echo "!!! queued BiB run FAILED ($(date '+%F %T'))"
echo "=== BIB QUEUE DONE $(date '+%F %T') ==="
