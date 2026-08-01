#!/usr/bin/env bash
# Time-resolved (sampling-trajectory) leakage eval, cheery env (faster1 venv).
# Usage: bash tinystories_demo/sample_traj_cheery.sh <gpu> <tag:dir> [...]
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/faster1/lc2762/venv/bin/activate
GPU="$1"; shift

export CUDA_VISIBLE_DEVICES="${GPU}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster1/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster1/lc2762/hf_cache

mkdir -p tinystories_demo/logs
for spec in "$@"; do
  tag="${spec%%:*}"; dir="${spec#*:}"
  ckpt=$(ls -d "${dir}"/checkpoint_* | sort -t_ -k2 -n | tail -1)
  echo "=== traj ${tag}: ${ckpt} ==="
  python3 src/eval_leakage_trajectory.py \
    --config "${dir}/config.yml" --checkpoint_path "${ckpt}" \
    --out "paper/traj_${tag}.json" ${TRAJ_ARGS:-} \
    2>&1 | tee "tinystories_demo/logs/traj_${tag}.log" | grep -aE "TRAJ_SUMMARY|WINDOW_SUMMARY|===" || true
done
echo "TRAJ CHAIN cheery (gpu ${GPU}) DONE"
