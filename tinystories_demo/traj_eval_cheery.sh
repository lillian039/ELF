#!/usr/bin/env bash
# Training-trajectory evals on cheery GPU 7: for each intermediate checkpoint of
# a rho run, measure (a) code-space superposition (Manifold Probe) and (b)
# sentiment<->gender directed leakage. Gives the interference/leakage
# co-evolution over training (auxiliary evidence; fixed classifiers per eval).
#
# Usage (on cheery): bash tinystories_demo/traj_eval_cheery.sh <ckpt_dir> <tag> [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/faster1/lc2762/venv/bin/activate

DIR="${1:?dir with checkpoint_* and config.yml}"
TAG="${2:?tag, e.g. p030}"
GPU="${3:-7}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster1/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster1/lc2762/hf_cache

mkdir -p tinystories_demo/logs traj_results
for CKPT in $(ls -d "${DIR}"/checkpoint_* | sort -t_ -k2 -n); do
  STEP=$(basename "${CKPT}" | cut -d_ -f2)
  echo "=== $(date '+%F %T') traj ${TAG} step ${STEP}: superposition ==="
  python3 src/eval_superposition.py \
    --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
    2>&1 | tee "tinystories_demo/logs/traj_${TAG}_sup_${STEP}.log" | grep -E "SUPERPOSITION_SUMMARY" || true
  echo "=== $(date '+%F %T') traj ${TAG} step ${STEP}: leakage (sent/gender) ==="
  python3 src/eval_leakage_pairs.py \
    --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
    --sources sentiment,gender --seeds 3 --samples-per-alpha 24 \
    --out "traj_results/traj_${TAG}_pairs_${STEP}.json" \
    2>&1 | tee "tinystories_demo/logs/traj_${TAG}_pairs_${STEP}.log" | grep -E "LEAKPAIR_SUMMARY" || true
done
echo "=== TRAJ EVAL ${TAG} DONE $(date '+%F %T') ==="
