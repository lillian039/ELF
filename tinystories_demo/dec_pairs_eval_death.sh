#!/usr/bin/env bash
# Superposition + directed-pair leakage eval for one decorrelation model, on
# death. Complements the within-k analysis: did the decorrelation regularizer
# lower measured sent-gender interference while raising leakage (pair-level
# dissociation)?
#
# Usage (on death): bash tinystories_demo/dec_pairs_eval_death.sh <dir> <step> <tag> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."

DIR="${1:?model dir}"; STEP="${2:?checkpoint step}"; TAG="${3:?tag}"; GPU="${4:-0}"

# NOTE: bare env on purpose. On death, prepending the pip nvidia libs to
# LD_LIBRARY_PATH segfaults jax at GPU init, and the ptxas XLA_FLAGS/PATH hack
# hangs it; the untouched non-interactive ssh env works.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"

mkdir -p tinystories_demo/logs dec_results
CKPT="${DIR}/checkpoint_${STEP}"

echo "=== $(date '+%F %T') ${TAG} superposition @${STEP} ==="
python3 src/eval_superposition.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  2>&1 | tee "tinystories_demo/logs/dec_${TAG}_sup.log" | grep -E "SUPERPOSITION_SUMMARY" || true

echo "=== $(date '+%F %T') ${TAG} directed pairs @${STEP} ==="
python3 src/eval_leakage_pairs.py \
  --config "${DIR}/config.yml" --checkpoint_path "${CKPT}" \
  --seeds 5 --samples-per-alpha 24 \
  --out "dec_results/pairs_${TAG}.json" \
  2>&1 | tee "tinystories_demo/logs/dec_${TAG}_pairs.log" | grep -E "LEAKPAIR_SUMMARY" || true
echo "=== DEC EVAL ${TAG} DONE $(date '+%F %T') ==="
