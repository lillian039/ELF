#!/bin/bash
#
# Progressive distillation (5 rounds) for ELF-B on OpenWebText.
#
# Both the student's t_s and the teacher's per-row sub-step times are drawn on a
# continuous schedule (logit-normal by default) rather than uniformly in t:
#   t_s ~ sigmoid(N(P_mean, P_std)) * (1 - pd_step_size)
#   teacher sub-steps placed on the schedule inside [t_s, t_t]
#
# Per round, pd_step_size * pd_teacher_steps = 1/64, so the teacher's total
# Euler resolution stays constant across the curriculum:
#
#   round | student steps | teacher substeps | pd_step_size
#   ----- | ------------- | ---------------- | ------------
#     1   |      16       |         4        |    1/16
#     2   |       8       |         8        |    1/8
#     3   |       4       |        16        |    1/4
#     4   |       2       |        32        |    1/2
#     5   |       1       |        64        |    1.0
#
# Round 1 warm-starts the student from the teacher (same architecture); each
# later round warm-starts from the previous round's checkpoint.
#
# Override via env vars:
#     TEACHER_PATH=...  INIT_STUDENT=...  CONFIG=...  BASE_DIR=... \
#     STUDENT_SCHEDULE=logit_normal  TEACHER_SCHEDULE=logit_normal \
#     START_ROUND=3  NGPU=8  bash scripts/progressive_distillation.sh

set -euo pipefail

# Run from the repo root regardless of where this is invoked from.
cd "$(dirname "$0")/.."

TEACHER_PATH=${TEACHER_PATH:-embedded-language-flows/ELF-B-owt-torch}
BASE_DIR=${BASE_DIR:-outputs/elf_b-owt_progressive_distillation}
CONFIG=${CONFIG:-src/configs/training_configs/distill_curriculum_ELF-B.yml}

# TEACHER_PATH may be a local dir or a HF repo id (downloaded to the local HF
# cache by the loader; private repos need `hf auth login` / HF_TOKEN).

# Round 1 warm-start. Empty -> use the config's initialize_student_from_teacher
# (copies the in-memory teacher). Set to a checkpoint to start somewhere else.
INIT_STUDENT=${INIT_STUDENT:-}

# Schedules for the student t_s and the teacher sub-steps. Both share P_mean /
# P_std from denoiser_p_mean / denoiser_p_std in the YAML. Set either to
# "uniform" to fall back to uniform-in-t for that side.
STUDENT_SCHEDULE=${STUDENT_SCHEDULE:-logit_normal}
TEACHER_SCHEDULE=${TEACHER_SCHEDULE:-logit_normal}

# Pair-wise: pd_step_size * pd_teacher_steps = 1/64 every round.
STUDENT_STEPS=(16 8 4 2 1)
TEACHER_SUBSTEPS=(4 8 16 32 64)
NUM_ROUNDS=${#STUDENT_STEPS[@]}
START_ROUND=${START_ROUND:-1}
NGPU=${NGPU:-8}

if [[ "${#STUDENT_STEPS[@]}" -ne "${#TEACHER_SUBSTEPS[@]}" ]]; then
    echo "[progressive-distillation] STUDENT_STEPS and TEACHER_SUBSTEPS must be the same length" >&2
    exit 1
fi

PREV_STUDENT="$INIT_STUDENT"
if [[ "$START_ROUND" -gt 1 ]]; then
    prev_idx=$((START_ROUND - 1))
    PREV_STUDENT="${BASE_DIR}/round${prev_idx}_${STUDENT_STEPS[$((prev_idx - 1))]}step"
    echo "[progressive-distillation] resuming at round $START_ROUND, prev student = $PREV_STUDENT"
elif [[ -n "$PREV_STUDENT" ]]; then
    echo "[progressive-distillation] round 1 warm-start from: $PREV_STUDENT"
else
    echo "[progressive-distillation] round 1 warm-start from teacher (initialize_student_from_teacher)"
fi

for ((k = START_ROUND; k <= NUM_ROUNDS; k++)); do
    N=${STUDENT_STEPS[$((k - 1))]}
    T=${TEACHER_SUBSTEPS[$((k - 1))]}
    step_size=$(python -c "print(1.0 / $N)")
    out_dir="${BASE_DIR}/round${k}_${N}step"
    run_name="$(basename "$BASE_DIR")_round${k}_${N}step"

    extra=()
    if [[ -n "$PREV_STUDENT" ]]; then
        extra+=(--config_override "student_init_from_checkpoint=$PREV_STUDENT")
    fi

    echo "================================================================="
    echo "[progressive-distillation] round $k: student=${N}step, teacher=${T}substeps, step_size=${step_size}"
    echo "[progressive-distillation] schedules: student=${STUDENT_SCHEDULE}, teacher=${TEACHER_SCHEDULE}"
    echo "[progressive-distillation] teacher : $TEACHER_PATH"
    echo "[progressive-distillation] output  : $out_dir"
    echo "[progressive-distillation] warm-start: ${PREV_STUDENT:-<teacher>}"
    echo "================================================================="

    NGPU="$NGPU" bash scripts/launch.sh distill "$CONFIG" \
        --config_override "teacher_path=$TEACHER_PATH" \
        --config_override "output_dir=$out_dir" \
        --config_override "wandb_run_name=$run_name" \
        --config_override "pd_step_size=$step_size" \
        --config_override "pd_teacher_steps=$T" \
        --config_override "pd_student_schedule=$STUDENT_SCHEDULE" \
        --config_override "pd_teacher_schedule=$TEACHER_SCHEDULE" \
        ${extra[@]+"${extra[@]}"}

    PREV_STUDENT="$out_dir"
done

echo "[progressive-distillation] done — final 1-step student at: $PREV_STUDENT"
