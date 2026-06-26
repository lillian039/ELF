#!/usr/bin/env bash
# Launcher for the PyTorch ELF port.
#
# Single GPU / CPU:
#     bash scripts/launch.sh train   src/configs/training_configs/train_owt_ELF-B.yml
#     bash scripts/launch.sh eval    src/configs/training_configs/train_owt_ELF-B.yml --checkpoint_path embedded-language-flows/ELF-B-owt
#     bash scripts/launch.sh distill src/configs/training_configs/distill_curriculum_ELF-B.yml
#
# Multi-GPU (single-host):
#     NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
#
# Multi-host (torchrun rendezvous):
#     NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=node-0 MASTER_PORT=29500 \
#         bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/launch.sh <train|eval|distill> <config.yml> [extra args...]"
    exit 1
fi

MODE=$1
CONFIG=$2
shift 2
EXTRA="$@"

# Pick up the script we want to run.
case "$MODE" in
    train)   ENTRY=src/train.py ;;
    eval)    ENTRY=src/eval.py ;;
    distill) ENTRY=src/distill_train.py ;;
    *) echo "Unknown mode: $MODE (expected 'train', 'eval', or 'distill')"; exit 1 ;;
esac

NGPU=${NGPU:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [[ "$NGPU" == "1" && "$NNODES" == "1" ]]; then
    echo "[launch] single-process: python $ENTRY --config $CONFIG $EXTRA"
    exec python "$ENTRY" --config "$CONFIG" $EXTRA
else
    echo "[launch] torchrun nproc_per_node=$NGPU nnodes=$NNODES node_rank=$NODE_RANK $ENTRY"
    exec torchrun \
        --nproc_per_node="$NGPU" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$ENTRY" --config "$CONFIG" $EXTRA
fi
