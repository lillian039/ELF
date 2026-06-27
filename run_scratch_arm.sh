#!/usr/bin/env bash
# Usage: run_scratch_arm.sh <gpus> <config> <port> <logfile>
GPUS=$1; CONFIG=$2; PORT=$3; LOG=$4
cd /sxl/sxl_code/ELF_pytorch
source /opt/conda/etc/profile.d/conda.sh && conda activate elf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" NGPU=$NGPU \
  bash scripts/launch.sh train "$CONFIG" > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
