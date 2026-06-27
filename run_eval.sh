#!/usr/bin/env bash
# Usage: run_eval.sh <gpu> <config> <ckpt> <eval_data_dir> <seeds> <gbs> <num_samples> <output_dir> <logfile>
set -u
GPU=$1; CONFIG=$2; CKPT=$3; EVAL_DATA=$4; SEEDS=$5; GBS=$6; NS=$7; OUT=$8; LOG=$9
cd /sxl/sxl_code/ELF_pytorch
source /opt/conda/etc/profile.d/conda.sh && conda activate elf
export PYTHONPATH="$(pwd)/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="$GPU" python src/eval.py \
  --config "$CONFIG" \
  --checkpoint_path "$CKPT" \
  --seeds "$SEEDS" \
  --config_override eval_data_path="$EVAL_DATA" \
  --config_override use_bf16=true \
  --config_override use_wandb=false \
  --config_override num_samples="$NS" \
  --config_override global_batch_size="$GBS" \
  --config_override output_dir="$OUT" \
  > "$LOG" 2>&1
echo "EXIT=$? ($OUT)" >> "$LOG"
