#!/usr/bin/env bash
# watch_and_stop.sh <arm>
# Stop a from-scratch training arm cleanly once epoch 5 is fully done.
# The training loop saves the epoch-5 checkpoint and runs the epoch-5
# generation eval BEFORE it prints "Epoch 6/...", so waiting for that line
# guarantees the epoch-5 artifacts are flushed before we kill the process.
arm=$1
log=/tmp/scratch_${arm}.log
pat="train_de-en_ELF-B_scratch_${arm}.yml"
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "watching arm '$arm' (log=$log) for epoch-5 completion" > /tmp/watch_${arm}.log
while true; do
  if grep -q 'Epoch 6/' "$log" 2>/dev/null; then
    pkill -9 -f "$pat"
    stamp "epoch 5 done -> stopped arm '$arm'" >> /tmp/watch_${arm}.log
    break
  fi
  # Bail out if the training process already died on its own.
  if ! pgrep -f "$pat" >/dev/null 2>&1; then
    stamp "arm '$arm' process gone before epoch 6 (crash or finished)" >> /tmp/watch_${arm}.log
    break
  fi
  sleep 60
done
