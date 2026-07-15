#!/usr/bin/env bash
# Pull k x rho training outputs + logs from cheery back to binky, so everything
# (checkpoints, configs, train logs) lives on binky for later evals.
# Safe to run repeatedly (incremental rsync); run it while training too.
#
# Usage (on binky): bash tinystories_demo/sync_from_cheery.sh
set -uo pipefail
cd "$(dirname "$0")/.."

DEST=/mnt/faster3/lc2762/rho_runs_from_cheery
mkdir -p "${DEST}"

# training outputs (checkpoints every 10 epochs + final, config.yml)
for TAG in p030 p000 m030 p015; do
  SRC="cheery:/mnt/faster1/lc2762/elf_rho_${TAG}_k64"
  rsync -a --info=stats0,flist0 "${SRC}" "${DEST}/" 2>/dev/null \
    && echo "synced elf_rho_${TAG}_k64" || echo "(rho ${TAG}: nothing yet)"
done

# queue + per-run logs
rsync -a cheery:/mnt/faster1/lc2762/ELF/tinystories_demo/logs/ \
  tinystories_demo/logs/cheery/ 2>/dev/null && echo "synced cheery logs -> tinystories_demo/logs/cheery/"

echo "sync done: $(du -sh ${DEST} 2>/dev/null | cut -f1) in ${DEST}"
