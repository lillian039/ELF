#!/usr/bin/env bash
# Autonomous, resilient fetch of the ELF conditional-eval assets through the
# flaky root proxy (127.0.0.1:7899). Avoids the blocked /api endpoint:
#   - filenames + LFS pointer (oid/size) come from `git clone` (smart-HTTP,
#     LFS smudge skipped)
#   - big blobs come from /<repo>/resolve/main/<file> via resumable curl,
#     resuming into a stable <file>.dl until its size matches the pointer.
# Re-run safe; loops forever until every target is verified complete.
export http_proxy=http://127.0.0.1:7899 https_proxy=http://127.0.0.1:7899 no_proxy=127.0.0.1,localhost
export GIT_LFS_SKIP_SMUDGE=1
DEST=/sxl/sxl_code/ELF_pytorch/ckpts
mkdir -p "$DEST"
HFCO=https://huggingface.co

log(){ echo "[$(date +%H:%M:%S)] $*"; }

MODEL_REPOS=("embedded-language-flows/ELF-B-de-en-torch" "embedded-language-flows/ELF-B-xsum-torch")
DATA_REPOS=("embedded-language-flows/wmt14_de-en_validation_t5" "embedded-language-flows/xsum_validation_t5")

clone_pointers(){ # repo type dir
  local repo=$1 rtype=$2 dir=$3
  [ -d "$dir/.git" ] && { log "pointers present: $repo"; return 0; }
  local url="$HFCO/$repo"; [ "$rtype" = dataset ] && url="$HFCO/datasets/$repo"
  while true; do
    rm -rf "$dir"
    if git clone --depth 1 "$url" "$dir" >/tmp/clone.$$.log 2>&1; then
      log "cloned pointers: $repo"; return 0
    fi
    log "clone $repo failed: $(tail -1 /tmp/clone.$$.log | cut -c1-90); retry 15s"
    sleep 15
  done
}

is_pointer(){ head -c 50 "$1" 2>/dev/null | grep -q 'git-lfs.github.com'; }

fetch_one(){ # repo rtype dir relpath
  local repo=$1 rtype=$2 dir=$3 rel=$4
  local ptr="$dir/$rel" final="$dir/$rel" dl="$dir/$rel.dl"
  # Already finalized (real file, not a pointer)?
  if [ -f "$final" ] && ! is_pointer "$final"; then log "ok (cached) $rel"; return 0; fi
  local want oid
  want=$(grep -E '^size '       "$ptr" | awk '{print $2}')
  oid=$(grep  -E '^oid sha256:'  "$ptr" | sed 's/.*sha256://')
  local base="$HFCO/$repo/resolve/main"; [ "$rtype" = dataset ] && base="$HFCO/datasets/$repo/resolve/main"
  mkdir -p "$(dirname "$dl")"
  while true; do
    local have; have=$(stat -c%s "$dl" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$have" = "$want" ]; then break; fi
    log "GET $rel have=$have/${want:-?}"
    curl --proxy http://127.0.0.1:7899 --connect-timeout 8 --max-time 1800 \
         -C - -L -s -S --speed-limit 2000 --speed-time 30 \
         -o "$dl" "$base/$rel" 2>/tmp/curl.$$.err || true
    have=$(stat -c%s "$dl" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$have" = "$want" ]; then break; fi
    log "  partial $rel have=$have/${want:-?} ($(tail -1 /tmp/curl.$$.err 2>/dev/null|cut -c1-50)); retry 8s"
    sleep 8
  done
  if [ -n "$oid" ]; then
    local got; got=$(sha256sum "$dl" | awk '{print $1}')
    if [ "$got" != "$oid" ]; then log "CHECKSUM MISMATCH $rel ($got != $oid); redownloading"; rm -f "$dl"; fetch_one "$@"; return; fi
    log "sha256 ok $rel"
  fi
  mv -f "$dl" "$final"
  log "DONE $rel ($(stat -c%s "$final"))"
}

process_repo(){ # repo rtype
  local repo=$1 rtype=$2
  local dir="$DEST/$(basename "$repo")"
  clone_pointers "$repo" "$rtype" "$dir"
  local f
  while read -r f; do
    [ -z "$f" ] && continue
    if is_pointer "$dir/$f"; then fetch_one "$repo" "$rtype" "$dir" "$f"; fi
  done < <(cd "$dir" && git ls-files)
  log "REPO COMPLETE: $repo -> $dir"
}

log "=== resilient_download start (proxy 127.0.0.1:7899) ==="
for r in "${MODEL_REPOS[@]}"; do process_repo "$r" model; done
for r in "${DATA_REPOS[@]}"; do process_repo "$r" dataset; done
log "=== ALL ASSETS DOWNLOADED ==="
touch "$DEST/.ALL_DONE"
