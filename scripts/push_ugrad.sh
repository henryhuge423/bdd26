#!/usr/bin/env bash
# From LM1: push code (+ data/pannuke, + optional weights) to each ugrad machine's local /tmp,
# then bootstrap the venv there.   Usage: scripts/push_ugrad.sh [--no-data] [--weights] [host ...]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/hosts.sh
DATA=1; WEIGHTS=0; HOSTS=()
for a in "$@"; do case $a in --no-data) DATA=0;; --weights) WEIGHTS=1;; *) HOSTS+=("$a");; esac; done
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("${UGRAD_HOSTS[@]}")
RSH="ssh ${SSH_OPTS[*]}"
for h in "${HOSTS[@]}"; do
  echo "== $h"
  ssh "${SSH_OPTS[@]}" "$UGRAD_USER@$h" "mkdir -p $UGRAD_ROOT/bdd26 && chmod 700 $UGRAD_ROOT"
  rsync -az --delete -e "$RSH" --exclude-from=scripts/rsync_excludes.txt --exclude=/hg-token.txt ./ "$UGRAD_USER@$h:$UGRAD_ROOT/bdd26/"
  rsync -az -e "$RSH" --chmod=F600 hg-token.txt "$UGRAD_USER@$h:$UGRAD_ROOT/hg-token.txt"
  [ $DATA -eq 1 ] && rsync -az -e "$RSH" --exclude='*.tmp' data/pannuke "$UGRAD_USER@$h:$UGRAD_ROOT/bdd26/data/"
  [ $WEIGHTS -eq 1 ] && rsync -az -e "$RSH" weights "$UGRAD_USER@$h:$UGRAD_ROOT/bdd26/"
  ssh "${SSH_OPTS[@]}" "$UGRAD_USER@$h" "bash $UGRAD_ROOT/bdd26/scripts/ugrad_bootstrap.sh"
done
