#!/usr/bin/env bash
# From LM1: pull runs/ (and optionally weights/) back from ugrad machines. Usage: scripts/pull_ugrad.sh [--weights] [host ...]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/hosts.sh
WEIGHTS=0; HOSTS=()
for a in "$@"; do case $a in --weights) WEIGHTS=1;; *) HOSTS+=("$a");; esac; done
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("${UGRAD_HOSTS[@]}")
RSH="ssh ${SSH_OPTS[*]}"
for h in "${HOSTS[@]}"; do
  rsync -az -e "$RSH" "$UGRAD_USER@$h:$UGRAD_ROOT/bdd26/runs/" runs/ 2>/dev/null || true
  [ $WEIGHTS -eq 1 ] && rsync -az -e "$RSH" --exclude='.cache/' "$UGRAD_USER@$h:$UGRAD_ROOT/bdd26/weights/" weights/
done
