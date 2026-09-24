#!/usr/bin/env bash
# From LM1: push code (+ data/pannuke, + weights) to LM2 and make sure the `nuclei` env exists there.
#   Usage: scripts/push_lm2.sh [--no-data] [--weights]
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/hosts.sh
DATA=1; WEIGHTS=0
for a in "$@"; do case $a in --no-data) DATA=0;; --weights) WEIGHTS=1;; esac; done
RSH="ssh ${SSH_OPTS[*]} -p $LM2_PORT"
ssh "${SSH_OPTS[@]}" -p $LM2_PORT $LM2_USER_HOST "mkdir -p $LM2_ROOT"
rsync -az --delete -e "$RSH" --exclude-from=scripts/rsync_excludes.txt ./ "$LM2_USER_HOST:$LM2_ROOT/"
[ $DATA -eq 1 ] && rsync -az -e "$RSH" data/pannuke "$LM2_USER_HOST:$LM2_ROOT/data/"
[ $WEIGHTS -eq 1 ] && rsync -az -e "$RSH" --exclude='.cache/' weights "$LM2_USER_HOST:$LM2_ROOT/"
ssh "${SSH_OPTS[@]}" -p $LM2_PORT $LM2_USER_HOST "bash -lc 'cd $LM2_ROOT && bash scripts/lm_env_setup.sh'"
