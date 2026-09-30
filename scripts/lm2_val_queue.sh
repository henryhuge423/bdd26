#!/usr/bin/env bash
# LM2-only runner: wait for the in-flight x2 seed chains (train_cellvit) to finish, then run the
# pre-registered x2 decode-lever VAL-fold predictions (findings 2026-09-30 next decode-menu levers:
# tune merge / sliver-drop on VAL folds; marker-u is the other pre-registered decode suspect).
#   nohup bash scripts/lm2_val_queue.sh 7 gpu7.log "cmd1" "cmd2" ... &
# GPU id is $1; the remaining args are predict commands run sequentially with CUDA_VISIBLE_DEVICES set.
set -u
cd /data6/jinxinhao/bdd26
GPU=$1; LOG=$2; shift 2
echo "$(date '+%F %T') waiting for train_cellvit chains to exit" >> "runs/$LOG"
while pgrep -f "train_cellvit.py" > /dev/null; do sleep 600; done
echo "$(date '+%F %T') chains done; starting queue on GPU $GPU" >> "runs/$LOG"
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data6/jinxinhao/.conda/envs/nuclei/bin/python
for cmd in "$@"; do
  echo "$(date '+%F %T') RUN $cmd" >> "runs/$LOG"
  eval "$cmd" >> "runs/$LOG" 2>&1
  echo "$(date '+%F %T') DONE (exit $?)" >> "runs/$LOG"
done
echo "$(date '+%F %T') queue complete" >> "runs/$LOG"
