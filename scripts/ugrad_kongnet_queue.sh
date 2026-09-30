#!/usr/bin/env bash
# ugrad runner for the phase-2 line-A KongNet strict-protocol evaluation (2026-09-30).
# Seq A (GPU0): val-fold threshold sweeps for ckpt1/2/3 (contour grid extended past its
# earlier 0.5 edge), then ckpt1 test-fold predict (fold 3). Seq B (GPU1): ckpt2/ckpt3 test
# predicts (folds 3 and 1), waiting for the sweeps to pick thresholds. Sweep = VAL folds
# only; test folds run once with the chosen constants.
set -u
cd /tmp/cgf2604/bdd26
source scripts/ugrad_env.sh
REPO=/tmp/cgf2604/repos/KongNet_Inference_Main
WD=/tmp/cgf2604/weights_a1/kongnet
PY=python
GRID="--seg-thrs 0.1 0.2 0.3 --contour-thrs 0.9 1.0"

best_thr () {  # -> "--seg-thr S --contour-thr C" from a sweep JSON
  $PY - "$1" <<'EOF'
import json, sys
b = json.load(open(sys.argv[1]))["best"]
print(f"--seg-thr {b['seg_thr']} --contour-thr {b['contour_thr']}")
EOF
}

sweep () {  # $1 = ckpt
  [ -f runs/kongnet/sweep_c$1.json ] && return 0
  $PY scripts/kongnet_eval.py --repo $REPO --weights-dir $WD --ckpt $1 --mode sweep \
      --n-images 800 $GRID --out runs/kongnet/sweep_c$1.json > runs/kongnet/sweep_c$1.log 2>&1
}

run_seq_a () {
  for c in 1 2 3; do sweep $c; done
  echo "$(date '+%F %T') seq A sweeps done; ckpt1 test predict (fold 3)" >> runs/kongnet/queue_a.log
  $PY scripts/kongnet_eval.py --repo $REPO --weights-dir $WD --ckpt 1 --mode predict \
      --fold 3 --batch-size 8 $(best_thr runs/kongnet/sweep_c1.json) \
      --out runs/kongnet/pred_fold3_kong_c1.npz > runs/kongnet/predict_c1_f3.log 2>&1
  echo "$(date '+%F %T') seq A done" >> runs/kongnet/queue_a.log
}

run_seq_b () {
  while [ ! -f runs/kongnet/sweep_c2.json ] || [ ! -f runs/kongnet/sweep_c3.json ]; do sleep 120; done
  T2=$(best_thr runs/kongnet/sweep_c2.json)
  T3=$(best_thr runs/kongnet/sweep_c3.json)
  echo "$(date '+%F %T') seq B starts; ckpt2: $T2 | ckpt3: $T3" >> runs/kongnet/queue_b.log
  $PY scripts/kongnet_eval.py --repo $REPO --weights-dir $WD --ckpt 2 --mode predict \
      --fold 3 --batch-size 8 $T2 --out runs/kongnet/pred_fold3_kong_c2.npz \
      > runs/kongnet/predict_c2_f3.log 2>&1
  $PY scripts/kongnet_eval.py --repo $REPO --weights-dir $WD --ckpt 3 --mode predict \
      --fold 1 --batch-size 8 $T3 --out runs/kongnet/pred_fold1_kong_c3.npz \
      > runs/kongnet/predict_c3_f1.log 2>&1
  echo "$(date '+%F %T') seq B done" >> runs/kongnet/queue_b.log
}

case ${1:-} in
  a) run_seq_a ;;
  b) run_seq_b ;;
  *) echo "usage: $0 a|b"; exit 2 ;;
esac
