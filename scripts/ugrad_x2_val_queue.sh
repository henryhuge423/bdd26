#!/usr/bin/env bash
# ugrad L4 queue: x2 VAL-fold predicts feeding the decode-menu extension (2026-09-30).
# VAL folds only (split1->2, split2->1, split3->2); predict_cellvit.py auto-evals each
# fold, so the val mPQ/bPQ baselines for the lever sweeps come out on the same machine.
#  seq a (GPU0): plain x2 decode, split1 + split2 val folds
#  seq b (GPU1): du2 + marker-u2 decode, split1 val fold, then plain x2 split3 val fold
# batch-size 4: 512^2 activations fit the 6.9 GB L4 budget (32 is the A100 default).
set -u
cd /tmp/cgf2604/bdd26 || exit 1
source scripts/ugrad_env.sh
PY=python
R=runs/cellvit_uni_x2

seq_a () {
  mkdir -p $R/split1 $R/split2
  CUDA_VISIBLE_DEVICES=0 $PY scripts/predict_cellvit.py --run $R/split1 --fold 2 --batch-size 4 \
    > $R/split1/predict_val2.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY scripts/predict_cellvit.py --run $R/split2 --fold 1 --batch-size 4 \
    > $R/split2/predict_val1.log 2>&1
  echo "$(date '+%F %T') seq a done" >> runs/_ugrad_x2_val.log
}

seq_b () {
  mkdir -p $R/split1 $R/split3
  CUDA_VISIBLE_DEVICES=1 $PY scripts/predict_cellvit.py --run $R/split1 --fold 2 \
    --decode-u 2 --marker-u 2 --batch-size 4 > $R/split1/predict_val2_du2mk2.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY scripts/predict_cellvit.py --run $R/split3 --fold 2 --batch-size 4 \
    > $R/split3/predict_val2.log 2>&1
  echo "$(date '+%F %T') seq b done" >> runs/_ugrad_x2_val.log
}

case ${1:-} in
  a) seq_a ;;
  b) seq_b ;;
  *) echo "usage: $0 a|b"; exit 2 ;;
esac
