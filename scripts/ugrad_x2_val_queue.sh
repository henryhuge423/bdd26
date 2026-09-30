#!/usr/bin/env bash
# ugrad L4 queue: x2 VAL-fold predicts feeding the decode-menu extension (2026-09-30).
# VAL folds only (split1->2, split2->1, split3->2); predict_cellvit.py auto-evals each fold,
# so the val mPQ/bPQ baselines for the lever sweeps come out on the same machine.
# Memory-light for the ugrad L4s — TWO hard limits, both hit today: (1) a PER-PROCESS GPU
# cap of ~3.6 GiB (third attempt: CUDA OOM at 3.61 GiB in use while 18.4 GiB was free;
# batch 2 sits at a stable 3.33 GiB — do not raise UGRAD_BS above 2); (2) the 16 GB host
# address space (fixed in engine.py by preallocating the output maps, 9af4c99).
# Batch size is result-invariant: patches decode independently (A/B-verified on LM2,
# runs/_ab_*). --no-inst-probs skips the retype table (lever sweeps consume inst/type only).
#  a (ugradx GPU0): plain x2, split1 val fold 2, then split2 val fold 1
#  c (ugradv GPU0): du2 + marker-u2, split1 val fold 2
#  d (ugradv GPU1): plain x2, split3 val fold 2
set -u
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=16777216 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /tmp/cgf2604/bdd26 || exit 1
source scripts/ugrad_env.sh
PY=python
R=runs/cellvit_uni_x2

pred () {  # $1 gpu  $2 split  $3 fold  $4 extra-flags  $5 logfile
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/predict_cellvit.py --run $R/split$2 --fold $3 \
    --batch-size ${UGRAD_BS:-2} --no-inst-probs $4 > $R/split$2/$5 2>&1
}

case ${1:-} in
  a) pred 0 1 2 ""                          predict_val2.log
     pred 0 2 1 ""                          predict_val1.log
     echo "$(date '+%F %T') seq a done" >> runs/_ugrad_x2_val.log ;;
  c) pred 0 1 2 "--decode-u 2 --marker-u 2" predict_val2_du2mk2.log
     echo "$(date '+%F %T') seq c done" >> runs/_ugrad_x2_val.log ;;
  d) pred 1 3 2 ""                          predict_val2.log
     echo "$(date '+%F %T') seq d done" >> runs/_ugrad_x2_val.log ;;
  *) echo "usage: $0 a|c|d"; exit 2 ;;
esac
