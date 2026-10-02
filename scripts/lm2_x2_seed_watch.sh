#!/usr/bin/env bash
# LM2 watcher for the x2 3x3 seed grid (2026-09-30). Test protocol per chain:
# predict_cellvit.py --fold <TEST> --decode-u 2 (the pre-registered du2 decode, non-TTA
# headline) -> eval_fold{k}_x2_du2 next to the chain's checkpoints.
#  - split1_seed1 done -> launch split2_seed2 training on its freed GPU7
#  - split1_seed2 done -> launch split3_seed2 training on its freed GPU1
#  - any chain with final.pth and no test eval yet -> sequential predict+eval on GPU3
#    (shares with the split2_seed1 training; a failed predict retries on the next poll)
# Exits once all six chains have eval dirs.
set -u
cd /data6/jinxinhao/bdd26 || exit 1
PY=/data6/jinxinhao/.conda/envs/nuclei/bin/python
PGPU=3
LOG=runs/_x2_seed_watch.log

declare -A TF=( [split1_seed1]=3 [split1_seed2]=3 [split2_seed1]=3 [split2_seed2]=3 [split3_seed1]=1 [split3_seed2]=1 )

launch_train () {  # $1 split  $2 outdir  $3 seed  $4 gpu
  local d=runs/cellvit_uni_x2/$2
  [ -d "$d" ] && return 0
  ( setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$4 $PY scripts/train_cellvit.py --split $1 --out $d --seed $3 --upscale 2 --tta" > runs/_x2_grid_gpu$4.log 2>&1 < /dev/null & )
  echo "$(date '+%F %T') launched training $2 on gpu$4" >> $LOG
}

while true; do
  [ -f runs/cellvit_uni_x2/split1_seed1/final.pth ] && launch_train 2 split2_seed2 2 7
  [ -f runs/cellvit_uni_x2/split1_seed2/final.pth ] && launch_train 3 split3_seed2 2 1
  for d in "${!TF[@]}"; do
    dir=runs/cellvit_uni_x2/$d
    ev=$dir/eval_fold${TF[$d]}_x2_du2
    if [ -f "$dir/final.pth" ] && [ ! -d "$ev" ] && [ ! -f "$dir/.pred_running" ]; then
      touch "$dir/.pred_running"
      echo "$(date '+%F %T') predict+eval $d fold ${TF[$d]}" >> $LOG
      if CUDA_VISIBLE_DEVICES=$PGPU $PY scripts/predict_cellvit.py --run "$dir" --fold "${TF[$d]}" \
           --decode-u 2 --batch-size 32 > "$dir/predict_du2.log" 2>&1; then
        rm -f "$dir/.pred_running"
      else
        echo "$(date '+%F %T') predict $d FAILED (retry next poll)" >> $LOG
        rm -f "$dir/.pred_running"
      fi
    fi
  done
  ok=1
  for d in "${!TF[@]}"; do
    [ -d "runs/cellvit_uni_x2/$d/eval_fold${TF[$d]}_x2_du2" ] || ok=0
  done
  if [ "$ok" = 1 ]; then
    echo "$(date '+%F %T') all six seed evals done" >> $LOG
    exit 0
  fi
  sleep 300
done
