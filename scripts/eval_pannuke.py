#!/usr/bin/env python
"""Evaluate a prediction file on a PanNuke test fold.

    python scripts/eval_pannuke.py --pred runs/x/pred_fold3.npz --fold 3 --out runs/x/eval_fold3
    python scripts/eval_pannuke.py --gt-as-pred --fold 3 --out /tmp/sanity   # pipeline sanity check

The .npz must hold `inst` (N,256,256) and `type` (N,256,256) in fold order.
"""
import argparse
from pathlib import Path

import numpy as np

from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--pred", type=Path)
p.add_argument("--gt-as-pred", action="store_true")
p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--workers", type=int, default=16)
a = p.parse_args()

f = PanNukeFold(a.fold)
if a.gt_as_pred:
    inst, typ = f.inst, f.type
else:
    d = np.load(a.pred)
    inst, typ = d["inst"], d["type"]
res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ, workers=a.workers)
save_report(res, a.out)
print(format_summary(res["summary"]))
print(f"-> {a.out}")
