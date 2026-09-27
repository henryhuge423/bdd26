#!/usr/bin/env python
"""Predict (+ evaluate) official HoVer-NeXt-T per-fold weights on a PanNuke fold.

Sanity target: pannuke_convnextv2_tiny_{1,2} (test fold 3) should land near the paper's
HoVer-NeXt-T mPQ 0.477; _3 (test fold 1) on fold 1 likewise. A much higher number means the
fold mapping is wrong (evaluating on training data).

    python scripts/predict_hovernext.py --weights runs/hovernext_t/split1 --fold 3 [--bs 24]
-> <weights>/eval_test_fold{N}/ with summary.json + pred.npz (name matches analyze_cf's glob)
"""
import argparse
from pathlib import Path

import numpy as np

from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernext.engine import build_model, predict_fold
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--bs", type=int, default=24)
a = p.parse_args()

m = build_model(a.weights)
f = PanNukeFold(a.fold)
inst, typ = predict_fold(m, f, batch_size=a.bs)
out = a.weights / f"eval_test_fold{a.fold}"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "pred.npz", inst=inst.astype(np.int32), type=typ)
res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ, workers=16)
save_report(res, out)
print(f"fold{a.fold} ({a.weights.name})\n" + format_summary(res["summary"]))
print(f"-> {out}")
