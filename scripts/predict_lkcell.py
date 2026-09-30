#!/usr/bin/env python
"""Predict LKCell released per-fold checkpoints on PanNuke test folds under the strict
protocol (line A, phase 2). The network is the vendored LKCell (UniRepLKNet-S encoder per
the released config/checkpoint); everything downstream — data order, decode (official
HoVer-Net constants), eval — is the shared nucseg pipeline, so only the network differs
from the CellViT-UNI baseline. Normalization is LKCell's (mean/std 0.5), NOT UNI's.

Fold mapping (survey_2026_09.md §4, read off each run's config.yaml train/val/test):
  split 1 = 2024-04-22T232903...  (train0/val1/test2 -> test fold 3)
  split 2 = 2024-07-31T221951...  (train1/val0/test2 -> test fold 3)
  split 3 = 2024-04-24T013604...  (train2/val1/test0 -> test fold 1)
Protocol note: released model_best.pth is val-selected best (like HoVer-NeXt), not last.

    python scripts/predict_lkcell.py --ckpt <weights_a1>/lkcell_l/<dir>/checkpoints/model_best.pth \
        --fold 3 --out runs/lkcell/split1 [--tta] [--no-eval]
-> <out>/pred_fold{k}[_tta].npz (inst int32, type uint8) + eval_fold{k}[_tta] unless --no-eval
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import predict_fold
from nucseg.data.pannuke import PanNukeFold
from nucseg.lkcell import build_lkcell
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

LKCELL_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--tta", action="store_true")
p.add_argument("--no-eval", action="store_true",
               help="skip the in-process eval (16 GB-VA hosts: run scripts/eval_pannuke.py separately)")
p.add_argument("--batch-size", type=int, default=8, help="lower on the 3.6 GiB-capped L4s")
a = p.parse_args()

a.out.mkdir(parents=True, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model = build_lkcell(ck["model_state_dict"]).cuda()  # strict; frees nothing else from the 1.5 GB file
del ck
print(f"loaded {a.ckpt.name} (epoch tag ignored; val-selected best)")

f = PanNukeFold(a.fold)
inst, typ, tissue = predict_fold(model, f, tta=a.tta, inst_probs=False,
                                 batch_size=a.batch_size, norm=LKCELL_NORM)
tag = f"fold{a.fold}" + ("_tta" if a.tta else "")
np.savez_compressed(a.out / f"pred_{tag}.npz", inst=inst, type=typ, tissue_prob=tissue)
if not a.no_eval:
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
    save_report(res, a.out / f"eval_{tag}")
    print(tag, "\n" + format_summary(res["summary"]))
