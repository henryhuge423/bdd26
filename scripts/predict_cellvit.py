#!/usr/bin/env python
"""Predict (+ evaluate) a trained CellViT-UNI checkpoint on any PanNuke fold, saving per-instance type
probabilities for post-hoc re-typing (pillar A).

    python scripts/predict_cellvit.py --run runs/cellvit_uni/split1 --fold 2 [--tta] [--no-eval]
-> <run>/pred_fold2[_tta].npz with inst, type, tissue_prob, inst_img, inst_id, inst_prob
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import build_model, predict_fold
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--ckpt", default="final.pth")
p.add_argument("--tta", action="store_true")
p.add_argument("--no-eval", action="store_true")
a = p.parse_args()

model = build_model(pretrained=False).cuda()
model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu", weights_only=False)["model"])
f = PanNukeFold(a.fold)
inst, typ, tissue, (ii, iid, ip) = predict_fold(model, f, tta=a.tta, inst_probs=True)
tag = f"fold{a.fold}" + ("_tta" if a.tta else "")
np.savez_compressed(a.run / f"pred_{tag}.npz", inst=inst.astype(np.int32), type=typ, tissue_prob=tissue,
                    inst_img=ii, inst_id=iid, inst_prob=ip)
if not a.no_eval:
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
    save_report(res, a.run / f"eval_{tag}")
    print(tag, "\n" + format_summary(res["summary"]))
