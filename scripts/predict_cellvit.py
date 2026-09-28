#!/usr/bin/env python
"""Predict (+ evaluate) a trained CellViT-UNI checkpoint on any PanNuke fold, saving per-instance type
probabilities for post-hoc re-typing (pillar A).

    python scripts/predict_cellvit.py --run runs/cellvit_uni/split1 --fold 2 [--tta] [--no-eval]
-> <run>/pred_fold2[_x{k}][_tta].npz with inst, type, tissue_prob, inst_img, inst_id, inst_prob
   (the _x{k} resolution tag appears only for non-native working resolutions, so artifacts of
   different resolutions never overwrite each other; retype_conch.py resolves the same tag)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import build_model, predict_fold, res_tag, run_upscale
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--ckpt", default="final.pth")
p.add_argument("--tta", action="store_true")
p.add_argument("--no-eval", action="store_true")
p.add_argument("--upscale", type=int, default=None,
               help="override the run's working resolution (default: config.json upscale)")
a = p.parse_args()

upscale = run_upscale(a.run, a.upscale)
model = build_model(pretrained=False).cuda()
model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu", weights_only=False)["model"])
f = PanNukeFold(a.fold)
inst, typ, tissue, (ii, iid, ip) = predict_fold(model, f, tta=a.tta, inst_probs=True, upscale=upscale)
tag = f"fold{a.fold}{res_tag(upscale)}" + ("_tta" if a.tta else "")
np.savez_compressed(a.run / f"pred_{tag}.npz", inst=inst.astype(np.int32), type=typ, tissue_prob=tissue,
                    inst_img=ii, inst_id=iid, inst_prob=ip)
if not a.no_eval:
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
    save_report(res, a.run / f"eval_{tag}")
    print(tag, "\n" + format_summary(res["summary"]))
