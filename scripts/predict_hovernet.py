#!/usr/bin/env python
"""Predict + evaluate any HoVer-Net fast-mode checkpoint on a PanNuke fold.

    python scripts/predict_hovernet.py --ckpt weights/hovernet/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --fold 3 --out runs/sanity/official_ckpt_fold3      # NOTE: official ckpt saw all folds (leaky)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernet.engine import build_model, predict_fold
from nucseg.hovernet.official import convert_pytorch_checkpoint
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--tta", action="store_true")
a = p.parse_args()

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = sd["model"] if "model" in sd else convert_pytorch_checkpoint(sd["desc"])
model = build_model(freeze=False).cuda()
print(model.load_state_dict(sd, strict=True))
f = PanNukeFold(a.fold)
inst, typ = predict_fold(model, f, tta=a.tta)
a.out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out / "pred.npz", inst=inst.astype(np.int32), type=typ)
res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
save_report(res, a.out)
print(format_summary(res["summary"]))
