#!/usr/bin/env python
"""Train official HoVer-Net (fast mode) on one official PanNuke split, then predict + evaluate the test fold.

    python scripts/train_hovernet.py --split 1 --out runs/hovernet/split1
Resumes automatically from <out>/last.pth. Uses the LAST checkpoint for testing (no test-set selection).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.hovernet.engine import Phase, TrainConfig, build_model, predict_fold, train
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--split", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--epochs", type=int, nargs=2, default=[50, 50], help="epochs of phase 1 (frozen) / phase 2")
p.add_argument("--batch", type=int, nargs=2, default=[16, 4])
p.add_argument("--workers", type=int, default=8)
p.add_argument("--amp", action="store_true")
p.add_argument("--val-every", type=int, default=5)
p.add_argument("--pretrained", default="weights/hovernet/ImageNet-ResNet50-Preact_pytorch.tar")
p.add_argument("--skip-train", action="store_true")
p.add_argument("--tta", action="store_true", help="also evaluate with 8x dihedral TTA")
a = p.parse_args()

tr, va, te = split_folds(a.split)
cfg = TrainConfig(split=a.split, out_dir=str(a.out), pretrained=a.pretrained, workers=a.workers, amp=a.amp,
                  val_every=a.val_every,
                  phases=[Phase(a.epochs[0], a.batch[0], True, step=max(a.epochs[0] // 2, 1)),
                          Phase(a.epochs[1], a.batch[1], False, step=max(a.epochs[1] // 2, 1))])
if not a.skip_train:
    train(cfg, [tr], [va])

model = build_model(freeze=False).cuda()
model.load_state_dict(torch.load(a.out / "final.pth", map_location="cpu", weights_only=False)["model"])
f = PanNukeFold(te)
for tta in [False] + ([True] if a.tta else []):
    tag = f"test_fold{te}" + ("_tta" if tta else "")
    inst, typ = predict_fold(model, f, tta=tta)
    np.savez_compressed(a.out / f"pred_{tag}.npz", inst=inst.astype(np.int32), type=typ)
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
    save_report(res, a.out / f"eval_{tag}")
    print(tag, "\n" + format_summary(res["summary"]))
