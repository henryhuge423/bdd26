#!/usr/bin/env python
"""Train CellViT-UNI on one official PanNuke split, then predict + evaluate the test fold (last checkpoint).

    python scripts/train_cellvit.py --split 1 --out runs/cellvit_uni/split1 --tta
Resumes automatically from <out>/last.pth.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.data import tissue_ids
from nucseg.cellvit.engine import TrainConfig, build_model, predict_fold, train
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--split", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--epochs", type=int, default=130)
p.add_argument("--unfreeze-epoch", type=int, default=25)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--val-every", type=int, default=5)
p.add_argument("--seed", type=int, default=19, help="global seed: decoder init, sampler, augmentation workers")
p.add_argument("--sampling-gamma", type=float, default=0.85, help="cell+tissue sampler balance (1 = full)")
p.add_argument("--np-wce", type=float, default=0.0, help="weight of the pixel-weighted NP cross-entropy")
p.add_argument("--dead-w", type=float, default=0.0, help="extra NP-CE weight on Dead-nucleus pixels")
p.add_argument("--small-w", type=float, default=0.0, help="extra NP-CE weight on small-nucleus pixels")
p.add_argument("--small-area", type=int, default=100)
p.add_argument("--cp-prob", type=float, default=0.0, help="copy-paste: fraction of patches augmented")
p.add_argument("--cp-lam", type=float, default=3.0, help="copy-paste: mean insertions per augmented patch")
p.add_argument("--cp-dead-w", type=float, default=0.4, help="copy-paste: donor weight for Dead class")
p.add_argument("--cp-area", type=int, nargs=2, default=[50, 400], help="copy-paste: donor area range px")
p.add_argument("--cp-clearance", type=int, default=8, help="copy-paste: min px distance to other nuclei")
p.add_argument("--synth", type=str, default=None, help="synthetic dir (images/inst/type/base .npy)")
p.add_argument("--synth-frac", type=float, default=0.0, help="synthetic samples as a fraction of |real|")
p.add_argument("--upscale", type=int, default=1,
               help="working-resolution multiplier (2 = 512px; phase-2 B1 resolution ablation)")
p.add_argument("--skip-train", action="store_true")
p.add_argument("--tta", action="store_true", help="also evaluate with 8x dihedral TTA")
a = p.parse_args()

tr, va, te = split_folds(a.split)
cfg = TrainConfig(split=a.split, out_dir=str(a.out), epochs=a.epochs, unfreeze_epoch=a.unfreeze_epoch,
                  batch_size=a.batch, lr=a.lr, workers=a.workers, val_every=a.val_every,
                  seed=a.seed, sampling_gamma=a.sampling_gamma, np_wce=a.np_wce, dead_w=a.dead_w, small_w=a.small_w,
                  small_area=a.small_area, cp_prob=a.cp_prob, cp_lam=a.cp_lam, cp_dead_w=a.cp_dead_w,
                  cp_area=tuple(a.cp_area), cp_clearance=a.cp_clearance,
                  synth=a.synth, synth_frac=a.synth_frac, upscale=a.upscale)
if not a.skip_train:
    train(cfg, [tr], [va])

model = build_model(cfg, pretrained=False).cuda()
model.load_state_dict(torch.load(a.out / "final.pth", map_location="cpu", weights_only=False)["model"])
f = PanNukeFold(te)
for tta in [False] + ([True] if a.tta else []):
    tag = f"test_fold{te}" + ("_tta" if tta else "")
    inst, typ, tissue = predict_fold(model, f, tta=tta, upscale=a.upscale)
    np.savez_compressed(a.out / f"pred_{tag}.npz", inst=inst.astype(np.int32), type=typ, tissue_prob=tissue)
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
    res["summary"]["tissue_acc"] = float((tissue.argmax(-1) == tissue_ids(f.tissue)).mean())
    save_report(res, a.out / f"eval_{tag}")
    print(tag, "\n" + format_summary(res["summary"]))
