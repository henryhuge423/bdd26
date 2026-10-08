#!/usr/bin/env python
"""Train CellViT-UNI, optionally followed by test inference from the final checkpoint.

    python scripts/train_cellvit.py --split 1 --out runs/cellvit_uni/split1 --tta
    python scripts/train_cellvit.py --split 1 --out runs/hv_control --hv-min-size 8 --train-only
Resumes automatically from <out>/last.pth; scientific configuration changes are refused.
Use --train-only for development experiments that must not access the test fold.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.data import tissue_ids
from nucseg.cellvit.engine import (
    TrainConfig, build_model, predict_fold, res_tag, run_hv_min_size, run_upscale, train,
)
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.hovernet.targets import validate_hv_min_size
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report


def _hv_cutoff(text):
    try:
        return validate_hv_min_size(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HV cutoff must be a positive integer in working pixels") from exc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
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
    p.add_argument("--upscale", type=int, default=None,
                   help="working-resolution multiplier; default: run config.json, else native256")
    p.add_argument("--hv-min-size", type=_hv_cutoff, default=None,
                   help="minimum instance area for nonzero HV targets in WORKING pixels (not auto-scaled); "
                        "default: run config.json, else30. NP/TP labels are unchanged.")
    p.add_argument("--dead-expert", action="store_true",
                   help="add the Dead-detection expert branch and its positive-image masked loss "
                        "(DSB spec 2026-10-07); resuming such a run requires the flag again")
    p.add_argument("--dead-neg-w", type=float, default=0.0,
                   help="BCE weight on Dead-free images' dead-fg channel (frozen menu {0.0, 0.1})")
    p.add_argument("--widen", type=int, default=0,
                   help="C1 equal-parameter widening of the NP/HV decoder bottlenecks; resuming "
                        "a widened run requires the same value again")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--skip-train", action="store_true")
    mode.add_argument("--train-only", action="store_true", help="train on train/val folds and exit WITHOUT test inference")
    p.add_argument("--tta", action="store_true", help="also evaluate with 8x dihedral TTA")
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    tr, va, te = split_folds(a.split)
    upscale = run_upscale(a.out, a.upscale)
    hv_min_size = run_hv_min_size(a.out, a.hv_min_size)
    cfg = TrainConfig(split=a.split, out_dir=str(a.out), epochs=a.epochs, unfreeze_epoch=a.unfreeze_epoch,
                      batch_size=a.batch, lr=a.lr, workers=a.workers, val_every=a.val_every,
                      seed=a.seed, sampling_gamma=a.sampling_gamma, np_wce=a.np_wce,
                      dead_w=a.dead_w, small_w=a.small_w, small_area=a.small_area,
                      cp_prob=a.cp_prob, cp_lam=a.cp_lam, cp_dead_w=a.cp_dead_w,
                      cp_area=tuple(a.cp_area), cp_clearance=a.cp_clearance,
                      synth=a.synth, synth_frac=a.synth_frac, upscale=upscale, hv_min_size=hv_min_size,
                      dead_expert=a.dead_expert, dead_neg_w=a.dead_neg_w, widen=a.widen)
    if not a.skip_train:
        train(cfg, [tr], [va])
    if a.train_only:
        return

    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(torch.load(a.out / "final.pth", map_location="cpu", weights_only=False)["model"])
    f = PanNukeFold(te)
    for tta in [False] + ([True] if a.tta else []):
        tag = f"test_fold{te}{res_tag(upscale)}" + ("_tta" if tta else "")
        inst, typ, tissue = predict_fold(model, f, tta=tta, upscale=upscale)
        np.savez_compressed(a.out / f"pred_{tag}.npz", inst=inst.astype(np.int32), type=typ, tissue_prob=tissue)
        res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
        res["summary"]["tissue_acc"] = float((tissue.argmax(-1) == tissue_ids(f.tissue)).mean())
        save_report(res, a.out / f"eval_{tag}")
        print(tag, "\n" + format_summary(res["summary"]))


if __name__ == "__main__":
    main()
