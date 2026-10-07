#!/usr/bin/env python
"""Predict (+ evaluate) a trained CellViT-UNI checkpoint on any PanNuke fold, saving per-instance type
probabilities for post-hoc re-typing (pillar A).

    python scripts/predict_cellvit.py --run runs/cellvit_uni/split1 --fold 2 [--tta] [--no-eval] [--dead]
-> <run>/pred_fold2[_x{k}][_du{u}][_tta].npz with inst, type, tissue_prob, inst_img, inst_id, inst_prob
   (the _x{k} resolution tag appears only for non-native working resolutions, and _du{u} only for
   u-scaled decode constants, so artifacts of different settings never overwrite each other;
   retype_conch.py resolves the same tag)
-> with --dead (dead-expert checkpoints) also <run>/pred_fold2..._dead.npz with the expert's decoded
   candidate instance map (dead_inst); the base artifact is never touched.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import (TrainConfig, build_model, predict_fold, res_tag, run_dead_expert,
                                   run_upscale, run_widen)
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--ckpt", default="final.pth")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--dead", action="store_true",
                   help="also decode the checkpoint's dead-expert channels into candidate instances "
                        "(pred_<tag>_dead.npz); requires a --dead-expert-trained run")
    p.add_argument("--upscale", type=int, default=None,
                   help="override the run's working resolution (default: config.json upscale)")
    p.add_argument("--decode-u", type=float, default=1.0,
                   help="scale the px-unit decode constants (min_size 10*u^2, Sobel ksize odd(21*u)); "
                        "1 = official decode (findings 2026-09-29 pre-registered x2 re-decode)")
    p.add_argument("--marker-u", type=float, default=1.0,
                   help="scale the 5x5 marker-open kernel (odd(5*u)); 1 = official kernel "
                        "(findings 2026-09-30 residual x2 decode suspect)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="forward batch; lower it on small GPUs (32 is the A100 default). Results "
                        "are batch-invariant: patches are decoded independently into preallocated "
                        "maps")
    p.add_argument("--no-inst-probs", action="store_true",
                   help="skip the per-instance retype table (inst_img/inst_id/inst_prob) for a "
                        "memory-light run — enough for decode/lever experiments, not for pillar-A "
                        "re-typing")
    return p


def write_dead_predictions(run: Path, tag: str, dead_inst: np.ndarray) -> Path:
    """Write the expert's decoded candidates under their own tag; base artifacts are never touched."""
    path = run / f"pred_{tag}_dead.npz"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    np.savez_compressed(path, dead_inst=dead_inst)
    return path


def main(argv=None):
    a = build_parser().parse_args(argv)
    upscale = run_upscale(a.run, a.upscale)
    # a TrainConfig shim is the cheapest way to hand build_model the run's architecture flags
    # (the dead branch / widening live in the checkpoint; --dead only chooses whether to decode it)
    cfg = TrainConfig(split=1, out_dir=str(a.run), dead_expert=run_dead_expert(a.run),
                      widen=run_widen(a.run))
    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu", weights_only=False)["model"])
    f = PanNukeFold(a.fold)
    res = predict_fold(model, f, tta=a.tta, inst_probs=not a.no_inst_probs, upscale=upscale,
                       decode_u=a.decode_u, marker_u=a.marker_u, batch_size=a.batch_size,
                       dead_expert=a.dead)
    inst, typ, tissue = res[0], res[1], res[2]
    tag = (f"fold{a.fold}{res_tag(upscale)}" + (f"_du{a.decode_u:g}" if a.decode_u != 1.0 else "")
           + (f"_mk{a.marker_u:g}" if a.marker_u != 1.0 else "") + ("_tta" if a.tta else ""))
    # predict_fold already returns int32/uint8; guard instead of astype (astype copies by default
    # even when the dtype matches — 663+158 MB extra at the exact worst moment on 16 GB-VA hosts)
    if inst.dtype != np.int32:
        inst = inst.astype(np.int32)
    if typ.dtype != np.uint8:
        typ = typ.astype(np.uint8)
    payload = dict(inst=inst, type=typ, tissue_prob=tissue)
    if not a.no_inst_probs:
        payload.update(inst_img=res[3][0], inst_id=res[3][1], inst_prob=res[3][2])
    np.savez_compressed(a.run / f"pred_{tag}.npz", **payload)
    if a.dead:
        write_dead_predictions(a.run, tag, res[-1])
    if not a.no_eval:
        # NB on 16 GB-address-space hosts the eval is best run as a separate process
        # (scripts/eval_pannuke.py, fresh VA: gt_channels alone decompresses to 1.54 GiB uint16);
        # in-process is for big-RAM hosts.
        res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)
        save_report(res, a.run / f"eval_{tag}")
        print(tag, "\n" + format_summary(res["summary"]))


if __name__ == "__main__":
    main()
