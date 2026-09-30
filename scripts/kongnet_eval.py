#!/usr/bin/env python
"""KongNet (detection-first A1) under our strict PanNuke protocol.

Phase-2 line A: KongNet is the untested detection-first family for the Dead/resolution
question (2026-09-28 artifact entry: every dense decoder shares the Dead deficit; Dead
F_c reference points are HoVer-NeXt paper .49 and detection-first KongNet paper .59).
Checkpoints are per-split (mapping resolved empirically -> `kongnet_foldmap.json`;
identity, train-fold-F1 margins >= .13). All decode thresholds are tuned on VAL folds
only; test folds are run once with the chosen constants.

Channel semantics: the released pipeline uses only channel 2 of each head (class
heatmap). Channels 0/1 are undocumented; `--mode probe` resolves seg vs contour
against train-fold GT (filled mask should have high IoU with the class foreground,
the contour channel should sit on its boundary) before anything else runs.

Modes:
  probe   channel semantics + decode smoke PQ on a few TRAIN-fold images (GT-aware sample)
  sweep   forward a VAL-fold subset once + decode-threshold grid -> val mPQ/bPQ
  predict full-fold forward + decode -> pred_fold{k}_kong_c{ckpt}.npz for eval_pannuke.py

Usage (ugrad, after `source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh`):
  python scripts/kongnet_eval.py --repo /tmp/cgf2604/repos/KongNet_Inference_Main \
      --weights-dir /tmp/cgf2604/weights_a1/kongnet --ckpt 1 --mode probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucseg.constants import CLASS_NAMES  # noqa: E402
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.kongnet.decode import (CLASS_HEATMAP_CH, DEFAULT_CONTOUR_THR,  # noqa: E402
                                   DEFAULT_SEG_THR, decode_instances)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = {1: (1, 2, 3), 2: (2, 1, 3), 3: (3, 2, 1)}  # split -> (train, val, test)


def build_model(repo: Path, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from model.KongNet import get_KongNet

    model = get_KongNet(num_heads=6, decoders_out_channels=[3] * 6)
    return model.to(device).eval()


@torch.no_grad()
def fold_probs(model, images: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """(N, 256, 256, 18) sigmoid probs, released-pipeline preprocessing (ImageNet
    norm, fp16 autocast) as validated by kongnet_foldmap.py."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.empty((len(images), images.shape[1], images.shape[2], 18), np.float16)
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[i : i + batch_size]))
        x = x.permute(0, 3, 1, 2).to(device)
        x = (x.float() / 255.0 - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probs = torch.sigmoid(model(x)).float()
        out[i : i + len(x)] = probs.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
    return out


def gt_class_fg(fold: PanNukeFold, i: int) -> np.ndarray:
    """(5, H, W) bool GT foreground per class for one image (type.npy is 1..5)."""
    return np.stack([(fold.type[i] == c + 1) & (fold.inst[i] > 0) for c in range(5)])


def _iou_stats(ch: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(fg IoU of ch>0.5 vs gt, fraction of the GT boundary ring covered by ch>0.5)."""
    pred = ch > 0.5
    union = (pred | gt).sum()
    iou = (pred & gt).sum() / union if union else 0.0
    ring = gt ^ ndi_compat(gt)
    return iou, float((pred & ring).sum() / ring.sum()) if ring.sum() else 0.0


def mode_probe(model, args, device) -> dict:
    """Channel semantics on train-fold images + a decode smoke PQ on the subset.

    Head layout (verified 2026-09-30): head 0 = overall (ch0 filled fg, ch1 contour,
    ch2 heatmap); head c+1 = class c -> (seg 3c+3, contour 3c+4, heatmap 3c+5). Rare
    classes are sampled GT-aware (images CONTAINING the class) so their seg/contour
    channels are measured where they matter."""
    from nucseg.metrics import light

    fold = PanNukeFold(SPLITS[args.ckpt][0])  # train fold of this checkpoint
    cand = np.unique(np.linspace(0, len(fold) - 1, 6 * args.n_images).astype(int))
    per_class = {c: [] for c in range(5)}
    for i in cand:
        t = np.asarray(fold.type[i])
        for c in range(5):
            if len(per_class[c]) < args.n_images and (t == c + 1).any():
                per_class[c].append(int(i))
        if all(len(v) >= args.n_images for v in per_class.values()):
            break
    idx = np.array(sorted({i for v in per_class.values() for i in v}))
    probs = fold_probs(model, np.asarray(fold.images[idx]), device, args.batch_size)
    pos = {i: j for j, i in enumerate(idx)}

    stats = {c: {} for c in range(5)}
    for c in range(5):
        seg_ch, ct_ch, hm_ch = 3 * c + 3, 3 * c + 4, CLASS_HEATMAP_CH[c]
        for name, ch_i in ((f"seg_ch{seg_ch}_fg_iou", seg_ch), (f"seg_ch{seg_ch}_ring", seg_ch),
                           (f"contour_ch{ct_ch}_fg_iou", ct_ch), (f"contour_ch{ct_ch}_ring", ct_ch),
                           (f"heatmap_ch{hm_ch}_fg_iou", hm_ch), (f"heatmap_ch{hm_ch}_ring", hm_ch)):
            vals = []
            for i in per_class[c]:
                gt = gt_class_fg(fold, i)[c]
                iou, ring = _iou_stats(probs[pos[i], :, :, ch_i].astype(np.float32), gt)
                vals.append(ring if name.endswith("_ring") else iou)
            stats[c][name] = float(np.mean(vals)) if vals else 0.0
        stats[c]["n_images"] = len(per_class[c])

    # decode smoke at the repo-default thresholds on the pooled subset
    smoke = {}
    sc, cc = [3, 6, 9, 12, 15], [4, 7, 10, 13, 16]
    pred, pred_type = zip(*[decode_instances(probs[j].astype(np.float32), sc, cc)
                            for j in range(len(idx))])
    rows = []
    for j, i in enumerate(idx):
        gt_i, gt_t = np.asarray(fold.inst[int(i)]), np.asarray(fold.type[int(i)])
        gt_ch = np.stack([np.where(gt_t == c + 1, gt_i, 0) for c in range(5)], -1).astype(np.uint16)
        rows.append(light.image_stats(gt_ch, gt_i, gt_t, pred[j], pred_type[j]))
    stats_arr = np.stack(rows)
    s = light.summarize(stats_arr, tissue)
    smoke["repo_defaults"] = {k: s[k] for k in ("mPQ", "bPQ") if k in s}
    return {"stats": {CLASS_NAMES[c]: stats[c] for c in range(5)}, "smoke": smoke,
            "n_images": len(idx), "fold": fold.fold}


_KONG: dict = {}


def _decode_stats(j: int) -> list:
    """Worker: decode subset image j (= fold image _KONG['idx'][j]) at every (seg_thr,
    contour_thr) and score vs ITS fold image. GT class channels are rebuilt per image
    from inst+type (the full-fold gt_channels npz breaches the ugrad 16 GB address-space
    cap next to the model + CUDA context)."""
    from nucseg.metrics import light

    probs, fold, idx = _KONG["probs"], _KONG["fold"], _KONG["idx"]
    fi = int(idx[j])
    gt_i, gt_t = np.asarray(fold.inst[fi]), np.asarray(fold.type[fi])
    gt_ch = np.stack([np.where(gt_t == c + 1, gt_i, 0) for c in range(5)], -1).astype(np.uint16)
    out = []
    for seg_thr, contour_thr in _KONG["grid"]:
        inst, typ = decode_instances(probs[j].astype(np.float32), _KONG["seg_ch"],
                                     _KONG["contour_ch"], seg_thr, contour_thr)
        out.append(light.image_stats(gt_ch, gt_i, gt_t, inst, typ))
    return out


def mode_sweep(model, args, device) -> None:
    """Forward a VAL-fold subset once, sweep decode thresholds, score with light mPQ.
    The sweep sample is a plain linspace (representative of the fold), NOT GT-aware."""
    from multiprocessing import Pool

    from nucseg.metrics import light

    fold = PanNukeFold(args.fold)
    idx = np.unique(np.linspace(0, len(fold) - 1, args.n_images).astype(int))
    probs = fold_probs(model, np.asarray(fold.images[idx]), device, args.batch_size)
    grid = [(s, c) for s in args.seg_thrs for c in args.contour_thrs]
    _KONG.update(probs=probs, fold=fold, idx=idx, grid=grid, seg_ch=args.seg_ch,
                 contour_ch=args.contour_ch)
    with Pool(args.workers) as pool:
        per_img = pool.map(_decode_stats, range(len(idx)), chunksize=4)
    stats = np.stack(per_img)  # (n_img, n_cfg, 6, 5)
    tissue = np.asarray(fold.tissue)[idx]
    rows = []
    for c, (seg_thr, contour_thr) in enumerate(grid):
        s = light.summarize(stats[:, c], tissue)
        rows.append({"seg_thr": seg_thr, "contour_thr": contour_thr,
                     "mPQ": s["mPQ"], "bPQ": s["bPQ"]})
        print(f"seg {seg_thr:.2f} cont {contour_thr:.2f}: "
              f"mPQ {s['mPQ']:.4f} bPQ {s['bPQ']:.4f}", flush=True)
    best = max(rows, key=lambda r: r["mPQ"])
    print(f"best: {best}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out = args.out.with_suffix(".json")
    args.out.write_text(json.dumps({"rows": rows, "best": best, "seg_ch": args.seg_ch,
                                    "contour_ch": args.contour_ch, "fold": args.fold,
                                    "n_images": len(idx),
                                    "ckpt": args.ckpt}, indent=1))
    print(f"-> {args.out}")


def mode_predict(model, args, device) -> None:
    fold = PanNukeFold(args.fold)
    inst = np.empty((len(fold), 256, 256), np.int32)
    typ = np.empty((len(fold), 256, 256), np.int64)
    for i in range(0, len(fold), args.batch_size):
        n = min(args.batch_size, len(fold) - i)
        probs = fold_probs(model, np.asarray(fold.images[i : i + n]), device, n)
        for j in range(n):
            p = probs[j].astype(np.float32)
            inst[i + j], typ[i + j] = decode_instances(
                p, args.seg_ch, args.contour_ch, args.seg_thr, args.contour_thr)
        if (i // args.batch_size) % 100 == 0:
            print(f"{i + n}/{len(fold)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, inst=inst, type=typ)
    print(f"wrote {args.out}")


def ndi_compat(gt: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_erosion(gt, np.ones((3, 3), bool), border_value=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=int, default=1, help="KongNet_PanNuke_{1,2,3}.pth")
    ap.add_argument("--fold", type=int, default=None, help="fold for cache/predict "
                    "(default: the checkpoint's VAL fold for cache, TEST fold for predict)")
    ap.add_argument("--mode", choices=["probe", "sweep", "predict"], required=True)
    ap.add_argument("--n-images", type=int, default=64, help="probe/sweep subset size "
                    "(sweep: plain linspace of the VAL fold, default 800 there)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seg-thr", type=float, default=DEFAULT_SEG_THR)
    ap.add_argument("--contour-thr", type=float, default=DEFAULT_CONTOUR_THR)
    ap.add_argument("--seg-channels", type=int, nargs=5, default=[3, 6, 9, 12, 15],
                    help="seg channel of each class head = 3(c+1) (resolved by probe)")
    ap.add_argument("--contour-channels", type=int, nargs=5, default=[4, 7, 10, 13, 16])
    ap.add_argument("--seg-thrs", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7], help="sweep grid")
    ap.add_argument("--contour-thrs", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.seg_ch, args.contour_ch = args.seg_channels, args.contour_channels
    if args.out is None:
        args.out = Path(f"runs/kongnet/ckpt{args.ckpt}_{args.mode}.npz")

    if args.mode == "sweep" and args.fold is None:
        args.fold = SPLITS[args.ckpt][1]  # VAL fold of this checkpoint
    if args.mode == "sweep" and args.n_images == 64:
        args.n_images = 800
    model = build_model(args.repo, args.device)
    # every GPU mode runs the released checkpoint (2.1 GB; the ugrad 16 GB address-space
    # cap needs one ckpt at a time, del after load)
    ckpt = torch.load(args.weights_dir / f"KongNet_PanNuke_{args.ckpt}.pth",
                      map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    if args.mode == "probe":
        res = mode_probe(model, args, args.device)
        print(json.dumps(res, indent=1))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out = args.out.with_suffix(".json")
        args.out.write_text(json.dumps(res, indent=1))
        print(f"-> {args.out}")
        return
    if args.mode == "sweep":
        mode_sweep(model, args, args.device)
    elif args.mode == "predict":
        if args.fold is None:
            args.fold = SPLITS[args.ckpt][2]
        if "pred_fold" not in args.out.name:
            args.out = args.out.parent / f"pred_fold{args.fold}_kong_c{args.ckpt}.npz"
        mode_predict(model, args, args.device)


if __name__ == "__main__":
    main()
