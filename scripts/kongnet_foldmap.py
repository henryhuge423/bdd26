#!/usr/bin/env python
"""Resolve the KongNet_PanNuke_{1,2,3}.pth checkpoint -> fold mapping empirically.

The released checkpoints are per-split but the numbering is undocumented. A checkpoint
of official split s was trained on fold s, so its detection F1 on its TRAIN fold is
inflated (memorisation) by a wide margin. Decision rule: train fold = argmax over
folds of pooled detection F1 (centroid pairing, radius 12 px, HoVer-Net F_d semantics);
the split (and hence val/test folds) then follows from the official protocol table.
The rule never uses the identified test fold's numbers for any decision, and nothing
is tuned on any fold (peak threshold/min_distance are the repo's released defaults).

Runs the vendored KongNet (third-party repo on the machine, --repo) on PanNuke folds
via the compact memory-mapped format; per-class F1 also verifies the head->class
channel map ([5,8,11,14,17] = neoplastic..epithelial) on the train folds only.

Usage (ugrad, venv + ugrad_env.sh):
  python scripts/kongnet_foldmap.py --repo /tmp/cgf2604/repos/KongNet_Inference_Main \
      --weights-dir /tmp/cgf2604/weights_a1/kongnet --out runs/analysis/kongnet_foldmap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.feature import peak_local_max

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucseg.constants import CLASS_NAMES  # noqa: E402
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.instance import centroids, pair_coordinates  # noqa: E402
from nucseg.metrics.pannuke_eval import instance_classes  # noqa: E402

# wsi_inference_PanNuke.py: PanNuke variant of the released inference repo
NUM_HEADS, HEAD_CHANNELS = 6, 3
CLASS_CHANNELS = [5, 8, 11, 14, 17]  # 3rd channel of heads 1..5, official class order
PEAK_MIN_DISTANCE = 11  # post_proc_size == nms_box_size in their config
PEAK_THRESHOLD = 0.5
PAIR_RADIUS = 12.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = {1: (1, 2, 3), 2: (2, 1, 3), 3: (3, 2, 1)}  # split -> (train, val, test) folds


def build_model(repo: Path, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from model.KongNet import get_KongNet

    model = get_KongNet(num_heads=NUM_HEADS, decoders_out_channels=[HEAD_CHANNELS] * NUM_HEADS)
    return model.to(device).eval()


@torch.no_grad()
def predict_points(model, images: np.ndarray, device: str, batch_size: int):
    """List per image of (n,2) xy points and (n,) class ids from the class heatmaps."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).to(device)
        x = (x.float() / 255.0 - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probs = torch.sigmoid(model(x)).float()
        for j in range(probs.shape[0]):
            p = probs[j].cpu().numpy()
            pts, cls = [], []
            for c, ch in enumerate(CLASS_CHANNELS):
                peaks = peak_local_max(
                    p[ch], min_distance=PEAK_MIN_DISTANCE,
                    threshold_abs=PEAK_THRESHOLD, exclude_border=False,
                )
                pts.extend(peaks[:, ::-1].astype(np.float32))  # (row, col) -> xy
                cls.extend([c] * len(peaks))
            out.append((np.asarray(pts, np.float32).reshape(-1, 2), np.asarray(cls, int)))
    return out


def fold_points(fold: PanNukeFold, idx: np.ndarray):
    """GT (xy centroids, class) for the sampled images, aligned with instance ids."""
    pts, cls = [], []
    for i in idx:
        lab, g_cls = instance_classes(fold.inst[i], fold.type[i])  # g_cls: 1..5
        _, xy = centroids(lab)
        keep = g_cls > 0  # untyped GT instances (none expected) can't be class-scored
        pts.append(xy[keep])
        cls.append(g_cls[keep] - 1)
    return pts, cls


def f1(tp: int, n_a: int, n_b: int) -> float:
    return 2 * tp / (n_a + n_b) if (n_a + n_b) else float("nan")


def detection_scores(gt_pts, gt_cls, pd_pts, pd_cls):
    """Per-image (as pannuke_eval) pooled + per-class detection F1 within PAIR_RADIUS,
    plus a (pred-channel x gt-class) confusion over matched pairs to read off the
    channel->class permutation without assuming the repo's cell_channel_map."""
    n_c = len(CLASS_NAMES)
    tp = fp = fn = 0
    conf = np.zeros((n_c, n_c), int)
    pc = np.zeros((n_c, 3), int)  # per class: tp, n_pred, n_gt
    for g_xy, g_c, p_xy, p_c in zip(gt_pts, gt_cls, pd_pts, pd_cls):
        pairing, up, ug = pair_coordinates(p_xy, g_xy, PAIR_RADIUS)
        tp += len(pairing); fp += len(up); fn += len(ug)
        if len(pairing):
            np.add.at(conf, (p_c[pairing[:, 0]], g_c[pairing[:, 1]]), 1)
        for c in range(n_c):
            gm, pm = g_c == c, p_c == c
            m, _, _ = pair_coordinates(p_xy[pm], g_xy[gm], PAIR_RADIUS)
            pc[c] += (len(m), int(pm.sum()), int(gm.sum()))
    res = {
        "n_gt": int(tp + fn), "n_pred": int(tp + fp), "pooled_tp": int(tp),
        "pooled_f1": f1(tp, tp + fp, tp + fn),
        "per_class": {CLASS_NAMES[c]: {"tp": int(pc[c, 0]), "n_pred": int(pc[c, 1]),
                                        "n_gt": int(pc[c, 2]), "f1": f1(*pc[c])}
                      for c in range(n_c)},
        "confusion": conf.tolist(),
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True, help="KongNet_Inference_Main checkout")
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/analysis/kongnet_foldmap.json"))
    args = ap.parse_args()

    model = build_model(args.repo, args.device)
    results, table = [], {}
    for j in args.checkpoints:
        # 16 GB address-space cap on ugrad: CUDA context (~9 GB) + one 2.1 GB ckpt at a time
        ckpt = torch.load(args.weights_dir / f"KongNet_PanNuke_{j}.pth",
                          map_location="cpu", weights_only=True)
        epoch = ckpt.get("epoch")
        model.load_state_dict(ckpt["model"], strict=True)
        del ckpt
        for k in args.folds:
            fold = PanNukeFold(k)
            idx = np.unique(np.linspace(0, len(fold) - 1, args.n_images).astype(int))
            preds = predict_points(model, np.asarray(fold.images[idx]), args.device, args.batch_size)
            pd_pts, pd_cls = zip(*preds)
            gt_pts, gt_cls = fold_points(fold, idx)
            s = detection_scores(gt_pts, gt_cls, pd_pts, pd_cls)
            s.update(checkpoint=j, fold=k, ckpt_epoch=epoch, n_images=len(idx))
            results.append(s)
            table[(j, k)] = s["pooled_f1"]
            print(f"ckpt {j} fold {k}: F1 {s['pooled_f1']:.4f} "
                  f"(gt {s['n_gt']}, pred {s['n_pred']})", flush=True)

    # Decision: train fold = argmax pooled F1 per checkpoint; must be a permutation.
    mapping = {j: max(args.folds, key=lambda k: table[(j, k)]) for j in args.checkpoints}
    perm_ok = len(set(mapping.values())) == len(mapping)
    print("\nPooled detection F1 matrix (rows=checkpoint, cols=fold):")
    print("ckpt  " + "  ".join(f"fold{k}" for k in args.folds) + "   -> train fold (split: train/val/test)")
    for j in args.checkpoints:
        cells = "  ".join(f"{table[(j, k)]:.4f}" for k in args.folds)
        tr = mapping[j]
        print(f"  {j}   {cells}   -> {tr} (split {tr}: {SPLITS[tr]})")
    if not perm_ok:
        print("WARNING: argmax folds are not a permutation - mapping ambiguous, rerun with --n-images higher")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"results": results, "mapping": mapping, "permutation_ok": perm_ok,
         "decision_rule": "train fold = argmax_fold pooled detection F1 (radius 12 px centroid pairing); "
                          "repo-default peak decode (thr .5, min_distance 11); no tuning"},
        indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
