#!/usr/bin/env python
"""Diagnose the ZERO-Dead-labels blocker found by the SYN-v2 smoke review (2026-09-26).

The smoke set (base model + paired ctx + Reinhard) kept 729/809 teacher-confirmed objects but
typed NONE of them Dead, although ~38% of inserted donors were Dead-class and Dead is ~11% of
base-layout GT objects. Two candidate causes:

1. the teacher / --type-thresh .6 gate: instance_mean_probs AVERAGES softmax probabilities, and
   Dead predictions are low-confidence — the gate may drop or re-type Dead systematically;
2. synthesis (+ Reinhard) destroys Dead morphology.

Part 1 answers (1) on REAL data: run the OOF teacher (split 2; fold 1 = its val fold) on Dead-rich
real fold-1 patches and report, per GT class, the argmax-type histogram and the max-prob /
P(Dead) distributions of best-coverage predictions. Part 2 answers (2): re-run the teacher on the
smoke synthetic images and report P(Dead) for the kept objects (confusion ~.3 runner-up vs
morphology loss ~.02).

    python scripts/diag_synth_dead.py --teacher runs/cellvit_uni_split2_final.pth --fold 1 \
        [--smoke runs/pixcell/synth_smoke_v2] [--n-real 64]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import _forward_probs, _post, build_model
from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.text.conch_prior import instance_mean_probs

p = argparse.ArgumentParser()
p.add_argument("--teacher", type=Path, required=True, help="CellViT final.pth, out-of-fold for --fold")
p.add_argument("--fold", type=int, default=1)
p.add_argument("--n-real", type=int, default=64, help="Dead-rich real patches for part 1")
p.add_argument("--smoke", type=Path, default=None, help="synth_pannuke.py output dir for part 2")
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()


def teacher_batch(model, imgs):
    """Mirror synth_pannuke.py's labelling pass (bf16 autocast, same post-processing)."""
    x = torch.from_numpy(np.stack(imgs)).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pr = _forward_probs(model, x)
    prob = pr["tp"].float().cpu().numpy()
    tp = pr["tp"].argmax(-1, keepdim=True).float()
    maps = torch.cat([tp, pr["np"][..., 1:], pr["hv"]], -1).cpu().numpy()
    with Pool(8) as pool:
        preds = [r[0] for r in pool.map(_post, ((m,) for m in maps), chunksize=2)]
    return preds, prob


def best_cov_probs(pred, prob_b, inst):
    """For each GT object id: (best-coverage prediction's 6-vector probs, coverage)."""
    pids = np.unique(pred)[1:]
    probs = instance_mean_probs(pred, prob_b, pids) if len(pids) else np.zeros((0, 6))
    out = {}
    for k in np.unique(inst)[1:]:
        obj = inst == k
        best, best_cov = -1, 0.0
        for pi, pid in enumerate(pids):
            inter = (pred == pid) & obj
            if inter.sum() and inter.sum() / obj.sum() > best_cov:
                best, best_cov = pi, inter.sum() / obj.sum()
        out[k] = (probs[best], best_cov)
    return out


def main():
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(a.teacher, map_location="cpu", weights_only=False)["model"])
    f = PanNukeFold(a.fold)

    # ---------------------------------------------------------------- part 1: REAL patches
    n_dead = [int((np.asarray(f.type[j])[np.asarray(f.inst[j]) > 0] == 4).sum()) for j in range(len(f))]
    idx = sorted(range(len(f)), key=lambda j: -n_dead[j])[:a.n_real]
    rec = {}  # gt class -> list of (argmax, maxp, p_dead)
    for s in range(0, len(idx), a.batch):
        js = idx[s:s + a.batch]
        preds, prob = teacher_batch(model, [np.asarray(f.images[j]) for j in js])
        for j, pred, pb in zip(js, preds, prob):
            inst = np.asarray(f.inst[j])
            typ = np.asarray(f.type[j])
            bcp = best_cov_probs(pred, pb, inst)
            for k, (pr, cov) in bcp.items():
                if cov < 0.4:
                    continue
                obj = inst == k
                g = int(np.bincount(typ[obj], minlength=8).argmax())
                rec.setdefault(g, []).append((int(pr.argmax()), float(pr.max()), float(pr[4])))
    print(f"[part1] REAL fold-{a.fold} patches, GT objects with coverage >= .4 (argmax / maxp / P(Dead))")
    for g in sorted(rec):
        r = np.array(rec[g])  # (n, 3)
        hist = {CLASS_NAMES[c - 1]: int((r[:, 0] == c).sum()) for c in range(1, 6) if (r[:, 0] == c).any()}
        print(f"  GT {CLASS_NAMES[g - 1]:13s} n={len(r):5d} argmax={hist} "
              f"maxp med {np.median(r[:, 1]):.2f} (>=.6: {(r[:, 1] >= .6).mean():.2f}, >=.4: {(r[:, 1] >= .4).mean():.2f}) "
              f"P(Dead) med {np.median(r[:, 2]):.3f}")

    # ---------------------------------------------------------------- part 2: SMOKE synthetics
    if a.smoke is None:
        return
    imgs = np.load(a.smoke / "images.npy")
    kept_inst = np.load(a.smoke / "inst.npy").astype(np.int32)
    kept_type = np.load(a.smoke / "type.npy")
    pd_all, mx_all, am_all, lbl_all = [], [], [], []
    for s in range(0, len(imgs), a.batch):
        preds, prob = teacher_batch(model, imgs[s:s + a.batch])
        for b, (pred, pb) in enumerate(zip(preds, prob)):
            inst = kept_inst[s + b]
            bcp = best_cov_probs(pred, pb, inst)
            for k, (pr, cov) in bcp.items():
                if cov < 0.4:
                    continue
                pd_all.append(float(pr[4]))
                mx_all.append(float(pr.max()))
                am_all.append(int(pr.argmax()))
                lbl_all.append(int(kept_type[s + b][inst == k].max()))
    pd_all, mx_all, am_all, lbl_all = map(np.array, (pd_all, mx_all, am_all, lbl_all))
    print(f"\n[part2] SMOKE kept objects, teacher re-read (n={len(pd_all)})")
    print(f"  argmax==Dead: {(am_all == 4).sum()}   label==Dead: {(lbl_all == 4).sum()}")
    print(f"  P(Dead) percentiles 50/75/90/95/99: "
          f"{np.percentile(pd_all, [50, 75, 90, 95, 99]).round(3)}")
    print(f"  objects with P(Dead)>=.3: {(pd_all >= .3).sum()}  >=.4: {(pd_all >= .4).sum()}  >=.5: {(pd_all >= .5).sum()}")
    top = np.argsort(-pd_all)[:10]
    for i in top:
        print(f"    P(Dead)={pd_all[i]:.3f} argmax={CLASS_NAMES[am_all[i] - 1]} maxp={mx_all[i]:.2f} "
              f"label={CLASS_NAMES[lbl_all[i] - 1]}")


if __name__ == "__main__":
    main()
