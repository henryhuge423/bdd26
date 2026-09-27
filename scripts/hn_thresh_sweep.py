#!/usr/bin/env python
"""Flat fg/seed threshold sweep for the HoVer-NeXt port (ceiling test for the raw maps).

    python scripts/hn_thresh_sweep.py --weights runs/hovernext_t/split1 --fold 3 --n 400

Computes raw maps once, then evaluates the validation-style decode (flat fg/seed + softmax-sum
typing) over a grid, in parallel. If the best cell lands near the paper's 0.477 the maps are
fine and only the decoding (their padded multi-window npy pipeline / TTA) is missing; a ceiling
around the port's 0.36-0.43 would mean the forward path itself is off.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import find_objects

from nucseg.data.external import LazyGTChannels
from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernext.engine import _faster_instance_seg, build_model
from nucseg.metrics.pannuke_eval import evaluate

p = argparse.ArgumentParser()
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--n", type=int, default=400)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--out", type=Path, default=None,
               help="write the best cell here as decode.json (give the weights dir)")
a = p.parse_args()

hn = build_model(a.weights)
m = hn.model.cuda().eval()
f = PanNukeFold(a.fold)
idx = np.linspace(0, len(f.images) - 1, a.n).astype(int)
gt_inst, gt_type = np.asarray(f.inst)[idx], np.asarray(f.type)[idx]
gt_ch = LazyGTChannels(gt_inst, gt_type)
gt_tis = np.asarray(f.tissue)[idx]

maps = []
with torch.inference_mode():
    for s in range(0, len(idx), a.bs):
        x = torch.from_numpy(np.asarray(f.images[idx[s:s + a.bs]], np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", torch.float16):
            out = m(x).float()
        for b in range(len(x)):
            maps.append((out[b, 2:5].softmax(0).cpu().numpy(), out[b, 5:].softmax(0).cpu().numpy()))
print(f"maps for {len(maps)} images computed")

FGS = [0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0.3, 0.4, 0.5, 0.6, 0.7]
flat = np.ones((1, 256, 256), bool)


def decode(inst3, cls6, fg_t, seed_t):
    lab, _ = _faster_instance_seg(inst3[:2], flat, np.array([fg_t]), np.array([seed_t]))
    probs = cls6[1:]
    typ = np.zeros(lab.shape, np.uint8)
    for j, sl in enumerate(find_objects(lab)):
        if not sl:
            continue
        sel = lab[sl] == (j + 1)
        v = int(np.sum(probs[(slice(None), *sl)][:, sel], axis=1).argmax()) + 1
        typ[lab == j + 1] = v
    return lab, typ


def score(args):
    fg_t, seed_t = args
    insts, typs = [], []
    for inst3, cls6 in maps:
        i, t = decode(inst3, cls6, fg_t, seed_t)
        insts.append(i)
        typs.append(t)
    r = evaluate(gt_ch, gt_inst, gt_type, gt_tis, np.stack(insts), np.stack(typs), workers=8)
    s = r["summary"]
    return fg_t, seed_t, s["official"]["mPQ"], s["official"]["bPQ"], s["detection"]["F_d"]


if __name__ == "__main__":
    grid = [(fg, sd) for fg in FGS for sd in SEEDS]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(score, grid))
    rows.sort(key=lambda r: -r[2])
    print(f"{'fg':>4} {'seed':>4} {'mPQ':>7} {'bPQ':>7} {'F_d':>6}")
    for fg_t, seed_t, mpq, bpq, fd in rows:
        print(f"{fg_t:>4} {seed_t:>4} {mpq:>7.4f} {bpq:>7.4f} {fd:>6.3f}")
    if a.out:
        import json
        fg_t, seed_t, mpq, _, _ = rows[0]
        a.out.write_text(json.dumps({"fg": fg_t, "seed": seed_t, "val_mPQ": round(mpq, 4),
                                     "fold": a.fold, "n": a.n}))
        print(f"decode.json written to {a.out}")
