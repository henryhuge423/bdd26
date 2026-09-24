#!/usr/bin/env python
"""Montage of GT nuclei by error status for visual inspection of annotation quality.

    python scripts/show_nuclei.py --eval runs/cellvit_uni/split1/eval_test_fold3 \
        --pred runs/cellvit_uni/split1/pred_test_fold3.npz --fold 3 --cls 4 --out runs/analysis/dead_split1.png

Rows: GT nuclei of class --cls that were missed (no overlapping prediction), matched, and (reference)
missed nuclei of other classes. Each tile: 64x64 px (16 um) around the GT centroid, x3; green = the GT
nucleus, cyan = other GT nuclei, yellow = predicted instances.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import instance_classes

p = argparse.ArgumentParser()
p.add_argument("--eval", type=Path, required=True)
p.add_argument("--pred", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--cls", type=int, default=4)
p.add_argument("--n", type=int, default=12, help="tiles per row")
p.add_argument("--rows", type=int, default=2, help="rows per group")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

g = pd.read_csv(a.eval / "gt_records.csv.gz")
g["idx"] = g.groupby("image").cumcount()  # 0-based relabelled GT instance index (as in the evaluator)
f = PanNukeFold(a.fold)
pred = np.load(a.pred)["inst"]
rng = np.random.default_rng(a.seed)
k = a.n * a.rows
groups = [(f"{CLASS_NAMES[a.cls - 1]} missed", g[(g.cls == a.cls) & (g.status == "missed_bg")]),
          (f"{CLASS_NAMES[a.cls - 1]} matched", g[(g.cls == a.cls) & (g.status == "matched")]),
          ("other classes missed", g[(g.cls != a.cls) & (g.status == "missed_bg")])]


def contour(img, mask, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, 1)


def tile(row):
    j = int(row.image)
    lab, _ = instance_classes(np.asarray(f.inst[j]), np.asarray(f.type[j]))
    m = lab == row.idx + 1
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = np.clip(cy - 32, 0, 192), np.clip(cx - 32, 0, 192)
    sl = (slice(y0, y0 + 64), slice(x0, x0 + 64))
    s = 3
    img = cv2.resize(np.asarray(f.images[j])[sl], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)[..., ::-1].copy()
    up = lambda x: cv2.resize(x.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    for i in np.unique(pred[j][sl]):
        if i:
            contour(img, up(pred[j][sl] == i), (0, 255, 255))
    for i in np.unique(lab[sl]):
        if i and i != row.idx + 1:
            contour(img, up(lab[sl] == i), (255, 255, 0))
    contour(img, up(m[sl]), (0, 255, 0))
    cv2.putText(img, f"{CLASS_NAMES[int(row.cls) - 1][:4]} {int(row.area)}px", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


rows = []
for name, d in groups:
    sel = d.iloc[rng.permutation(len(d))[:k]]
    tiles = [tile(r) for r in sel.itertuples()]
    tiles += [np.zeros_like(tiles[0])] * (k - len(tiles))
    header = np.zeros((20, tiles[0].shape[1] * a.n, 3), np.uint8)
    cv2.putText(header, f"{name} (n={len(d)})", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(header)
    rows += [np.hstack(tiles[r * a.n:(r + 1) * a.n]) for r in range(a.rows)]
a.out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(a.out), np.vstack(rows))
print("->", a.out)
