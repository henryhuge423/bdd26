#!/usr/bin/env python
"""Visual sanity montage for the CP1 copy-paste augmentation.

Renders [original | augmented] side by side with instance outlines (pasted nuclei in red,
original nuclei by class colour) so the isolation clearance and label consistency can be
checked by eye (montage review goes to a sonnet subagent).

    python scripts/show_copy_paste.py --fold 1 --n 10 --out runs/analysis/copy_paste.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from nucseg.augment.copy_paste import CopyPasteConfig, NucleusBank, apply_copy_paste
from nucseg.cellvit.data import cellvit_train_augs
from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=1)
p.add_argument("--n", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--prob", type=float, default=1.0)
p.add_argument("--no-augs", action="store_true", help="skip albumentations (isolates the CP effect)")
p.add_argument("--out", type=Path, default=Path("runs/analysis/copy_paste.png"))
a = p.parse_args()

CLASS_COLORS = [(80, 180, 0), (0, 200, 255), (255, 140, 0), (0, 80, 255), (200, 0, 200)]  # BGR, 1..5
RED = (0, 0, 255)


def contour(img, mask, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, 1)


def main():
    f = PanNukeFold(a.fold)
    bank = NucleusBank([f])
    cfg = CopyPasteConfig(prob=a.prob)
    rng = np.random.default_rng(a.seed)
    augs = None if a.no_augs else cellvit_train_augs()
    # patches with some free stroma + a few with many Dead
    n_nuc = np.array([len(np.unique(np.asarray(f.inst[j]))) - 1 for j in range(len(f))])
    n_dead = np.array([(np.asarray(f.type[j]) == 4).any() for j in range(len(f))])
    cand = np.flatnonzero((n_nuc > 10) & (n_nuc < 90))
    dead_first = cand[n_dead[cand]]
    idx = list(dead_first[:a.n // 2]) + list(rng.choice(np.setdiff1d(cand, dead_first[:a.n // 2]),
                                                        a.n - a.n // 2, replace=False))
    rows = []
    for j in idx:
        img = np.asarray(f.images[j]).copy()
        inst = np.asarray(f.inst[j]).astype(np.int32)
        typ = np.asarray(f.type[j]).astype(np.int32)
        img1, inst1, typ1 = apply_copy_paste(img.copy(), inst.copy(), typ.copy(), bank, cfg, rng)
        if augs is not None:  # same order as training: paste first, then augment jointly
            r = augs(image=img1, mask=np.stack([inst1, typ1], -1))
            img1, inst1, typ1 = r["image"], r["mask"][..., 0], r["mask"][..., 1]
        new_ids = set(np.setdiff1d(np.unique(inst1), np.unique(inst)))
        for im, ins, ty in ((img, inst, typ), (img1, inst1, typ1)):
            canvas = im[..., ::-1].copy()  # RGB -> BGR for cv2 drawing
            for k in np.unique(ins[ins > 0]):
                color = RED if int(k) in new_ids else CLASS_COLORS[int(ty[ins == k][0]) - 1]
                contour(canvas, ins == k, color)
            rows.append(canvas)
    H = W = 256
    grid = []
    for r in range(0, len(rows), 2):
        pair = np.concatenate([rows[r], rows[r + 1]], axis=1)
        grid.append(pair)
    canvas = np.concatenate(grid, axis=0)
    # red legend strip
    strip = np.zeros((28, canvas.shape[1], 3), np.uint8)
    for i, (name, c) in enumerate(zip(CLASS_NAMES + ["PASTED"], CLASS_COLORS + [RED])):
        cv2.putText(strip, f"{name}", (10 + i * 240, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    tuple(int(x) for x in c), 1, cv2.LINE_AA)
    canvas = np.concatenate([strip, canvas], axis=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), canvas)
    print("wrote", a.out, canvas.shape)


if __name__ == "__main__":
    main()
