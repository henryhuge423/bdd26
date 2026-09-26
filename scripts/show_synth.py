#!/usr/bin/env python
"""Montage viewer for synth_pannuke.py outputs (rows: base real | generated | labels).

Panel 3 draws the KEPT teacher-confirmed instance outlines on the generated image
(green = kept object, red = kept object typed Dead) so a reviewer can check placement
and spot visible-but-unlabelled nuclei (the known teacher-blind-spot caveat).

    python scripts/show_synth.py --dir runs/pixcell/synth_smoke_v2 --fold 1 --rows 12
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nucseg.data.pannuke import PanNukeFold

p = argparse.ArgumentParser()
p.add_argument("--dir", type=Path, required=True)
p.add_argument("--fold", type=int, required=True, help="train fold the layouts came from")
p.add_argument("--rows", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

z = np.load(a.dir / "images.npy")
inst = np.load(a.dir / "inst.npy").astype(np.int32)
typ = np.load(a.dir / "type.npy")
base = np.load(a.dir / "base.npy")
f = PanNukeFold(a.fold)
sel = np.random.default_rng(a.seed).choice(len(z), min(a.rows, len(z)), replace=False)

rows = []
for i in sel:
    real = np.asarray(f.images[int(base[i])])
    gen = z[i]
    ov = gen.copy()
    for k in np.unique(inst[i])[1:]:
        m = (inst[i] == k).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = (255, 64, 64) if typ[i][inst[i] == k].max() == 4 else (64, 255, 64)
        cv2.drawContours(ov, cnts, -1, color, 1)
    rows.append(np.concatenate([real, gen, ov], 1))
sep = np.full((2, rows[0].shape[1], 3), 255, np.uint8)
out = np.concatenate([r for row in rows for r in (row, sep)], 0)
Image.fromarray(out).save(a.dir / "montage.png")
print("wrote", a.dir / "montage.png", len(sel), "rows")
