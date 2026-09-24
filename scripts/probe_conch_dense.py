#!/usr/bin/env python
"""Per-nucleus CONCH embeddings from ONE trunk pass per 256px patch, via radius-restricted attentional
pooling (key mask = tokens within R px of the nucleus centroid). Reports zero-shot balanced accuracy
vs. R -- can CONCH give a cheap dense text-aligned prior for every nucleus?

    python scripts/probe_conch_dense.py --fold 1 --radii 24 48 96 1000
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from nucseg.data.pannuke import PanNukeFold
from nucseg.text.prototypes import load_conch

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=1)
p.add_argument("--protos", default="weights/text_protos/conch_v1.pt")
p.add_argument("--radii", type=float, nargs="+", default=[24, 48, 96, 1000])
p.add_argument("--size", type=int, default=448, help="CONCH input size (256 patch is resized to this)")
p.add_argument("--per-class", type=int, default=800)
a = p.parse_args()

protos = torch.load(a.protos)
f = PanNukeFold(a.fold)
rng = np.random.default_rng(0)
recs = []
for j in range(len(f)):
    inst, typ = np.asarray(f.inst[j]), np.asarray(f.type[j])
    ids = np.unique(inst)[1:]
    if len(ids) == 0:
        continue
    cms = ndimage.center_of_mass(np.ones_like(inst), inst, ids)
    for iid, (cy, cx) in zip(ids, cms):
        recs.append((j, cy, cx, int(np.bincount(typ[inst == iid]).argmax())))
recs = np.array(recs)
sel = np.concatenate([rng.permutation(np.where(recs[:, 3] == c)[0])[:a.per_class] for c in range(1, 6)])
recs = recs[sel]
y = recs[:, 3].astype(int)

model, _, _ = load_conch()
vis = model.visual
mean = torch.tensor(vis.image_mean if hasattr(vis, "image_mean") else (0.485, 0.456, 0.406)).view(1, 3, 1, 1).cuda()
std = torch.tensor(vis.image_std if hasattr(vis, "image_std") else (0.229, 0.224, 0.225)).view(1, 3, 1, 1).cuda()
g = a.size // 16
tok_xy = (torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), -1).float() + 0.5) * (256 / g)
tok_xy = tok_xy.view(-1, 2).cuda()  # (L, 2) token centres in original-pixel (y, x)

embs = {r: [] for r in a.radii}
order = np.argsort(recs[:, 0], kind="stable")
with torch.no_grad():
    for j in np.unique(recs[:, 0]).astype(int):
        idx = np.where(recs[:, 0] == j)[0]
        x = torch.from_numpy(np.asarray(f.images[j])).cuda().permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, size=(a.size, a.size), mode="bilinear", align_corners=False)
        t = vis.trunk.forward_features((x - mean) / std)
        t = t[:, vis.trunk.num_prefix_tokens:] if t.shape[1] != g * g else t     # (1, L, C)
        c = torch.from_numpy(recs[idx, 1:3].astype(np.float32)).cuda()        # (n, 2)
        d = torch.cdist(c, tok_xy)                                            # (n, L)
        for r in a.radii:
            keep = d <= r
            keep[torch.arange(len(idx)), d.argmin(1)] = True
            pooled = vis.attn_pool_contrast(t.expand(len(idx), -1, -1), attn_mask=keep)[:, 0]
            e = F.normalize(vis.ln_contrast(pooled) @ vis.proj_contrast, dim=-1)
            embs[r].append((idx, e.float().cpu()))

for r in a.radii:
    idx = np.concatenate([i for i, _ in embs[r]])
    e = torch.cat([v for _, v in embs[r]])[np.argsort(idx)]
    for name in ("names", "desc"):
        logit = (e @ protos[name][1:].T).numpy()
        pred = (logit - logit.mean(0, keepdims=True)).argmax(1) + 1
        print(f"R={r:6.0f}px  {name:5s} balanced acc {balanced_accuracy_score(y, pred):.3f}  "
              f"per-class recall {np.round(confusion_matrix(y, pred).diagonal() / np.bincount(y)[1:], 2)}")
