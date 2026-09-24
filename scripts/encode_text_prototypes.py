#!/usr/bin/env python
"""Encode the nucleus prompt bank with CONCH and (optionally) probe it zero-shot on nucleus crops.

    python scripts/encode_text_prototypes.py --out weights/text_protos/conch_v1.pt --probe-fold 1
The probe crops a 64x64 (16 um) window around GT nuclei of a TRAINING fold, embeds it with the CONCH
image tower, and reports balanced zero-shot accuracy for each prototype flavour, plus a linear-probe
reference (5-fold CV logistic regression) -- i.e. how much class information the text space carries.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

from nucseg.cellvit.data import tissue_ids
from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.text.prototypes import build_prototypes, load_conch

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=Path("weights/text_protos/conch_v1.pt"))
p.add_argument("--probe-fold", type=int, default=0, help="0 = no probe")
p.add_argument("--per-class", type=int, default=400)
p.add_argument("--crop", type=int, default=64)
p.add_argument("--calibrate", action="store_true", help="subtract the per-class mean logit (transductive)")
a = p.parse_args()

protos = build_prototypes()
a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(protos, a.out)
print("saved", a.out, {k: tuple(v.shape) for k, v in protos.items() if torch.is_tensor(v)})
sim = protos["desc"] @ protos["desc"].T
print("desc prototype cosine sim (bg, " + ", ".join(CLASS_NAMES) + "):\n", np.round(sim.numpy(), 3))
if not a.probe_fold:
    raise SystemExit

f = PanNukeFold(a.probe_fold)
rng = np.random.default_rng(0)
recs = []  # (img idx, inst id, class)
for j in range(len(f)):
    inst, typ = np.asarray(f.inst[j]), np.asarray(f.type[j])
    for iid in np.unique(inst)[1:]:
        recs.append((j, iid, int(np.bincount(typ[inst == iid]).argmax())))
recs = np.array(recs)
sel = np.concatenate([rng.permutation(np.where(recs[:, 2] == c)[0])[:a.per_class] for c in range(1, 6)])
recs = recs[sel]
print("probe nuclei per class", np.bincount(recs[:, 2], minlength=6)[1:])

model, preprocess, _ = load_conch()
h = a.crop // 2
crops = []
for j, iid, _ in recs:
    img = np.pad(np.asarray(f.images[j]), ((h, h), (h, h), (0, 0)), mode="reflect")
    cy, cx = ndimage.center_of_mass(np.asarray(f.inst[j]) == iid)
    cy, cx = int(round(cy)) + h, int(round(cx)) + h
    crops.append(preprocess(Image.fromarray(img[cy - h:cy + h, cx - h:cx + h])))
with torch.no_grad():
    emb = torch.cat([model.encode_image(torch.stack(crops[i:i + 128]).cuda()) for i in range(0, len(crops), 128)])
emb = emb.float().cpu()
y = recs[:, 2]
tid = tissue_ids(np.asarray(f.tissue)[recs[:, 0]])

def report(name, logits):
    logits = logits - logits.mean(0, keepdims=True) if a.calibrate else logits
    pred = logits.argmax(1) + 1
    print(f"{name:28s} balanced acc {balanced_accuracy_score(y, pred):.3f}")
    return pred

report("zero-shot names", (emb @ protos["names"][1:].T).numpy())
pd = report("zero-shot descriptions", (emb @ protos["desc"][1:].T).numpy())
print(confusion_matrix(y, pd))
report("zero-shot desc (max over K)", (emb @ protos["desc_all"][1:].flatten(0, 1).T).view(len(y), 5, -1).amax(-1).numpy())
pt = report("zero-shot tissue-conditioned", torch.einsum("nd,ncd->nc", emb, protos["tissue"][tid, 1:]).numpy())
print(confusion_matrix(y, pt))
lp = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), emb.numpy(), y, cv=5)
print(f"{'linear probe (5-fold CV)':28s} balanced acc {balanced_accuracy_score(y, lp):.3f}")
print(confusion_matrix(y, lp))
