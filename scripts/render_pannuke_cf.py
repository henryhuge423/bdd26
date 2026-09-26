#!/usr/bin/env python
"""PanNuke-CF: counterfactual re-rendering of TEST-fold patches with FIXED GT labels (pillar C).

Fixed labels: the condition is the patch's own GT instance mask and the labels stay the GT
inst/type maps — no teacher relabelling, no layout change. Only APPEARANCE is intervened on,
factorially over two axes (both use the locked SYN-v2 recipe: base model + Reinhard):

    context axis   (paired | swapped):  UNI2-h context embedding from the patch itself, or from
                                        a different-tissue patch of the same fold
    stain axis     (self | donor):      Reinhard target = the patch's own image, or the context
                                        donor's image

Arms: control=(paired,self)  stain=(paired,donor)  ctx=(swapped,self)  tissue=(swapped,donor).
control vs real measures the pure generator-rendering gap; stain/ctx/tissue vs control measure
appearance robustness with labels held fixed. --seeds K adds generator-noise replicates of the
control arm (validity: CF deltas must exceed the seed spread).

    python scripts/render_pannuke_cf.py --fold 3 --n 500 --out runs/pixcell/cf_fold3
Outputs <out>/{arm}_{rep}/ images.npy + inst/type/tissue/base.npy in the external-data layout,
directly consumable by predict_external via ExternalSet(name, root=...) with a small meta.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nucseg.constants import TISSUES
from nucseg.data.pannuke import PanNukeFold
from nucseg.pixcell import batched_generate, load_pipeline, load_uni2h, reinhard_lab, uni_embed

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=3, help="test fold (3 for all official splits)")
p.add_argument("--n", type=int, default=500)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--arms", nargs="+", default=["control", "stain", "ctx", "tissue"])
p.add_argument("--seeds", type=int, default=1, help="generator replicates per arm")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--guidance", type=float, default=2.5)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--amp", action="store_true")
p.add_argument("--seed", type=int, default=19)
a = p.parse_args()

ARMS = {"control": (False, False), "stain": (False, True), "ctx": (True, False), "tissue": (True, True)}

rng = np.random.default_rng(a.seed)
f = PanNukeFold(a.fold)
tissue = f.tissue
uni, tf = load_uni2h()
pipe = load_pipeline()

# ---- sample patches (stratified by tissue) and one different-tissue context donor per patch
tis_list = sorted(set(tissue.tolist()))
per = max(a.n // len(tis_list), 1)
idx = []
for t in tis_list:
    cand = np.where(tissue == t)[0]
    idx += rng.choice(cand, min(per, len(cand)), replace=False).tolist()
idx = np.array(sorted(idx[:a.n]))
images = np.stack([np.asarray(f.images[i]) for i in idx])          # real patches (real-arm GT)
inst = np.stack([np.asarray(f.inst[i]) for i in idx])
typ = np.stack([np.asarray(f.type[i]) for i in idx])
tis = tissue[idx]
ctx_idx = []
for t in tis:
    other = [u for u in tis_list if u != t]
    pool = np.where(np.isin(tissue, other))[0]
    ctx_idx.append(rng.choice(pool, a.n, replace=True))
ctx_idx = np.stack(ctx_idx)                                        # (n_tissue, n) donor patch ids

embs_own = torch.cat([uni_embed(uni, tf, Image.fromarray(images[b])) for b in range(len(idx))])
donor_imgs = np.stack([np.asarray(f.images[ctx_idx[list(tis_list).index(tis[b]), b]]) for b in range(len(idx))])
embs_don = torch.cat([uni_embed(uni, tf, Image.fromarray(donor_imgs[b])) for b in range(len(idx))])
masks = [np.where(inst[b] > 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1) for b in range(len(idx))]
print(f"[cf] {len(idx)} patches, {len(tis_list)} tissues; embeddings ready")

a.out.mkdir(parents=True, exist_ok=True)
for arm in a.arms:
    sw_ctx, sw_stain = ARMS[arm]
    for rep in range(a.seeds):
        out = a.out / (arm if a.seeds == 1 else f"{arm}_rep{rep}")
        if (out / "images.npy").exists():
            print(f"[cf] {out.name} exists, skip")
            continue
        embs = embs_don if sw_ctx else embs_own
        gens = []
        for s in range(0, len(idx), a.batch):
            seeds = [a.seed + 1000 * rep + int(i) for i in idx[s:s + a.batch]]
            if a.amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = batched_generate(pipe, embs[s:s + a.batch].unsqueeze(1), masks[s:s + a.batch],
                                         a.steps, a.guidance, seeds)
            else:
                o = batched_generate(pipe, embs[s:s + a.batch].unsqueeze(1), masks[s:s + a.batch],
                                     a.steps, a.guidance, seeds)
            tgt = donor_imgs[s:s + a.batch] if sw_stain else images[s:s + a.batch]
            gens.append(np.stack([reinhard_lab(g, t) for g, t in zip(o, tgt)]))
            if (s // a.batch) % 10 == 0:
                print(f"[cf] {out.name} {s}/{len(idx)}", flush=True)
        gens = np.concatenate(gens)
        np.save(out / "images.npy", gens)
        np.save(out / "inst.npy", inst.astype(np.uint16))
        np.save(out / "type.npy", typ.astype(np.uint8))
        np.save(out / "tissue.npy", tis)
        np.save(out / "base.npy", idx.astype(np.int32))
        # real reference arm (same patches, real images) + external-layout meta for predict_external
        real_dir = a.out / "real"
        if not (real_dir / "images.npy").exists():
            np.save(real_dir / "images.npy", images)
            np.save(real_dir / "inst.npy", inst.astype(np.uint16))
            np.save(real_dir / "type.npy", typ.astype(np.uint8))
            np.save(real_dir / "tissue.npy", tis)
            np.save(real_dir / "base.npy", idx.astype(np.int32))
        for d in [out, real_dir]:
            if not (d / "meta.json").exists():
                (d / "meta.json").write_text(json.dumps({
                    "dataset": f"pannuke_cf_{d.name}", "n_patches": int(len(idx)),
                    "tissue_names": TISSUES, "arm": d.name,
                    "notes": "PanNuke-CF counterfactual arm; labels = original fold GT"},
                    indent=2))
        print(f"[cf] {out.name}: {gens.shape} saved")
print("[cf] all arms done")
