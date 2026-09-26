#!/usr/bin/env python
"""Convert external cross-domain datasets to the compact PanNuke-like layout (pillar C).

    python scripts/prepare_external.py --dataset conic --src /data/raw_external/conic --out data/external/conic
    python scripts/prepare_external.py --dataset monusac --src /data/raw_external/monusac --split test

Output per dataset dir: images.npy u8 (N,256,256,3), inst.npy u16, type.npy u8 (PanNuke class
ids 1..5), tissue.npy i16 (dataset-specific pseudo-tissues, names in meta.json), meta.json.

conic  : HF MedOtter/CoNIC2022 parquets. 20x (0.5 um/px) -> upsample 2x to 40x scale, then
         quad-split into 256 tiles (the training input size; border-cut nuclei are inherent
         to the already-tiled CoNIC patches). The 112 patches with source == 'pannuke' are
         EXCLUDED (overlap with our training data). Classes: Neutrophil/Lymphocyte/Plasma/
         Eosinophil -> Inflammatory, Epithelial -> Epithelial, Connective -> Connective;
         Neoplastic and Dead do not occur.
monusac: HF RationAI/MoNuSAC parquets (whole images at 40x). Official test split by default.
         RGBA -> RGB (alpha verified fully opaque), Ambiguous instances (category 0) dropped,
         Epithelial -> Epithelial, Lymphocyte/Macrophage/Neutrophil -> Inflammatory, tiled
         256x256 non-overlapping; only full tiles with >= 1 kept nucleus are written.

Memory-safe two passes (count, then write through a memmap) so LM1 / ugrad can run it too.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from nucseg.constants import CLASS_NAMES, NUM_CLASSES

# source-class -> PanNuke class id (1..5 in CLASS_NAMES order); 0 = drop the instance.
CONIC_MAP = {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3}  # neut, epi, lymph, plasma, eos, conn
CONIC_NAMES = {1: "Neutrophil", 2: "Epithelial", 3: "Lymphocyte", 4: "Plasma", 5: "Eosinophil", 6: "Connective"}
MONUSAC_MAP = {0: 0, 1: 5, 2: 2, 3: 2, 4: 2}  # ambiguous, epi, lymph, macro, neut
MONUSAC_NAMES = {0: "Ambiguous", 1: "Epithelial", 2: "Lymphocyte", 3: "Macrophage", 4: "Neutrophil"}
MONUSAC_TISSUES = ["Breast", "Kidney", "Lung", "Prostate"]


def _pil(rec) -> Image.Image:
    return Image.open(io.BytesIO(rec["bytes"]))


def conic_tiles(src: Path):
    """Yield (image tile, inst tile, type tile, tissue name str, provenance str, counter)."""
    import glob
    shards = sorted(glob.glob(str(src / "*.parquet")))
    assert shards, f"no parquet shards under {src}"
    n_src = collections.Counter()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pydict()
            for i in range(len(t["image"])):
                src_name = t["source"][i]
                n_src[src_name] += 1
                if src_name == "pannuke":  # leakage guard: skip PanNuke-derived patches
                    continue
                img = _pil(t["image"][i]).convert("RGB").resize((512, 512), Image.BILINEAR)
                inst = _pil(t["inst_map"][i]).resize((512, 512), Image.NEAREST)
                cls = _pil(t["class_map"][i]).resize((512, 512), Image.NEAREST)
                im, ins, cl = np.asarray(img), np.asarray(inst), np.asarray(cls)
                typ = np.zeros_like(cl)
                for k, v in CONIC_MAP.items():
                    typ[cl == k] = v
                typ[ins == 0] = 0
                for r in (0, 256):
                    for c in (0, 256):
                        prov = f"{t['patch_info'][i]}_q{r // 256}{c // 256}"
                        yield im[r:r + 256, c:c + 256], ins[r:r + 256, c:c + 256], \
                            typ[r:r + 256, c:c + 256], "Colon", prov, dict(n_src)


def monusac_tiles(src: Path, split: str):
    shards = [str(src / f"{split}.parquet")]
    assert Path(shards[0]).exists(), f"missing {shards[0]}"
    n_cat = collections.Counter()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pydict()
            for i in range(len(t["image"])):
                img = np.asarray(_pil(t["image"][i]).convert("RGB"))
                h, w = img.shape[:2]
                ins = np.zeros((h, w), np.uint16)
                typ = np.zeros((h, w), np.uint8)
                for k, (rec, cat) in enumerate(zip(t["instances"][i], t["categories"][i]), 1):
                    n_cat[cat] += 1
                    m = MONUSAC_MAP[cat]
                    if not m:
                        continue
                    mask = np.asarray(_pil(rec)) > 0
                    ins[mask], typ[mask] = k, m
                for r in range(0, h - 255, 256):
                    for c in range(0, w - 255, 256):
                        sub_i, sub_t = ins[r:r + 256, c:c + 256], typ[r:r + 256, c:c + 256]
                        if not (sub_i > 0).any():
                            continue
                        yield img[r:r + 256, c:c + 256], sub_i, sub_t, MONUSAC_TISSUES[int(t["tissue"][i])], \
                            f"{t['patient'][i]}_{r}_{c}", dict(n_cat)


DATASETS = {"conic": conic_tiles, "monusac": monusac_tiles}


def convert(name: str, src: Path, out: Path, split: str):
    gen = DATASETS[name](src, split) if name == "monusac" else DATASETS[name](src)
    n, last_prov = 0, None
    for *_, prov, _ in gen:  # pass 1: count (also decodes; ~minutes)
        n += 1
        last_prov = prov
    print(f"[pass 1] {name}: {n} tiles")

    out.mkdir(parents=True, exist_ok=True)
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256, 3))
    inst = np.lib.format.open_memmap(out / "inst.npy", mode="w+", dtype=np.uint16, shape=(n, 256, 256))
    typ = np.lib.format.open_memmap(out / "type.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256))
    tis, provs = np.empty(n, "<U16"), []
    gen = DATASETS[name](src, split) if name == "monusac" else DATASETS[name](src)
    for i, (im, ins, ty, ti, prov, _) in enumerate(gen):  # pass 2: write
        imgs[i], inst[i], typ[i], tis[i] = im, ins, ty, ti
        provs.append(prov)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}")
    for a in (imgs, inst, typ):
        a.flush()
    np.save(out / "tissue.npy", tis)
    # per-class nucleus counts: majority class per instance, vectorised per tile
    class_counts = np.zeros(NUM_CLASSES, np.int64)
    for i in range(n):
        ins, ty = np.asarray(inst[i]), np.asarray(typ[i])
        fg = ins > 0
        comb = ins[fg].astype(np.int64) * 8 + ty[fg]
        u, cnt = np.unique(comb, return_counts=True)
        lab_m, cls_m = u // 8, u % 8
        order = np.lexsort((-cnt, lab_m))
        lab_m, cls_m = lab_m[order], cls_m[order]
        first = np.ones(len(lab_m), bool)
        first[1:] = lab_m[1:] != lab_m[:-1]
        class_counts += np.bincount(cls_m[first] - 1, minlength=NUM_CLASSES)
    tissue_names = MONUSAC_TISSUES if name == "monusac" else ["Colon"]
    meta = {
        "dataset": name, "split": split if name == "monusac" else "train",
        "n_patches": int(n),
        "nuclei_per_class": {k: int(v) for k, v in zip(CLASS_NAMES, class_counts)},
        "tissue_names": tissue_names,
        "tissue_counts": {t: int((tis == t).sum()) for t in np.unique(tis)},
        "pannuke_classes_absent": [CLASS_NAMES[c] for c in range(NUM_CLASSES) if class_counts[c] == 0],
        "class_mapping": {"conic": {CONIC_NAMES[k]: CLASS_NAMES[v - 1] for k, v in CONIC_MAP.items()},
                          "monusac": {MONUSAC_NAMES[k]: (CLASS_NAMES[v - 1] if v else "dropped")
                                      for k, v in MONUSAC_MAP.items()}}[name],
        "notes": {"conic": "source==pannuke patches excluded (leakage); 20x upsampled 2x, quad-split 256",
                  "monusac": "official test split; RGBA->RGB; Ambiguous dropped; full 256 tiles with >=1 nucleus"}[name],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "provenance.txt").write_text("\n".join(provs))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--src", type=Path, required=True, help="dir with the downloaded parquet shards")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test", help="monusac only")
    a = p.parse_args()
    convert(a.dataset, a.src, a.out, a.split)
