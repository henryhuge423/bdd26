#!/usr/bin/env python
"""Convert PUMA (Zenodo 15050523, melanoma 40x ROI training set) to the PanNuke-like layout.

    python scripts/prepare_puma.py --src data/external/puma_raw --out data/external/puma
    # expects <src>/01_training_dataset_tif_ROIs/*.tif + 01_training_dataset_geojson_nuclei/*.geojson

Output matches prepare_external.py: images.npy u8 (N,256,256,3), inst.npy u16, type.npy u8
(PanNuke ids), tissue.npy, meta.json. 1024x1024 ROIs are tiled 4x4 (256, non-overlapping);
tiles with >=1 kept nucleus are written. PUMA classes map:

    nuclei_tumor -> Neoplastic          nuclei_apoptosis -> Dead  (the point of this set)
    nuclei_lymphocyte / histiocyte / melanophage / plasma_cell / neutrophil -> Inflammatory
    nuclei_stroma / endothelium -> Connective          nuclei_epithelium -> Epithelial

Caveat (documented in meta): PUMA labels were HoVer-Net-initialised and pathologist-corrected.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nucseg.constants import CLASS_NAMES, NUM_CLASSES

PUMA_MAP = {
    "nuclei_tumor": 1,
    "nuclei_lymphocyte": 2, "nuclei_histiocyte": 2, "nuclei_melanophage": 2,
    "nuclei_plasma_cell": 2, "nuclei_neutrophil": 2,
    "nuclei_stroma": 3, "nuclei_endothelium": 3,
    "nuclei_apoptosis": 4,
    "nuclei_epithelium": 5,
}


def rasterize(geo: dict, h: int, w: int):
    """GeoJSON features -> (inst u16, type u8); later polygons overwrite earlier on overlap."""
    inst, typ = np.zeros((h, w), np.uint16), np.zeros((h, w), np.uint8)
    for k, f in enumerate(geo["features"], 1):
        cls = PUMA_MAP.get(f["properties"].get("classification", {}).get("name"))
        if cls is None:
            continue
        polys = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon" \
            else f["geometry"]["coordinates"]
        pts = [np.round(np.asarray(p[0], np.float32)).astype(np.int32) for p in polys]
        cv2.fillPoly(inst, pts, k)
        cv2.fillPoly(typ, pts, cls)
    return inst, typ


def puma_tiles(src: Path):
    tifs = sorted((src / "01_training_dataset_tif_ROIs").glob("*.tif"))
    assert tifs, f"no ROIs under {src}"
    n_cls = collections.Counter()
    for tif in tifs:
        stem = tif.name[:-4]
        geo = json.loads((src / "01_training_dataset_geojson_nuclei" / f"{stem}_nuclei.geojson").read_text())
        img = np.asarray(Image.open(tif).convert("RGB"))
        h, w = img.shape[:2]
        inst, typ = rasterize(geo, h, w)
        n_cls.update(f["properties"].get("classification", {}).get("name", "NONE")
                     for f in geo["features"])
        tis = "Melanoma-metastatic" if "metastatic" in stem else "Melanoma-primary"
        for r in range(0, h - 255, 256):
            for c in range(0, w - 255, 256):
                si, st = inst[r:r + 256, c:c + 256], typ[r:r + 256, c:c + 256]
                if not (si > 0).any():
                    continue
                yield img[r:r + 256, c:c + 256], si, st, tis, f"{stem}_{r}_{c}", dict(n_cls)


def convert(src: Path, out: Path):
    n, areas = 0, []
    for im, si, st, *_ in puma_tiles(src):  # pass 1: count + nucleus-scale sanity stat
        n += 1
        _, cnt = np.unique(si[si > 0], return_counts=True)
        areas.append(cnt.astype(np.float64))
    print(f"[pass 1] puma: {n} tiles")
    all_areas = np.concatenate(areas) if areas else np.array([0.0])
    print(f"  median nucleus area {np.median(all_areas):.0f} px "
          f"(PanNuke fold-3 measured median 464, IQR 251-682; tif tags say 0.226 um/px = 40x)")

    out.mkdir(parents=True, exist_ok=True)
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256, 3))
    inst = np.lib.format.open_memmap(out / "inst.npy", mode="w+", dtype=np.uint16, shape=(n, 256, 256))
    typ = np.lib.format.open_memmap(out / "type.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256))
    tis, provs = np.empty(n, "<U20"), []
    for i, (im, si, st, ti, prov, _) in enumerate(puma_tiles(src)):
        imgs[i], inst[i], typ[i], tis[i] = im, si, st, ti
        provs.append(prov)
    for a in (imgs, inst, typ):
        a.flush()
    np.save(out / "tissue.npy", tis)
    class_counts = np.zeros(NUM_CLASSES, np.int64)
    for i in range(n):
        si, st = np.asarray(inst[i]), np.asarray(typ[i])
        fg = si > 0
        comb = si[fg].astype(np.int64) * 8 + st[fg]
        u, cnt = np.unique(comb, return_counts=True)
        lab_m, cls_m = u // 8, u % 8
        order = np.lexsort((-cnt, lab_m))
        lab_m, cls_m = lab_m[order], cls_m[order]
        first = np.ones(len(lab_m), bool)
        first[1:] = lab_m[1:] != lab_m[:-1]
        class_counts += np.bincount(cls_m[first] - 1, minlength=NUM_CLASSES)
    tissues = ["Melanoma-primary", "Melanoma-metastatic"]
    meta = {
        "dataset": "puma", "split": "train (zero-shot: never seen in PanNuke training)",
        "n_patches": int(n),
        "nuclei_per_class": {k: int(v) for k, v in zip(CLASS_NAMES, class_counts)},
        "tissue_names": tissues,
        "tissue_counts": {t: int((tis == t).sum()) for t in np.unique(tis)},
        "pannuke_classes_absent": [CLASS_NAMES[c] for c in range(NUM_CLASSES) if class_counts[c] == 0],
        "class_mapping": {k: CLASS_NAMES[v - 1] for k, v in PUMA_MAP.items()},
        "notes": "Zenodo 15050523 ROI training set; RGBA->RGB; 1024 ROIs tiled 4x4 256px; "
                 "labels HoVer-Net-initialised and pathologist-corrected (per challenge docs)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "provenance.txt").write_text("\n".join(provs))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    convert(a.src, a.out)
