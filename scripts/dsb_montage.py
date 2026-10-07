#!/usr/bin/env python
"""Stratified montage of dead-expert candidates for blind visual review (DSB spec §8).

    python scripts/dsb_montage.py --run runs/dsb_dev_20261007/dsb_seed19 --fold 2 \
        --k 48 --out runs/dsb_dev_20261007/dsb_seed19/montage_fold2.png

Rows are image crops with base instances outlined green and dead-expert candidates outlined
orange; the reviewer is told the layout but never which arm produced a grid (blind review).
Sampling is stratified by tissue x border x area-bin with seeded, proportional allocation.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from nucseg.data.pannuke import PanNukeFold

GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)


def outline(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Color the border ring of a boolean mask on an RGB uint8 copy of img."""
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def build_grid(patches, cols: int) -> np.ndarray:
    """Deterministic row-major grid of equally shaped HxWx3 patches; trailing cells are blank."""
    patches = [np.asarray(p) for p in patches]
    h, w, c = patches[0].shape
    rows = math.ceil(len(patches) / cols)
    grid = np.zeros((rows * h, cols * w, c), np.uint8)
    for i, p in enumerate(patches):
        r, cc = divmod(i, cols)
        grid[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = p
    return grid


def stratify(records, keys, k: int, seed: int) -> list[int]:
    """Exactly k unique record indices: every non-empty stratum gets one pick first (the rare
    cases are the point of the review), the rest goes proportional (largest-remainder), seeded."""
    if k > len(records):
        raise ValueError("k exceeds the number of records")
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        strata.setdefault(tuple(r[key] for key in keys), []).append(i)
    quota = {s: len(v) * k / len(records) for s, v in strata.items()}
    alloc = {s: min(1, len(v)) for s, v in strata.items()}       # coverage first
    rem = k - sum(alloc.values())
    order = sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), str(s)))
    while rem > 0:
        progressed = False
        for s in order:
            if rem == 0:
                break
            if alloc[s] < len(strata[s]):
                alloc[s] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(strata):
        members = sorted(strata[s])
        take = min(alloc[s], len(members))
        out.extend(int(i) for i in rng.choice(members, size=take, replace=False))
    # defensive top-up (unreachable when alloc respects stratum sizes)
    for i in range(len(records)):
        if len(out) >= k:
            break
        if i not in out:
            out.append(i)
    return sorted(out[:k])


def candidate_records(base_inst, dead_inst, tissues):
    """One record per dead candidate: image, id, area, border flag, tissue, area bin."""
    records = []
    for i in range(len(dead_inst)):
        for cid in np.unique(dead_inst[i]):
            if cid == 0:
                continue
            mask = dead_inst[i] == cid
            area = int(mask.sum())
            records.append({
                "img": i, "id": int(cid), "area": area,
                "border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                "overlaps_base": bool((base_inst[i][mask] > 0).any()),
                "tissue": str(tissues[i]),
                "area_bin": 0 if area < 30 else (1 if area < 60 else 2),
            })
    return records


def crop_around(img, inst, dead, cid, half=64):
    """128x128 crop centered on candidate cid: base outlines green, the candidate orange."""
    mask = dead == cid
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = img.shape[:2]
    y0, x0 = max(0, cy - half), max(0, cx - half)
    y1, x1 = min(h, cy + half), min(w, cx + half)
    crop = outline(outline(img[y0:y1, x0:x1], (inst[y0:y1, x0:x1] > 0), GREEN),
                   mask[y0:y1, x0:x1], ORANGE)
    return crop


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--tag", default=None, help="prediction tag (default: fold{k})")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    tag = a.tag or f"fold{a.fold}"
    base = np.load(a.run / f"pred_{tag}.npz")
    dead = np.load(a.run / f"pred_{tag}_dead.npz")["dead_inst"]
    f = PanNukeFold(a.fold)
    records = candidate_records(base["inst"], dead, f.tissue)
    if not records:
        raise SystemExit("no dead candidates to montage")
    chosen = stratify(records, keys=("tissue", "border", "area_bin"), k=min(a.k, len(records)),
                      seed=a.seed)
    patches = [crop_around(np.asarray(f.images[r["img"]]), base["inst"][r["img"]],
                           dead[r["img"]], r["id"]) for r in (records[i] for i in chosen)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), build_grid(patches, a.cols))
    (a.out.with_suffix(".records.json")).write_text(
        __import__("json").dumps([records[i] for i in chosen], indent=2))
    print(f"{len(chosen)} candidates -> {a.out}")


if __name__ == "__main__":
    main()
