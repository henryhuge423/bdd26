#!/usr/bin/env python
"""Pre-registered x2 decode-menu levers on saved predictions (CPU only).

findings 2026-09-29/30: x2+du2 keeps the Dead gain but still loses bPQ/mPQ to base via
(i) interior fragmentation of large nuclei into same-class mosaics with clean outer
boundaries, and (ii) a tiny-fragment flood, ~1/3 of sampled fragments being border
slivers at the 256-px patch edge. The pre-registered decode-menu levers, applied
POST-decode on a saved pred npz:

  lever M  merge adjacent same-class pred instances whose shared boundary length is
           >= frac * min(perimeter)  (frac=0 merges every adjacent same-class pair)
  lever S  drop pred instances that touch the patch border with area < a_min

Modes:
  sweep   grid over (frac, a_min) on one pred file, scored with nucseg.metrics.light
          against the fold's GT (VAL folds for tuning — same rule as every decode
          constant in this project); writes JSON rows + the best config
  apply   write a new pred npz with the chosen constants (for eval; the inst_prob
          table is copied but not merged — retype on lever-applied preds is invalid)

    python scripts/lever_postproc.py --pred runs/x/pred_fold2_x2_du2.npz --fold 2 --mode sweep
    python scripts/lever_postproc.py --pred ... --fold 2 --mode apply --frac 0.5 --a-min 40 \
        --out runs/x/pred_fold2_x2_du2_lev.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.pannuke_eval import instance_classes  # noqa: E402


def image_geometry(inst: np.ndarray) -> dict:
    """Structural facts of one instance map that no lever config changes:
    n, areas, perimeters, border-touch flags, and the {(i, j): shared_len} dict of
    adjacent instance pairs (4-neighbour contact, i < j)."""
    n = int(inst.max())
    areas = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    border = np.zeros(n, bool)
    for line in (inst[0, :], inst[-1, :], inst[:, 0], inst[:, -1]):
        border[np.unique(line[line > 0]) - 1] = True
    per = np.zeros(n, np.int64)
    for a, b in ((inst[1:, :], inst[:-1, :]), (inst[:, 1:], inst[:, :-1])):
        m = a != b
        np.add.at(per, a[m] - 1, 1)
        np.add.at(per, b[m] - 1, 1)
    for line in (inst[0, :], inst[-1, :], inst[:, 0], inst[:, -1]):
        np.add.at(per, line[line > 0] - 1, 1)
    pairs: dict[tuple[int, int], int] = {}
    for a, b in ((inst[1:, :], inst[:-1, :]), (inst[:, 1:], inst[:, :-1])):
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        k, cnt = np.unique(lo * (n + 1) + hi, return_counts=True)
        for kk, c in zip(k.tolist(), cnt.tolist()):
            pairs[(int(kk // (n + 1)), int(kk % (n + 1)))] = c
    return {"n": n, "areas": areas, "per": per, "border": border, "pairs": pairs}


def apply_levers(inst: np.ndarray, geo: dict, frac: float | None, a_min: float,
                 cls_of: np.ndarray) -> np.ndarray:
    """lever S (border slivers, area < a_min; 0 disables) then lever M (same-class merge,
    shared >= frac*min perimeter; frac=None disables). cls_of is dense over ids 1..inst.max():
    cls_of[i-1] = majority class of instance i (1..5) as pannuke_eval computes it."""
    n = geo["n"]
    drop = geo["border"] & (geo["areas"] < a_min) if a_min > 0 else np.zeros(n, bool)
    keep = ~drop
    parent = np.arange(n + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if frac is not None:
        for (i, j), sh in geo["pairs"].items():
            if not (keep[i - 1] and keep[j - 1]) or cls_of[i - 1] != cls_of[j - 1]:
                continue
            if sh >= frac * min(geo["per"][i - 1], geo["per"][j - 1]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    lut = np.zeros(n + 1, np.int64)  # 0 = dropped
    roots: dict[int, int] = {}
    nxt = 0
    for i in range(1, n + 1):
        if not keep[i - 1]:
            continue
        r = find(i)
        if r not in roots:
            nxt += 1
            roots[r] = nxt
        lut[i] = roots[r]
    return lut[inst].astype(np.int32)


_G: dict = {}


def _one_image(j: int) -> list:
    """Worker: all lever configs for image j (geometry computed once). State is
    inherited from the parent via fork copy-on-write; GT class channels are rebuilt
    per image from inst+type (mirrors kongnet_eval._decode_stats)."""
    from nucseg.metrics import light

    inst, typ = _G["inst"], _G["typ"]
    fold = _G["fold"]
    inst_j = np.asarray(inst[j])
    gj = image_geometry(inst_j)
    cls = instance_classes(inst_j, np.asarray(typ[j]))[1]
    # x2 nearest downsampling can drop ids, so id -> row index is NOT id-1: dense-ify the
    # per-instance class array over 1..max_id (the geometry arrays are already dense over max)
    ids = np.unique(inst_j)
    cls_dense = np.zeros(int(inst_j.max()) + 1, cls.dtype)
    cls_dense[ids[ids > 0]] = cls
    gt_i, gt_t = np.asarray(fold.inst[j]), np.asarray(fold.type[j])
    gt_ch = np.stack([np.where(gt_t == c + 1, gt_i, 0) for c in range(5)], -1).astype(np.uint16)
    return [light.image_stats(gt_ch, gt_i, gt_t,
                              apply_levers(inst_j, gj, frac, a_min, cls_dense),
                              np.asarray(typ[j]))
            for frac, a_min in _G["grid"]]


_AP: dict = {}


def _apply_one(j: int) -> np.ndarray:
    """Worker for apply mode: the chosen (frac, a_min) for image j. Class array is dense over
    1..max_id (x2 nearest downsampling can drop ids — see _one_image)."""
    inst, typ = _AP["inst"], _AP["typ"]
    frac, a_min = _AP["cfg"]
    inst_j = np.asarray(inst[j])
    geo = image_geometry(inst_j)
    cls = instance_classes(inst_j, np.asarray(typ[j]))[1]
    ids = np.unique(inst_j)
    cls_dense = np.zeros(int(inst_j.max()) + 1, cls.dtype)
    cls_dense[ids[ids > 0]] = cls
    return apply_levers(inst_j, geo, frac, a_min, cls_dense)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=["sweep", "apply"], required=True)
    ap.add_argument("--fracs", type=str, nargs="+", default=["none", "0", "0.25", "0.5", "0.75"],
                    help="sweep grid; 'none' disables merging")
    ap.add_argument("--a-mins", type=float, nargs="+", default=[0, 20, 40, 80])
    ap.add_argument("--frac", type=str, default="none", help="apply mode; 'none' disables")
    ap.add_argument("--a-min", type=float, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (machinery smoke)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    def parse_frac(s: str) -> float | None:
        return None if s == "none" else float(s)

    d = np.load(args.pred)
    inst, typ = d["inst"], d["type"]
    if args.mode == "apply":
        _AP.update(inst=inst, typ=typ, cfg=(parse_frac(args.frac), args.a_min))
        n = len(inst) if args.limit is None else min(args.limit, len(inst))
        with Pool(args.workers) as pool:
            merged = pool.map(_apply_one, range(n), chunksize=8)
        out = np.stack(merged).astype(inst.dtype, copy=False)
        args.out = args.out or args.pred.with_name(args.pred.stem + "_lev.npz")
        np.savez_compressed(args.out, inst=out, type=typ,
                            **({k: d[k] for k in ("inst_img", "inst_id", "inst_prob", "tissue_prob")
                                if k in d}))
        print(f"wrote {args.out}")
        return

    from nucseg.metrics import light
    fold = PanNukeFold(args.fold)
    tissue = np.asarray(fold.tissue)
    grid = [(parse_frac(fs), a_min) for fs in args.fracs for a_min in args.a_mins]
    _G.update(inst=inst, typ=typ, fold=fold, grid=grid)
    n_img = len(inst) if args.limit is None else min(args.limit, len(inst))
    with Pool(args.workers) as pool:
        per_img = pool.map(_one_image, range(n_img), chunksize=8)
    stats = np.stack(per_img)  # (n_img, n_cfg, 6, 5)
    rows = []
    for c, (frac, a_min) in enumerate(grid):
        s = light.summarize(stats[:, c], tissue[:n_img])
        fs = args.fracs[c // len(args.a_mins)]
        rows.append({"frac": fs, "a_min": a_min, "mPQ": s["mPQ"], "bPQ": s["bPQ"]})
        print(f"frac {fs:>5} a_min {a_min:>5.0f}: mPQ {s['mPQ']:.4f} bPQ {s['bPQ']:.4f}",
              flush=True)
    best = max(rows, key=lambda r: r["mPQ"])
    print(f"best: {best}")
    out = args.pred.with_name(args.pred.stem + "_sweep.json")
    out.write_text(json.dumps({"rows": rows, "best": best,
                               "pred": str(args.pred), "fold": args.fold}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
