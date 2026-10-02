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
  apply   write a new pred npz with the chosen constants (for eval); stale per-instance
          probability tables are omitted after merging/renumbering.

    python scripts/lever_postproc.py --pred runs/x/pred_fold2_x2_du2.npz --fold 2 --mode sweep
    python scripts/lever_postproc.py --pred ... --fold 2 --mode apply --frac 0.5 --a-min 40 \
        --out runs/x/pred_fold2_x2_du2_lev.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.pannuke_eval import instance_classes  # noqa: E402


def image_geometry(inst: np.ndarray) -> dict:
    """Structural facts of one instance map that no lever config changes:
    n, areas, perimeters, border-touch flags, and the {(i, j): shared_len} dict of
    adjacent instance pairs (4-neighbour contact, i < j)."""
    inst = np.asarray(inst)
    if inst.ndim != 2 or not inst.size or not np.issubdtype(inst.dtype, np.integer):
        raise ValueError("instance map must be a nonempty 2-D integer array")
    if inst.min() < 0 or inst.max() > np.iinfo(np.int32).max:
        raise ValueError("instance IDs must be nonnegative int32 values")
    # Signed arithmetic is required before subtracting one from uint16 IDs.
    inst = inst.astype(np.int64, copy=False)
    n = int(inst.max())
    areas = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    border = np.zeros(n, bool)
    for line in (inst[0, :], inst[-1, :], inst[:, 0], inst[:, -1]):
        border[np.unique(line[line > 0]) - 1] = True
    per = np.zeros(n, np.int64)
    for a, b in ((inst[1:, :], inst[:-1, :]), (inst[:, 1:], inst[:, :-1])):
        m = a != b
        np.add.at(per, a[m & (a > 0)] - 1, 1)
        np.add.at(per, b[m & (b > 0)] - 1, 1)
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
            pair = (int(kk // (n + 1)), int(kk % (n + 1)))
            pairs[pair] = pairs.get(pair, 0) + c
    return {"n": n, "areas": areas, "per": per, "border": border, "pairs": pairs}


def apply_levers(inst: np.ndarray, geo: dict, frac: float | None, a_min: float,
                 cls_of: np.ndarray) -> np.ndarray:
    """lever S (border slivers, area < a_min; 0 disables) then lever M (same-class merge,
    shared >= frac*min perimeter; frac=None disables). cls_of has length max_id+1:
    cls_of[i] = majority class of instance i (0..5), with background slot zero."""
    n = geo["n"]
    cls_of = np.asarray(cls_of)
    if cls_of.shape != (n + 1,) or not np.issubdtype(cls_of.dtype, np.integer):
        raise ValueError("class LUT must be integer, length max_id+1, indexed by instance ID")
    if cls_of[0] != 0 or np.any((cls_of < 0) | (cls_of > 5)):
        raise ValueError("class LUT must contain classes 0..5 and background slot zero")
    if (frac is not None and (not np.isfinite(frac) or frac < 0)) or not np.isfinite(a_min) or a_min < 0:
        raise ValueError("lever thresholds must be finite and nonnegative")
    drop = geo["border"] & (geo["areas"] < a_min) if a_min > 0 else np.zeros(n, bool)
    keep = ~drop & (geo["areas"] > 0)
    parent = np.arange(n + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if frac is not None:
        for (i, j), sh in geo["pairs"].items():
            if not (keep[i - 1] and keep[j - 1]) or cls_of[i] != cls_of[j]:
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
    inherited via fork copy-on-write, including the original overlapping GT channels."""
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
    gt_ch = _G["gt_channels"][j]
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


def file_provenance(path: Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def code_provenance() -> dict:
    paths = [Path(__file__), ROOT / "scripts/run_p0_controls.py", ROOT / "src/nucseg/constants.py",
             ROOT / "src/nucseg/data/pannuke.py", ROOT / "src/nucseg/metrics/light.py",
             ROOT / "src/nucseg/metrics/pannuke_eval.py", ROOT / "src/nucseg/metrics/instance.py",
             ROOT / "src/nucseg/metrics/errors.py"]
    files = {str(p.relative_to(ROOT)): file_provenance(p)["sha256"] for p in paths if p.exists()}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"git_head": result.stdout.strip() if result.returncode == 0 else None, "files": files}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prediction_payload(original, transformed: np.ndarray) -> dict:
    n = len(transformed)
    typ = np.asarray(original["type"][:n]).copy()
    if typ.shape != transformed.shape:
        raise ValueError("transformed instances and type map must have identical shapes")
    typ[transformed == 0] = 0
    payload = {"inst": transformed.astype(np.int32, copy=False), "type": typ}
    for key in ("tissue_prob", "image_indices"):
        if key in original:
            values = np.asarray(original[key])
            if values.ndim == 0 or len(values) != len(original["type"]):
                raise ValueError(f"{key} is not aligned to the prediction images")
            payload[key] = values[:n]
    return payload


def transform_predictions(inst, typ, frac, a_min, workers=2):
    if inst.shape != typ.shape or inst.ndim != 3 or len(inst) == 0 or workers < 1:
        raise ValueError("predictions must be matching nonempty N,H,W arrays; workers >= 1")
    _AP.update(inst=inst, typ=typ, cfg=(frac, a_min))
    out = np.empty(inst.shape, np.int32)
    try:
        if workers == 1:
            for i in range(len(inst)):
                out[i] = _apply_one(i)
        else:
            with Pool(workers) as pool:
                for i, image in enumerate(pool.imap(_apply_one, range(len(inst)), chunksize=8)):
                    out[i] = image
    finally:
        _AP.clear()
    return out


def sweep_predictions(inst, typ, fold, grid, workers=2):
    from nucseg.metrics import light
    if inst.shape != typ.shape or inst.shape != fold.inst.shape or workers < 1:
        raise ValueError("sweep requires full-fold aligned instance/type arrays; workers >= 1")
    # Materialize once in the parent; workers must not independently decompress GT.
    gt_channels = np.asarray(fold.gt_channels)
    if gt_channels.shape != (*inst.shape, 5):
        raise ValueError("official GT channels must align with the full-fold predictions")
    _G.update(inst=inst, typ=typ, fold=fold, grid=grid, gt_channels=gt_channels)
    try:
        if workers == 1:
            per_img = [_one_image(i) for i in range(len(inst))]
        else:
            with Pool(workers) as pool:
                per_img = pool.map(_one_image, range(len(inst)), chunksize=8)
    finally:
        _G.clear()
    stats = np.stack(per_img)
    rows = []
    for c, (frac, a_min) in enumerate(grid):
        score = light.summarize(stats[:, c], np.asarray(fold.tissue))
        rows.append({"frac": frac, "a_min": a_min, "mPQ": score["mPQ"], "bPQ": score["bPQ"]})
    return rows


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
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        ap.error("workers and limit must be positive")

    def parse_frac(s: str) -> float | None:
        return None if s == "none" else float(s)

    suffix = "_lev.npz" if args.mode == "apply" else "_sweep.json"
    out = args.out or args.pred.with_name(args.pred.stem + suffix)
    sidecar = out.with_suffix(out.suffix + ".json")
    for target in (out, sidecar) if args.mode == "apply" else (out,):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    metadata = {"input": file_provenance(args.pred), "code": code_provenance(),
                "fold": args.fold, "mode": args.mode, "limit": args.limit}
    config = args.pred.parent / "config.json"
    if config.exists():
        cfg = json.loads(config.read_text())
        metadata.update(actual_seed=cfg.get("seed"), split=cfg.get("split"),
                        run_config=file_provenance(config))
    else:
        metadata.update(actual_seed=None, split=None)
    with np.load(args.pred) as d:
        inst, typ = d["inst"], d["type"]
        n = len(inst) if args.limit is None else min(args.limit, len(inst))
        if args.mode == "apply":
            frac = parse_frac(args.frac)
            transformed = transform_predictions(inst[:n], typ[:n], frac, args.a_min, args.workers)
            payload = prediction_payload(d, transformed)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as stream:
                np.savez_compressed(stream, **payload)
            metadata.update(frac=frac, a_min=args.a_min, n_images=n,
                            instance_probability_tables="omitted after transform")
            write_json(sidecar, metadata)
            print(f"wrote {out}")
            return
    from types import SimpleNamespace
    from run_p0_controls import select_config
    f = PanNukeFold(args.fold)
    fold = SimpleNamespace(inst=f.inst[:n], type=f.type[:n], tissue=f.tissue[:n],
                           gt_channels=f.gt_channels[:n])
    grid = [(parse_frac(fs), a_min) for fs in args.fracs for a_min in args.a_mins]
    rows = sweep_predictions(inst[:n], typ[:n], fold, grid, args.workers)
    best = select_config(rows)
    metadata.update(rows=rows, best=best, n_images=n, selection_bpq_margin=.002)
    write_json(out, metadata)
    print(f"best: {best}\n-> {out}")


if __name__ == "__main__":
    main()
