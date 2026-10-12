"""E3 fold1 jury support (spec 2026-10-11 §2): per-candidate detector-consensus counts.

Support = number of jury members with any instance at IoU >= thr against the candidate
mask (E1-C convention), excluding the candidate's own source run (its generating x2 arm).
Jury seats and candidate masks are materialized ONCE per npz before any per-image loops.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nucseg.metrics.instance import overlap

VOTE_THR = 0.3


def jury_paths_fold1(round_root: Path, hn_fold1: Path | None) -> list[dict]:
    """Six split2 val-fold seats (all trained on fold2) plus an optional HoVer-Net seat."""
    out = []
    for seed in (19, 1, 2):
        for scale, arm in (("x1", "base"), ("x2", "x2")):
            out.append({
                "name": f"cellvit_s2_e{seed}_{scale}",
                "path": str(round_root / "p0" / f"split2_seed{seed}" / "val" / arm
                         / "identity" / "pred.npz"),
                "source_pair": f"split2_seed{seed}" if scale == "x2" else None})
    if hn_fold1 is not None:
        out.append({"name": "hovernet_s2", "path": str(hn_fold1),
                    "source_pair": None})
    return out


def _source_inst(round_root: Path, pair: str) -> np.ndarray:
    path = round_root / "p0" / pair / "val" / "x2" / "identity" / "pred.npz"
    with np.load(path) as z:
        return z["inst"]


def support_matrix(rec: dict, round_root: Path, jury: list[dict],
                   thr: float = VOTE_THR) -> tuple[np.ndarray, np.ndarray]:
    """(support (N,J) uint8, excluded (N,J) bool) over the candidate rows in `rec`.

    `rec` needs pair/image/id columns. The candidate mask is taken from its source pair's
    frozen x2 validation prediction; a seat whose `source_pair` equals the candidate's pair
    is excluded (support forced to 0, flagged in `excluded`).
    """
    pairs = np.asarray(rec["pair"])
    images = np.asarray(rec["image"])
    ids = np.asarray(rec["id"])
    n, j = len(pairs), len(jury)
    mat = np.zeros((n, j), np.uint8)
    excl = np.zeros((n, j), bool)

    sources = {p: _source_inst(round_root, p) for p in dict.fromkeys(pairs.tolist())}
    seats = []
    for seat in jury:
        with np.load(seat["path"]) as z:
            seats.append(z["inst"])

    order = np.arange(n)
    for img in np.unique(images):
        idxs = order[images == img]
        for k in idxs:
            src = sources[pairs[k]]
            mask = src[img] == ids[k]
            if not np.any(mask):
                raise ValueError(f"candidate id {ids[k]} not found in {pairs[k]} image {img}")
            for s, seat in enumerate(jury):
                if seat.get("source_pair") == pairs[k]:
                    excl[k, s] = True
                    continue
                inst = seats[s][img]
                if not np.any(inst):
                    continue
                ov = overlap(mask.astype(np.int32), inst.astype(np.int32))
                if ov.iou.size and float(ov.iou.max()) >= thr:
                    mat[k, s] = 1
    return mat, excl


def effective_denominator(excluded: np.ndarray) -> np.ndarray:
    """Votes available per candidate: total seats minus the excluded source seat."""
    return excluded.shape[1] - excluded.sum(1)
