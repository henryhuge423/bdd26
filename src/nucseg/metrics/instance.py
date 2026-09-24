"""Instance-level metrics: PQ, AJI, AJI+, centroid detection matching.

Fast contingency-table implementations of the definitions in PanNuke-metrics / HoVer-Net
`stats_utils.py` (IoU = inter / (|t| + |p| - inter); PQ matching at IoU > 0.5).
`tests/test_metrics.py` checks them against the vendored official functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def relabel(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Map arbitrary instance ids to contiguous 1..n (0 stays background)."""
    ids, inv = np.unique(x, return_inverse=True)
    inv = inv.reshape(x.shape)
    if ids[0] == 0:
        return inv.astype(np.int32), len(ids) - 1
    return (inv + 1).astype(np.int32), len(ids)


@dataclass
class Overlap:
    """Pairwise statistics between GT (rows) and pred (cols) instances, background removed."""
    inter: np.ndarray  # (nt, np) pixel intersections
    area_t: np.ndarray  # (nt,)
    area_p: np.ndarray  # (np,)

    @property
    def iou(self) -> np.ndarray:
        union = self.area_t[:, None] + self.area_p[None, :] - self.inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, self.inter / union, 0.0)

    @property
    def nt(self) -> int:
        return len(self.area_t)

    @property
    def np_(self) -> int:
        return len(self.area_p)


def overlap(true: np.ndarray, pred: np.ndarray) -> Overlap:
    t, nt = relabel(true)
    p, npred = relabel(pred)
    joint = np.bincount((t.ravel().astype(np.int64) * (npred + 1) + p.ravel()), minlength=(nt + 1) * (npred + 1))
    joint = joint.reshape(nt + 1, npred + 1).astype(np.float64)
    return Overlap(inter=joint[1:, 1:], area_t=joint[1:, :].sum(1), area_p=joint[:, 1:].sum(0))


@dataclass
class PQStats:
    tp: int
    fp: int
    fn: int
    sum_iou: float
    paired: np.ndarray  # (tp, 2) 0-based (gt_idx, pred_idx) into the relabelled instances

    @property
    def dq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else np.nan

    @property
    def sq(self) -> float:
        return self.sum_iou / (self.tp + 1.0e-6)

    @property
    def pq(self) -> float:
        return self.dq * self.sq


def pq_from_overlap(ov: Overlap, match_iou: float = 0.5) -> PQStats:
    assert match_iou >= 0.5, "unique matching is only guaranteed for IoU >= 0.5"
    iou = ov.iou
    gi, pi = np.nonzero(iou > match_iou)
    return PQStats(tp=len(gi), fp=ov.np_ - len(pi), fn=ov.nt - len(gi),
                   sum_iou=float(iou[gi, pi].sum()), paired=np.stack([gi, pi], 1))


def pq(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> PQStats:
    return pq_from_overlap(overlap(true, pred), match_iou)


def aji_from_overlap(ov: Overlap, plus: bool = False) -> float:
    """AJI (MoNuSeg) or AJI+ (unique Munkres pairing), as in HoVer-Net stats_utils."""
    if ov.nt == 0 and ov.np_ == 0:
        return np.nan
    union = ov.area_t[:, None] + ov.area_p[None, :] - ov.inter
    iou = ov.inter / (union + 1.0e-6)
    if ov.nt == 0 or ov.np_ == 0:
        return 0.0
    if plus:
        gi, pi = linear_sum_assignment(-iou)
        keep = iou[gi, pi] > 0.0
    else:
        gi = np.arange(ov.nt)
        pi = np.argmax(iou, axis=1)
        keep = iou[gi, pi] > 0.0
    gi, pi = gi[keep], pi[keep]
    inter = ov.inter[gi, pi].sum()
    uni = union[gi, pi].sum()
    unpaired_t = np.setdiff1d(np.arange(ov.nt), gi)
    unpaired_p = np.setdiff1d(np.arange(ov.np_), pi)
    uni += ov.area_t[unpaired_t].sum() + ov.area_p[unpaired_p].sum()
    return float(inter / uni)


def centroids(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ids, (n, 2) float xy centroids) for instances in a label map."""
    lab, n = relabel(inst)
    if n == 0:
        return np.zeros(0, int), np.zeros((0, 2), np.float32)
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=n + 1)[1:]
    cx = np.bincount(ids, xs, minlength=n + 1)[1:] / cnt
    cy = np.bincount(ids, ys, minlength=n + 1)[1:] / cnt
    return np.arange(1, n + 1), np.stack([cx, cy], 1).astype(np.float32)


def pair_coordinates(a: np.ndarray, b: np.ndarray, radius: float):
    """HoVer-Net `pair_coordinates`: Munkres on Euclidean distance, keep pairs within radius."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), int), np.arange(len(a)), np.arange(len(b))
    d = cdist(a, b)
    ia, ib = linear_sum_assignment(d)
    keep = d[ia, ib] <= radius
    pairing = np.stack([ia[keep], ib[keep]], 1)
    return pairing, np.setdiff1d(np.arange(len(a)), ia[keep]), np.setdiff1d(np.arange(len(b)), ib[keep])
