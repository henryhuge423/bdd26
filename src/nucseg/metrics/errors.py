"""Instance error taxonomy at IoU 0.5 (binary level, class-agnostic).

GT nuclei are labelled with exactly one status:
    matched       IoU > 0.5 with some prediction (unique by construction)
    merged        unmatched; >= thr of it is covered by a prediction that also covers
                  >= thr of at least one other GT nucleus (1 pred : many GT)
    split         unmatched; >= 2 predictions each lie >= thr inside it (many pred : 1 GT)
    missed_bg     unmatched, not merged/split, max IoU < 0.1 (not detected at all)
    missed_shape  unmatched, not merged/split, 0.1 <= max IoU <= 0.5 (poorly delineated)
Predictions are labelled matched / fp_merge (the merged blob) / fp_split (a fragment) /
fp_bg (max IoU < 0.1, spurious) / fp_shape.
"""

from __future__ import annotations

import numpy as np

from .instance import Overlap

GT_STATUS = ["matched", "merged", "split", "missed_bg", "missed_shape"]
PRED_STATUS = ["matched", "fp_merge", "fp_split", "fp_bg", "fp_shape"]


def classify_errors(ov: Overlap, thr: float = 0.5, low_iou: float = 0.1):
    nt, npred = ov.nt, ov.np_
    gt_status = np.zeros(nt, np.int8)
    pred_status = np.zeros(npred, np.int8)
    if nt == 0 or npred == 0:
        gt_status[:] = GT_STATUS.index("missed_bg")
        pred_status[:] = PRED_STATUS.index("fp_bg")
        return gt_status, pred_status, np.zeros(nt)
    iou = ov.iou
    max_iou = iou.max(1)
    frac_t = ov.inter / np.maximum(ov.area_t[:, None], 1)  # share of GT g covered by pred p
    frac_p = ov.inter / np.maximum(ov.area_p[None, :], 1)  # share of pred p lying inside GT g

    g_match = (iou > 0.5).any(1)
    p_match = (iou > 0.5).any(0)
    merger = (frac_t >= thr).sum(0) >= 2  # pred covering >= 2 GT nuclei
    g_merged = ~g_match & ((frac_t >= thr) & merger[None, :]).any(1)
    g_split = ~g_match & ~g_merged & ((frac_p >= thr).sum(1) >= 2)

    gt_status[:] = np.where(max_iou < low_iou, GT_STATUS.index("missed_bg"), GT_STATUS.index("missed_shape"))
    gt_status[g_split] = GT_STATUS.index("split")
    gt_status[g_merged] = GT_STATUS.index("merged")
    gt_status[g_match] = GT_STATUS.index("matched")

    p_max_iou = iou.max(0)
    fragment = ((frac_p >= thr) & g_split[:, None]).any(0)
    pred_status[:] = np.where(p_max_iou < low_iou, PRED_STATUS.index("fp_bg"), PRED_STATUS.index("fp_shape"))
    pred_status[fragment] = PRED_STATUS.index("fp_split")
    pred_status[merger & ~p_match] = PRED_STATUS.index("fp_merge")
    pred_status[p_match] = PRED_STATUS.index("matched")
    return gt_status, pred_status, max_iou


def touching_counts(inst: np.ndarray, n: int, radius: int = 2) -> np.ndarray:
    """Number of other instances within `radius` px (Chebyshev) of each instance 1..n."""
    if n == 0:
        return np.zeros(0, int)
    pairs = []
    h, w = inst.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dy, dx) <= (0, 0):
                continue
            a = inst[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            b = inst[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
            m = (a > 0) & (b > 0) & (a != b)
            if m.any():
                pairs.append(np.stack([a[m], b[m]], 1))
    if not pairs:
        return np.zeros(n, int)
    p = np.unique(np.sort(np.concatenate(pairs), 1), axis=0)
    return np.bincount(p.ravel(), minlength=n + 1)[1:]
