"""Light-weight official mPQ / bPQ for fast sweeps (post-processing variants on a validation fold).

Computes only the binary and per-class PQ statistics of `pannuke_eval._eval_image` (same helpers, same
image -> tissue -> mean aggregation), skipping AJI, detection F-scores and the error taxonomy.
Tested equal to `pannuke_eval.evaluate` (tests/test_recovery.py).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..constants import CLASS_NAMES, NUM_CLASSES, TISSUES
from .instance import overlap, pq_from_overlap
from .pannuke_eval import _pq, instance_classes


def image_stats(gt_ch: np.ndarray, gt_inst: np.ndarray, gt_type: np.ndarray, pred_inst: np.ndarray,
                pred_type: np.ndarray) -> np.ndarray:
    """-> (6, 5) float: row 0 = binary (tp, fp, fn, sum_iou, n_gt), rows 1..5 = per class
    (tp, fp, fn, sum_iou, n_gt) exactly as pannuke_eval._eval_image."""
    out = np.zeros((NUM_CLASSES + 1, 5))
    g_lab, _ = instance_classes(gt_inst, gt_type)
    p_lab, p_cls = instance_classes(pred_inst, pred_type)
    ov = overlap(g_lab, p_lab)
    s = pq_from_overlap(ov)
    out[0] = (s.tp, s.fp, s.fn, s.sum_iou, ov.nt)
    for c in range(NUM_CLASSES):
        p = np.where(np.isin(p_lab, np.nonzero(p_cls == c + 1)[0] + 1), p_lab, 0)
        ovc = overlap(gt_ch[..., c], p)
        sc = pq_from_overlap(ovc)
        out[c + 1] = (sc.tp, sc.fp, sc.fn, sc.sum_iou, ovc.nt)
    return out


def summarize(stats: np.ndarray, tissue: np.ndarray) -> dict:
    """stats (N, 6, 5) from image_stats -> official mPQ, bPQ and per-class PQ (image nanmean)."""
    tissue = np.asarray(tissue)
    bpq = np.array([_pq(*s[0, :4]) if s[0, 4] > 0 else np.nan for s in stats])
    cls_pq = np.array([[_pq(*s[c + 1, :4]) if s[c + 1, 4] > 0 else np.nan for c in range(NUM_CLASSES)]
                       for s in stats])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img_mpq = np.nanmean(cls_pq, 1)

        def tissue_avg(v):
            return float(np.nanmean([np.nanmean(v[tissue == t]) for t in TISSUES if (tissue == t).any()]))

        return {"mPQ": tissue_avg(img_mpq), "bPQ": tissue_avg(bpq),
                "per_class_PQ": {CLASS_NAMES[c]: float(np.nanmean(cls_pq[:, c])) for c in range(NUM_CLASSES)}}
