"""Fast, exact official mPQ for *re-typing* experiments: instance masks fixed, only instance classes vary.

For fixed predicted instances, the official per-class matching (GT channel c vs. predicted instances of
class c, IoU > 0.5) is determined by the class-agnostic IoU>0.5 pairs between each GT channel and all
predicted instances (IoU > 0.5 pairs are unique). We precompute those pairs once; scoring an assignment
of classes to instances is then a few bincounts, with the same image -> tissue -> mean aggregation as
`pannuke_eval.evaluate`.
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np

from ..constants import NUM_CLASSES, TISSUES
from .instance import overlap


def _matches(args):
    gt_ch, pred_inst, ids = args
    lut = np.zeros(int(pred_inst.max()) + 1, np.int64)
    lut[ids] = np.arange(len(ids))
    out, n_gt = [], np.zeros(NUM_CLASSES, np.int64)
    for c in range(NUM_CLASSES):
        ov = overlap(gt_ch[..., c], pred_inst)
        n_gt[c] = ov.nt
        if ov.nt == 0 or ov.np_ == 0:
            continue
        iou = ov.iou
        gi, pi = np.nonzero(iou > 0.5)
        # relabel(pred) maps sorted unique ids -> 1..n, so column k is the k-th smallest id
        pid = np.unique(pred_inst)
        pid = pid[pid > 0][pi]
        out.append(np.stack([np.full(len(gi), c + 1), lut[pid], iou[gi, pi]], 1))
    return (np.concatenate(out) if out else np.zeros((0, 3))), n_gt


class RetypeEvaluator:
    """inst_img / inst_id: the per-instance table (rows = predicted instances) of a prediction file."""

    def __init__(self, gt_channels, pred_inst, inst_img, inst_id, tissue, workers: int = 16):
        self.n = len(tissue)
        self.tissue = np.asarray(tissue)
        order = np.lexsort((inst_id, inst_img))
        assert np.all(order == np.arange(len(order))), "instance table must be sorted by (img, id)"
        starts = np.searchsorted(inst_img, np.arange(self.n + 1))
        jobs = ((gt_channels[i], np.asarray(pred_inst[i]), inst_id[starts[i]:starts[i + 1]]) for i in range(self.n))
        with Pool(workers) as pool:
            res = pool.map(_matches, jobs, chunksize=16)
        rows = []
        for i, (m, _) in enumerate(res):
            if len(m):
                rows.append(np.c_[np.full(len(m), i), m[:, 0], m[:, 1] + starts[i], m[:, 2]])
        m = np.concatenate(rows)
        self.m_img, self.m_cls, self.m_row, self.m_iou = m[:, 0].astype(int), m[:, 1].astype(int), m[:, 2].astype(int), m[:, 3]
        self.n_gt = np.stack([r[1] for r in res])  # (n, 5)
        self.inst_img = inst_img

    def per_image_class_pq(self, cls: np.ndarray) -> np.ndarray:
        """cls (M,) class 0..5 of every predicted instance -> (n, 5) PQ, NaN where the class is absent in GT."""
        n, C = self.n, NUM_CLASSES
        tp_sel = cls[self.m_row] == self.m_cls
        key = self.m_img[tp_sel] * C + self.m_cls[tp_sel] - 1
        tp = np.bincount(key, minlength=n * C).reshape(n, C).astype(np.float64)
        siou = np.bincount(key, self.m_iou[tp_sel], minlength=n * C).reshape(n, C)
        v = cls > 0  # class 0 (untyped) instances count for no class, as in the official vote
        npred = np.bincount(self.inst_img[v] * C + cls[v] - 1, minlength=n * C).reshape(n, C)
        fp, fn = npred - tp, self.n_gt - tp
        with np.errstate(all="ignore"):
            d = tp + 0.5 * fp + 0.5 * fn
            pq = np.where(d > 0, (tp / d) * (siou / (tp + 1.0e-6)), np.nan)
        return np.where(self.n_gt > 0, pq, np.nan)

    def mpq(self, cls: np.ndarray) -> tuple[float, np.ndarray]:
        """Official tissue-averaged mPQ and per-class PQ (image nanmean) for an assignment."""
        pq = self.per_image_class_pq(cls)
        with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            img = np.nanmean(pq, 1)
            per_t = [np.nanmean(img[self.tissue == t]) for t in TISSUES if (self.tissue == t).any()]
            return float(np.nanmean(per_t)), np.nanmean(pq, 0)

    def _pred_counts(self, cls: np.ndarray) -> np.ndarray:
        """(n, C) predicted-instance counts per image and class (class 0 excluded)."""
        v = cls > 0
        return np.bincount(self.inst_img[v] * NUM_CLASSES + cls[v] - 1,
                           minlength=self.n * NUM_CLASSES).reshape(self.n, NUM_CLASSES)

    def mpq_strict(self, cls: np.ndarray) -> tuple[float, np.ndarray]:
        """Tissue-averaged strict mPQ: classes absent in GT but predicted score 0 (pannuke_eval)."""
        pq = self.per_image_class_pq(cls)
        strict = np.where((self.n_gt <= 0) & (self._pred_counts(cls) > 0), 0.0, pq)
        with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            img = np.nanmean(strict, 1)
            per_t = [np.nanmean(img[self.tissue == t]) for t in TISSUES if (self.tissue == t).any()]
            return float(np.nanmean(per_t)), np.nanmean(strict, 0)

    def mpq_bootstrap_delta(self, cls_a: np.ndarray, cls_b: np.ndarray,
                            n_boot: int = 1000, seed: int = 0) -> dict:
        """Paired image bootstrap of mPQ(a) - mPQ(b); images resampled WITHIN each tissue stratum
        (spec 2026-10-10 §2: 组织内分层, 1000 次, percentile CI)."""
        def img_mpq(pq):
            with np.errstate(all="ignore"):
                return np.nanmean(pq, 1)

        pa = img_mpq(self.per_image_class_pq(cls_a))
        pb = img_mpq(self.per_image_class_pq(cls_b))
        strata = [np.nonzero(self.tissue == t)[0] for t in np.unique(self.tissue)]
        rng = np.random.default_rng(seed)

        def m(v):
            return np.nanmean([np.nanmean(v[s]) for s in strata])

        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning)
                deltas[b] = m(pa[idx]) - m(pb[idx])
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
        return {"delta": float(m(pa) - m(pb)), "lo": float(lo), "hi": float(hi)}
