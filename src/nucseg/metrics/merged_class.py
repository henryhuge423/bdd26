"""Merged-class (4-of-5) official/strict mPQ over RetypeEvaluator's class-agnostic match tables.

Merging classes a,b remaps BOTH sides of every IoU>0.5 pair (GT class via ev.m_cls, predicted
class via the instance table) and BOTH count tables (per-image GT counts ev.n_gt, predicted
instance counts); PQ matching itself is class-agnostic so the pairs never change. Aggregation
is the official one with C=4: per-image nanmean over merged classes present in GT -> tissue
nanmean -> mean over tissues. bPQ is class-independent and is NOT derivable from these
channel-based tables (PanNuke gt_channels disagree with inst.npy on a small pixel fraction;
see scripts/merged_class_diag.bpq_inst_maps — spec 2026-10-13 invariant 1)."""

from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES, TISSUES
from .fast_retype import RetypeEvaluator


def class_map(pair: tuple[int, int] | None) -> np.ndarray:
    """(6,) compact merged class ids: 0->0; a,b -> one id; survivors compacted to 1..C.

    None -> identity [0,1,2,3,4,5]; (2,3) -> [0,1,2,2,3,4]; (1,5) -> [0,1,2,3,4,1].
    cmap[0] == 0 always (untyped instances stay untyped)."""
    if pair is not None:
        a, b = pair
        if not (1 <= a < b <= NUM_CLASSES):
            raise ValueError(f"invalid class pair {pair}: need 1 <= a < b <= {NUM_CLASSES}")
    cmap = np.arange(NUM_CLASSES + 1, dtype=np.int64)
    if pair is not None:
        a, b = sorted(pair)
        cmap[b] = a
    uniq, new = np.unique(cmap[1:], return_inverse=True)  # compact survivors to 1..C
    out = np.zeros_like(cmap)
    out[1:] = new + 1
    return out


def per_image_class_pq(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """(n, C) merged-class PQ, NaN where the merged class is absent from GT (official convention).

    Keeps fast_retype's row order (ev.m_* consumed as-is) so the identity map is bit-exact."""
    n, C = ev.n, int(cmap.max())
    g = cmap[ev.m_cls]                      # pair 的 GT 合并类（>=1）
    tp_sel = cmap[cls[ev.m_row]] == g       # cls==0 -> cmap[0]==0 != g，天然排除
    key = ev.m_img[tp_sel] * C + g[tp_sel] - 1
    tp = np.bincount(key, minlength=n * C).reshape(n, C).astype(np.float64)
    siou = np.bincount(key, ev.m_iou[tp_sel], minlength=n * C).reshape(n, C)
    n_gt_m = np.zeros((n, C), np.int64)     # 两侧计数同步重映射
    for c in range(1, NUM_CLASSES + 1):
        n_gt_m[:, cmap[c] - 1] += ev.n_gt[:, c - 1]
    v = cls > 0
    npred = np.bincount(ev.inst_img[v] * C + cmap[cls[v]] - 1, minlength=n * C).reshape(n, C)
    fp, fn = npred - tp, n_gt_m - tp
    with np.errstate(all="ignore"):
        d = tp + 0.5 * fp + 0.5 * fn
        pq = np.where(d > 0, (tp / d) * (siou / (tp + 1.0e-6)), np.nan)
    return np.where(n_gt_m > 0, pq, np.nan)


def mpq(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> tuple[float, np.ndarray]:
    """Tissue-averaged official mPQ and per-class PQ (image nanmean) in the merged space."""
    pq = per_image_class_pq(ev, cls, cmap)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        img = np.nanmean(pq, 1)
        per_t = [np.nanmean(img[ev.tissue == t]) for t in TISSUES if (ev.tissue == t).any()]
        return float(np.nanmean(per_t)), np.nanmean(pq, 0)


def _pred_counts(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """(n, C) predicted-instance counts per image and merged class (class 0 excluded)."""
    n, C = ev.n, int(cmap.max())
    v = cls > 0
    return np.bincount(ev.inst_img[v] * C + cmap[cls[v]] - 1, minlength=n * C).reshape(n, C)


def _strict_table(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """(n, C) strict PQ table: official table with 0 where GT lacks a merged class that is
    predicted (pannuke_eval's strict convention)."""
    pq = per_image_class_pq(ev, cls, cmap)
    n_gt_m = np.zeros(pq.shape, np.int64)
    for c in range(1, NUM_CLASSES + 1):
        n_gt_m[:, cmap[c] - 1] += ev.n_gt[:, c - 1]
    return np.where((n_gt_m <= 0) & (_pred_counts(ev, cls, cmap) > 0), 0.0, pq)


def mpq_strict(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> tuple[float, np.ndarray]:
    """Tissue-averaged strict mPQ: merged classes absent in GT but predicted score 0."""
    strict = _strict_table(ev, cls, cmap)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        img = np.nanmean(strict, 1)
        per_t = [np.nanmean(img[ev.tissue == t]) for t in TISSUES if (ev.tissue == t).any()]
        return float(np.nanmean(per_t)), np.nanmean(strict, 0)


def bootstrap_delta(ev: RetypeEvaluator, cls: np.ndarray, cmap_a: np.ndarray, cmap_b: np.ndarray,
                    n_boot: int = 1000, seed: int = 0, cls_b: np.ndarray | None = None,
                    strict: bool = False) -> dict:
    """Paired image bootstrap of mPQ(b) - mPQ(a); images resampled WITHIN each tissue stratum
    (spec 2026-10-13 §2: 组织内分层, 1000 次, percentile CI).

    cls_b: a different instance-class assignment for side b (same length as cls) — used for
    same-space intervention deltas (e.g. typing-oracle assignments under one merged cmap).
    strict: use the strict PQ tables instead of the official ones.

    Resampling aggregates each stratum over ITS OWN draws (block-wise), mirroring the point
    estimator; slicing a resample-concatenated vector by the original (interleaved) tissue
    positions mixes strata and miscenters the CI (review 2026-10-14)."""
    table = _strict_table if strict else per_image_class_pq

    def img_mpq(cls_x, cmap_x):
        with np.errstate(all="ignore"):
            return np.nanmean(table(ev, cls_x, cmap_x), 1)

    pa = img_mpq(cls, cmap_a)
    pb = img_mpq(cls_b if cls_b is not None else cls, cmap_b)
    # TISSUES order (as mpq) so the outer mean's float order matches the point estimator
    strata = [np.nonzero(ev.tissue == t)[0] for t in TISSUES if (ev.tissue == t).any()]
    rng = np.random.default_rng(seed)

    def m(v):
        return np.nanmean([np.nanmean(v[s]) for s in strata])

    deltas = np.empty(n_boot)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for b in range(n_boot):
            db, da = [], []
            for s in strata:
                picks = rng.choice(s, len(s), replace=True)  # one draw, both sides (paired)
                db.append(np.nanmean(pb[picks]))
                da.append(np.nanmean(pa[picks]))
            deltas[b] = np.nanmean(db) - np.nanmean(da)
        lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {"delta": float(m(pb) - m(pa)), "lo": float(lo), "hi": float(hi)}
