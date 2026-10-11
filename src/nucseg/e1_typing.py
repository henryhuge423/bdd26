"""E1-T pure logic: base-GT pairing, jury typing votes, base-independent majority rule.

All majority/vote definitions are written WITHOUT reference to the base model's own
prediction (only `pair_base` sees the base), per spec 2026-10-11 §2 so the G0 contrast
does not degenerate. Side-effect free; no disk access.
"""
from __future__ import annotations

import numpy as np

from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import instance_classes


def _sorted_ids(x: np.ndarray) -> np.ndarray:
    ids = np.unique(x)
    return ids[ids > 0]


def pair_base(gt: np.ndarray, base: np.ndarray, thr: float = 0.5) -> dict[int, tuple[int, float] | None]:
    """GT id -> (base id, IoU) for the unique max-IoU base instance above `thr`, else None.

    Two base instances tying for the maximum (within 1e-12) make the pairing ambiguous:
    the GT is excluded (None) and counted by the caller.
    """
    gt_ids = _sorted_ids(gt)
    out: dict[int, tuple[int, float] | None] = {}
    if len(gt_ids) == 0:
        return out
    base_ids = _sorted_ids(base)
    ov = overlap(gt, base)
    for g, gt_id in enumerate(gt_ids):
        col = ov.iou[g]
        if col.size == 0 or col.max() <= thr:
            out[int(gt_id)] = None
            continue
        j = int(np.argmax(col))
        ties = np.flatnonzero(np.isclose(col, col[j], rtol=0.0, atol=1e-12))
        out[int(gt_id)] = None if len(ties) > 1 else (int(base_ids[j]), float(col[j]))
    return out


def jury_votes(gt: np.ndarray, member_inst: np.ndarray, member_type: np.ndarray,
               thr: float = 0.3) -> dict[int, tuple[int, int] | None]:
    """GT id -> (that member's best-overlap instance id, its type) if max IoU >= thr, else None.

    One member casts at most one vote per GT instance. The voted type is the instance's
    pixel-majority class under the canonical `instance_classes` semantics (background
    votes zeroed; class 0 = untyped), so mixed or all-background type maps are legal.
    """
    gt_ids = _sorted_ids(gt)
    out: dict[int, tuple[int, int] | None] = {}
    if len(gt_ids) == 0:
        return out
    mem_ids = _sorted_ids(member_inst)
    _, classes = instance_classes(member_inst, member_type)
    ov = overlap(gt, member_inst)
    for g, gt_id in enumerate(gt_ids):
        col = ov.iou[g]
        if col.size == 0 or col.max() < thr:
            out[int(gt_id)] = None
            continue
        j = int(np.argmax(col))
        out[int(gt_id)] = (int(mem_ids[j]), int(classes[j]))
    return out


def majority_type(votes: dict[int, int], folds: dict[int, int],
                  min_votes: int = 3) -> tuple[int | None, dict]:
    """Base/GT-independent jury majority (spec 2026-11 §2).

    Majority requires: total votes >= min_votes, the top type strictly beats the runner-up,
    and the top type received votes from members of at least two distinct training folds.
    """
    counts: dict[int, int] = {}
    voters: dict[int, set[int]] = {}
    for member, cls in votes.items():
        counts[cls] = counts.get(cls, 0) + 1
        voters.setdefault(cls, set()).add(folds[member])
    n = len(votes)
    if n < min_votes or not counts:
        return None, {"n_votes": n, "counts": counts, "both_folds_ok": False}
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    (top, top_n), (runners) = order[0], order[1][1] if len(order) > 1 else -1
    both = len(voters[top]) >= 2
    ok = top_n > runners and both
    return (int(top) if ok else None), {"n_votes": n, "counts": counts, "both_folds_ok": both}


def clustered_rate_bootstrap(indicator: np.ndarray, image_ids: np.ndarray,
                             n_boot: int = 2000, seed: int = 20261011) -> tuple[float, float]:
    """Percentile 95% CI of the mean rate, resampling whole images (clusters) with replacement.

    NaN indicators are excluded from both numerator and denominator of each replicate.
    """
    indicator = np.asarray(indicator, float)
    image_ids = np.asarray(image_ids)
    if indicator.shape != image_ids.shape:
        raise ValueError("indicator and image_ids must align")
    uniq = np.unique(image_ids)
    groups = [indicator[image_ids == u] for u in uniq]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        pooled = np.concatenate([groups[k] for k in pick])
        pooled = pooled[~np.isnan(pooled)]
        stats.append(pooled.mean() if pooled.size else np.nan)
    arr = np.asarray(stats, float)
    ok = arr[~np.isnan(arr)]
    if ok.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))
