"""E4 visibility-probe pure logic: FOV-crop conditions, GT identity grouping, miss tables, DiD.

Everything here is side-effect free and torch-free: the runner feeds images/GT/predictions,
this module decides windows, groups, and statistics (spec 2026-10-13
docs/superpowers/specs/2026-10-13-visibility-probe-e4-design.md)."""
from __future__ import annotations

import numpy as np

from nucseg.constants import TISSUES
from nucseg.metrics.instance import overlap

PATCH = 256


def sample_windows(n: int, size: int, seed: int) -> np.ndarray:
    """(n, 2) int32 (dy, dx) crop origins, uniform over the legal range {0..PATCH-size}^2,
    one window per image, bit-exact under a fixed seed (spec E4 §3)."""
    hi = PATCH - size  # 192 -> 0..64, 160 -> 0..96
    rng = np.random.default_rng(seed)
    return rng.integers(0, hi + 1, size=(n, 2)).astype(np.int32)


def shift_pad(arr: np.ndarray, d: int) -> np.ndarray:
    """C1 replicate-pad shift: content moves +y+x by d pixels, edge-filled (last (H,W)
    planes untouched for (H,W,C) inputs)."""
    pad = ((d, 0), (d, 0)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad, mode="edge")[:arr.shape[0], :arr.shape[1]]


def crop_group(inst: np.ndarray, win: tuple[int, int, int]) -> dict[int, dict]:
    """Per original GT id (ALL ids of the full frame, so outside objects appear too)
    under crop window win=(y, x, size): the visible mask in crop coordinates, its area
    (crop-native px), whether it touches the crop border, and the spec §4 group —
    "truncated" (mask intersects the crop boundary), "full", "outside" (no visible px).

    Note the spec's literal criterion: an object ending at the IMAGE edge under a window
    edge also lying there counts as truncated; such objects are excluded from H2/H3 anyway
    by the original_full filter (they touch the original outer ring)."""
    y, x, size = win
    sub_frame = inst[y:y + size, x:x + size]
    out: dict[int, dict] = {}
    for i in np.unique(inst):
        if i == 0:
            continue
        visible = sub_frame == i
        touches = bool(visible[0].any() or visible[-1].any()
                       or visible[:, 0].any() or visible[:, -1].any())
        area = int(visible.sum())
        group = "outside" if area == 0 else ("truncated" if touches else "full")
        out[int(i)] = {"visible": visible, "area": area, "touches_crop": touches,
                       "group": group}
    return out


def original_full(inst: np.ndarray) -> dict[int, bool]:
    """Per GT id: True iff the nucleus does NOT touch the original 256 px outer ring
    (the H2/H3 admission filter; H1's 'border' stratum is the complement)."""
    out = {int(i): True for i in np.unique(inst) if i != 0}
    ring = np.concatenate([inst[0].ravel(), inst[-1].ravel(),
                           inst[:, 0].ravel(), inst[:, -1].ravel()])
    for i in np.unique(ring):
        if i != 0:
            out[int(i)] = False
    return out


def max_iou_table(gt_lab: np.ndarray, pred_inst: np.ndarray) -> dict[int, float]:
    """Per ORIGINAL GT label id -> max IoU against all predicted instances (any class).
    Ids are preserved (relabel's sorted-unique order maps row k -> k-th smallest id)."""
    ids = np.unique(gt_lab)
    ids = ids[ids > 0]
    iou = overlap(gt_lab, pred_inst).iou
    out = {}
    for k, i in enumerate(ids):
        out[int(i)] = 0.0 if iou.shape[1] == 0 else float(iou[k].max())
    return out


def miss_table(gt_lab: np.ndarray, pred_inst: np.ndarray, thr: float = 0.1) -> dict[int, bool]:
    """Per ORIGINAL GT label id -> max IoU against ALL predicted instances < thr (any class;
    spec E4 §2 miss definition). No predicted instances -> every GT nucleus misses."""
    return {i: v < thr for i, v in max_iou_table(gt_lab, pred_inst).items()}


def bin_area(a: int) -> str:
    """Fragment-area bin in crop-native px (spec §4)."""
    return "<30" if a < 30 else ("30-100" if a < 100 else ">=100")


def bin_area_native(a: int) -> str:
    """H1 stratification bin in original-frame px (spec §4)."""
    return "<100" if a < 100 else ("100-400" if a < 400 else ">=400")


def rate_gap_bootstrap(miss_a: np.ndarray, miss_b: np.ndarray, img: np.ndarray,
                       tissue: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """P(miss|a) - P(miss|b) over paired nuclei. Point estimate = pooled per-nucleus rate
    difference over the fold; each bootstrap draw resamples IMAGES with replacement WITHIN
    each tissue stratum (nuclear rows of an image stay together) and recomputes the SAME
    pooled statistic on the drawn multiset — stratification shapes the resampling design,
    not the estimand (lesson of the 2026-10-14 review: draws must mirror the point).
    NaN entries drop from their side's denominator. Percentile 95% CI."""
    a = np.asarray(miss_a, float)
    b = np.asarray(miss_b, float)
    strata = [s for t in TISSUES for s in [np.nonzero(tissue == t)[0]] if len(s)]

    def pooled_gap(rows):
        with np.errstate(all="ignore"):
            return np.nanmean(a[rows]) - np.nanmean(b[rows])

    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for k in range(n_boot):
            parts = []
            for s in strata:
                imgs = np.unique(img[s])
                picks = rng.choice(imgs, len(imgs), replace=True)
                parts.append(np.concatenate([np.nonzero(img == i)[0] for i in picks]))
            if parts:
                deltas[k] = pooled_gap(np.concatenate(parts))
        lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    # point: each side NaN-dropped independently (masked/unpaired use — H1 groups, H3 bins;
    # paired two-condition deltas go through the runner's did_bootstrap instead)
    with np.errstate(all="ignore"):
        point = float(np.nanmean(a) - np.nanmean(b))
    return {"gap": point, "lo": float(lo), "hi": float(hi)}


def did(trunc_miss_ck: np.ndarray, trunc_miss_c0: np.ndarray,
        full_miss_ck: np.ndarray, full_miss_c0: np.ndarray) -> float:
    """DiD = [P(miss|newly truncated, Ck) - P(miss|same nuclei, C0)]
          - [P(miss|stays full, Ck) - P(miss|same nuclei, C0)]  (spec E4 §2 H2)."""
    t_ck, t_c0 = np.nanmean(trunc_miss_ck), np.nanmean(trunc_miss_c0)
    f_ck, f_c0 = np.nanmean(full_miss_ck), np.nanmean(full_miss_c0)
    return float((t_ck - t_c0) - (f_ck - f_c0))
