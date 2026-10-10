"""E1 候选盲审抽样与绘图纯逻辑。

Spec: docs/superpowers/specs/2026-10-10-e1-blind-audit-design.md（2026-10-10 批准，值冻结）：
四源各 50、tissue×border 分层、比例分配、组配额 <3 并入同 tissue 另一 border 组、无放回、
窗口 160×160、种子 20261010。入样概率逐对象记录（Horvitz–Thompson 分析用）。
"""
from __future__ import annotations

import cv2
import numpy as np

WINDOW = 160
MERGE_MIN = 3


def _alloc(sizes: dict, n_total: int) -> dict:
    """Largest-remainder integer allocation of n_total across cells proportional to sizes."""
    tot = sum(sizes.values())
    raw = {k: n_total * n / tot for k, n in sizes.items()}
    base = {k: int(v) for k, v in raw.items()}
    rem = n_total - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    return base


def stratified_sample(rows, n_total, rng, merge_min=MERGE_MIN):
    """rows: dicts with `tissue`, `border` (+ any payload). Returns sampled row dicts with `prob`.

    Proportional allocation over tissue×border cells; a cell whose quota < merge_min is merged
    into the same tissue's other border cell (spec §1 pre-registered rule); sampling within a
    (possibly merged) cell is without replacement; prob = n_cell / N_cell of the merged cell.
    """
    cells = {}
    for r in rows:
        cells.setdefault((r["tissue"], bool(r["border"])), []).append(r)
    sizes = {k: len(v) for k, v in cells.items()}
    pre = {k: n_total * n / sum(sizes.values()) for k, n in sizes.items()}
    for key in [k for k, q in pre.items() if q < merge_min]:
        partner = (key[0], not key[1])
        if partner in cells:
            cells[partner].extend(cells.pop(key))
    alloc = _alloc({k: len(v) for k, v in cells.items()}, n_total)
    out = []
    for key in sorted(cells):
        members = cells[key]
        n = min(alloc[key], len(members))
        for i in rng.choice(len(members), n, replace=False):
            r = dict(members[i])
            r["prob"] = n / len(members)
            out.append(r)
    return out


def background_windows(tissue_arr, n_total, rng, shape=(256, 256), size=WINDOW):
    """S4 random windows: per-tissue allocation, images WITHOUT replacement, one uniform center each.

    pi_i = (n_t / N_t) * (1 / C), C = number of valid centers = (shape - size + 1)^2 (97^2 on 256).
    """
    by_t = {t: np.nonzero(tissue_arr == t)[0] for t in np.unique(tissue_arr)}
    alloc = _alloc({t: len(v) for t, v in by_t.items()}, n_total)
    C = (shape[0] - size + 1) * (shape[1] - size + 1)
    out = []
    for t, imgs in sorted(by_t.items()):
        n = min(alloc[t], len(imgs))
        for img in rng.choice(imgs, n, replace=False):
            cy = int(rng.integers(size // 2, shape[0] - size // 2 + 1))
            cx = int(rng.integers(size // 2, shape[1] - size // 2 + 1))
            out.append({"image": int(img), "cy": cy, "cx": cx, "tissue": str(t),
                        "prob": (n / len(imgs)) / C})
    return out


def window_slice(shape, cy, cx, size=WINDOW):
    """Constant-`size` window around (cy, cx), clamped into the image.

    Returns (slice_y, slice_x, (offset_y, offset_x)) where offset = window's top-left corner.
    """
    oy = int(np.clip(cy - size // 2, 0, shape[0] - size))
    ox = int(np.clip(cx - size // 2, 0, shape[1] - size))
    return slice(oy, oy + size), slice(ox, ox + size), (oy, ox)


def draw_cross(img: np.ndarray, cy: int, cx: int, half: int = 4, color=(0, 0, 255)) -> np.ndarray:
    """9 px red target cross at (cy, cx) on a copy (BGR)."""
    out = img.copy()
    cv2.line(out, (cx - half, cy), (cx + half, cy), color, 1)
    cv2.line(out, (cx, cy - half), (cx, cy + half), color, 1)
    return out


def overlay_outlines(img: np.ndarray, masks, colors) -> np.ndarray:
    """Draw each boolean mask's contour in its color (BGR) on a copy."""
    out = img.copy()
    for mask, color in zip(masks, colors):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 1)
    return out
