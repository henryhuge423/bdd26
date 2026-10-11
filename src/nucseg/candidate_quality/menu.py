"""E3 five-arm menu (spec 2026-10-11 §4): per-row allowed (image, id) sets over the frozen pool.

A1 reproduces the incumbent EF-P2 acceptance set (stage-1 P2 config plus the a30 area
floor) from the frozen record columns; the sweep runner separately hard-asserts the fused
row against the frozen selection. A2/A3/A4 are (floor, cap) / learned-threshold rules.
"""
from __future__ import annotations

import numpy as np

from nucseg.postproc.scale_fusion import select_fusion

AREA_GRID = [(0, 100), (0, 200), (0, 400), (30, 100), (30, 200), (30, 400)]
P_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)


def _keys(rec: dict) -> np.ndarray:
    return np.stack([np.asarray(rec["image"]), np.asarray(rec["id"])], 1)


def a1_allowed(rec: dict, p2cfg: dict, ef_floor: int = 30) -> tuple[set, int]:
    """Incumbent acceptance set: max_area cap, min_prob, optional interior-only, plus floor."""
    keep = ((np.asarray(rec["area"], float) <= p2cfg["max_area"])
            & (np.asarray(rec["pmax"], float) >= p2cfg["min_prob"])
            & (np.asarray(rec["area"], float) >= ef_floor))
    if p2cfg.get("interior_only"):
        keep &= ~np.asarray(rec["border"], bool)
    keys = _keys(rec)[keep]
    allowed = {(int(im), int(i)) for im, i in keys}
    return allowed, len(allowed)


def build_menu(rec: dict, p_low: np.ndarray, p_sup: np.ndarray) -> list[dict]:
    """18 rows: A0 identity, A1 incumbent, A2 area grid x6, A3/A4 learned thresholds x5."""
    keys = _keys(rec)
    area = np.asarray(rec["area"], float)
    rows = [{"name": "identity", "arm": "A0", "allowed": set()},
            {"name": "a1_incumbent", "arm": "A1", "allowed": None}]  # A1 由调用方经 a1_allowed 注入
    for f, c in AREA_GRID:
        m = (area >= f) & (area <= c)
        rows.append({"name": f"a2_f{f}_c{c}", "arm": "A2",
                     "allowed": {(int(a), int(b)) for a, b in keys[m]}})
    for t in P_GRID:
        rows.append({"name": f"a3_p{t:g}", "arm": "A3",
                     "allowed": {(int(a), int(b)) for a, b in keys[p_low >= t]}})
    for t in P_GRID:
        rows.append({"name": f"a4_p{t:g}", "arm": "A4",
                     "allowed": {(int(a), int(b)) for a, b in keys[p_sup >= t]}})
    if len(rows) != 18:
        raise AssertionError(f"menu must have 18 rows, got {len(rows)}")
    return rows


def selection_rows(sweep_rows: list[dict]) -> dict:
    """Per-arm EF-round selection: select_fusion([A0] + arm rows), identity reference = A0."""
    arms = ("A2", "A3", "A4")
    out = {}
    for r in sweep_rows:
        if r.get("arm") in ("A0", "A1"):
            out[r["arm"]] = r
    for arm in arms:
        arm_rows = [r for r in sweep_rows if r.get("arm") == arm]
        if arm_rows:
            out[arm] = select_fusion([out["A0"]] + arm_rows)
    return out
