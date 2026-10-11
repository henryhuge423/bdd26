"""E3 candidate feature matrix (spec 2026-10-11 §2): frozen 12-dim menu over record columns.

`select_training_rows` whitelists the fold1 OOF pairs (split2 seeds); anything else in the
records file (fold2/fold3 rows) is excluded before any fitting. `design_matrix` raises on
non-finite values and out-of-range classes instead of letting NaN propagate.
"""
from __future__ import annotations

import numpy as np

TRAIN_PAIRS = ("split2_seed19", "split2_seed1", "split2_seed2")
DIST_CAP = 362.0  # 256^2 对角线：图内距离的几何上界；dist=inf（图内无 base 实例）截断到此
_BASE = ["log_area", "extent", "circularity", "dist", "pmax", "margin", "border"]
FEATURE_NAMES = _BASE + [f"class{c}" for c in range(5)]


def select_training_rows(rec: dict) -> dict:
    """Restrict a records dict to the fold1 OOF training pairs, preserving row order."""
    mask = np.isin(np.asarray(rec["pair"]), TRAIN_PAIRS)
    return {k: np.asarray(v)[mask] for k, v in rec.items()}


def design_matrix(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X (N,12) float64, y (N,) bool) with log1p(area) and a 5-way class one-hot."""
    for col in ("area", "extent", "circularity", "dist", "pmax", "margin", "class", "matched"):
        if col not in rec:
            raise ValueError(f"records missing column {col}")
    cols = {"log_area": np.log1p(np.asarray(rec["area"], float)),
            "extent": np.asarray(rec["extent"], float),
            "circularity": np.asarray(rec["circularity"], float),
            "dist": np.minimum(np.asarray(rec["dist"], float), DIST_CAP),
            "pmax": np.asarray(rec["pmax"], float),
            "margin": np.asarray(rec["margin"], float),
            "border": np.asarray(rec["border"], float)}
    cls = np.asarray(rec["class"])
    if cls.size and (cls.min() < 1 or cls.max() > 5):
        raise ValueError(f"class out of range 1..5: {sorted(set(cls.tolist()))}")
    X = np.column_stack([cols[n] for n in _BASE]
                         + [np.eye(5, dtype=float)[cls - 1].reshape(len(cls), 5)])
    for j, name in enumerate(FEATURE_NAMES):
        if not np.isfinite(X[:, j]).all():
            bad = int(np.flatnonzero(~np.isfinite(X[:, j]))[0])
            raise ValueError(f"non-finite value in feature {name} at row {bad}")
    return X, np.asarray(rec["matched"], bool)
