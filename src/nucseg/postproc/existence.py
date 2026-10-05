"""Existence-side candidate features for scale-fusion additions (audit + filter support)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def candidate_geometry(mask):
    """Area, bbox extent, pixel-perimeter circularity and border contact of one instance mask."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2D boolean array")
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    extent = area / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    ring = mask & ~ndimage.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    perimeter = max(int(ring.sum()), 1)
    circularity = min(4.0 * np.pi * area / perimeter**2, 1.0)
    return {"area": area, "extent": float(extent), "circularity": float(circularity),
            "border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}


def distance_to_base_map(base):
    """EDT of the complement of the base instances: px distance to the nearest base nucleus."""
    base = np.asarray(base) > 0
    if base.ndim != 2 or not base.any():
        raise ValueError("base must contain at least one instance")
    return ndimage.distance_transform_edt(~base)


def distance_to_base(mask, dist_map):
    """Distance of a candidate's closest pixel to the nearest base instance."""
    mask = np.asarray(mask, bool)
    if mask.shape != np.asarray(dist_map).shape or not mask.any():
        raise ValueError("mask must be non-empty and aligned with the distance map")
    return float(np.asarray(dist_map)[mask].min())
