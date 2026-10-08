"""EF-style merge of dead-expert candidates into a base instance set (DSB spec 2026-10-07 §4, gate v0).

Whole-object additions only: a candidate survives iff it shares ZERO pixels with the base set
(the same disjointness rule as nucseg.postproc.scale_fusion.fuse) and passes the area floor.
Base pixels are never touched; candidates are renumbered above base.max() and typed Dead.
No GT and no model internals here — pure map algebra, replayable bit-exactly.
"""
from __future__ import annotations

import numpy as np

from ..constants import DEAD_TYPE


def merge_dead(base, base_type, dead_inst, min_area: int = 0):
    """Add whole, fully disjoint dead candidates with area >= min_area.

    Returns (inst, type, chosen_ids): merged int32 instance map, uint8 type map, and the
    candidate ids that were added."""
    base, base_type, dead_inst = (np.asarray(m) for m in (base, base_type, dead_inst))
    if base.ndim != 2 or base.shape != dead_inst.shape or base.shape != base_type.shape:
        raise ValueError("base, base_type and dead_inst must be aligned 2D maps")
    if base.dtype.kind not in "iu" or dead_inst.dtype.kind not in "iu":
        raise ValueError("instance maps must be integer arrays")
    out = base.astype(np.int32, copy=True)
    typ = np.where(base > 0, base_type, 0).astype(np.uint8)
    nxt = int(base.max())
    chosen = []
    for i in np.unique(dead_inst):
        if i == 0:
            continue
        mask = dead_inst == i
        if not mask.any() or bool((base[mask] > 0).any()):
            continue  # any base overlap disqualifies the whole candidate (fuse rule)
        if int(mask.sum()) < int(min_area):
            continue
        nxt += 1
        out[mask] = nxt
        typ[mask] = DEAD_TYPE
        chosen.append(int(i))
    return out, typ, chosen
