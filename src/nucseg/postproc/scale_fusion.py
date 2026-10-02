"""Conservative native-grid additions and diagnostics; no training or GT in deployable fusion."""
from __future__ import annotations

import numpy as np

from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import _pq, instance_classes


def _validate(*maps):
    shape = maps[0].shape
    if len(shape) != 2 or any(m.shape != shape or m.dtype.kind not in "iu" or np.any(m < 0) for m in maps):
        raise ValueError("maps must be aligned nonnegative 2D integer arrays")


def probability_rows(payload):
    """Validate an original (not lever-relabelled) table and index it by image then actual ID."""
    image, ids, prob = (np.asarray(payload[k]) for k in ("inst_img", "inst_id", "inst_prob"))
    if image.ndim != 1 or ids.shape != image.shape or prob.shape != (len(ids), 6):
        raise ValueError("invalid probability table dimensions")
    if image.dtype.kind not in "iu" or ids.dtype.kind not in "iu" or np.any(image < 0) or np.any(ids <= 0):
        raise ValueError("invalid image/instance IDs")
    if not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1) or not np.allclose(prob.sum(1), 1, atol=1e-4):
        raise ValueError("invalid class probabilities")
    rows = {}
    for image_id, instance_id, p in zip(image, ids, prob):
        row = rows.setdefault(int(image_id), {})
        if int(instance_id) in row:
            raise ValueError("duplicate image/instance probability row")
        row[int(instance_id)] = p
    return rows


def candidate_info(base, x2, x2_type):
    _validate(base, x2, x2_type)
    if np.any(x2_type > 5):
        raise ValueError("type IDs must be 0..5")
    _, classes = instance_classes(x2, x2_type)
    ids = np.unique(x2); ids = ids[ids > 0]
    result = []
    for i, cls in zip(ids, classes):
        mask = x2 == i
        result.append({"id": int(i), "class": int(cls), "area": int(mask.sum()),
                       "border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                       "disjoint": not bool(np.any(base[mask] > 0))})
    return result


def fuse(base, base_type, x2, x2_type, confidence, max_area, min_prob, interior_only,
         allowed_ids=None):
    """Preserve base exactly; add whole, non-overlapping x2 objects satisfying fixed filters."""
    _validate(base, base_type, x2, x2_type)
    if not np.isfinite(min_prob) or not 0 <= min_prob <= 1 or (max_area is not None and max_area <= 0):
        raise ValueError("invalid candidate threshold")
    out = base.astype(np.int32, copy=True)
    typ = np.where(base > 0, base_type, 0).astype(np.uint8)
    nxt = int(base.max())
    chosen = []
    for c in candidate_info(base, x2, x2_type):
        i = c["id"]
        if not c["disjoint"] or c["class"] == 0 or (allowed_ids is not None and i not in allowed_ids):
            continue
        if (max_area is not None and c["area"] > max_area) or (interior_only and c["border"]):
            continue
        if min_prob > 0:
            if i not in confidence or not np.isfinite(confidence[i]):
                raise ValueError(f"missing confidence for actual instance ID {i}")
            if confidence[i] < min_prob:
                continue
        nxt += 1
        mask = x2 == i
        out[mask] = nxt
        typ[mask] = c["class"]
        chosen.append(i)
    return out, typ, chosen


def transfer_types(base, base_type, source, source_type):
    """Change types only on IoU>.5 pairs with exactly one overlapping object on either side."""
    _validate(base, base_type, source, source_type)
    ov = overlap(base, source)
    base_ids = np.unique(base); base_ids = base_ids[base_ids > 0]
    _, classes = instance_classes(source, source_type)
    stable = (ov.iou > .5) & ((ov.inter > 0).sum(1)[:, None] == 1) & ((ov.inter > 0).sum(0)[None, :] == 1)
    out = np.where(base > 0, base_type, 0).astype(np.uint8)
    count = 0
    for b, s in zip(*np.nonzero(stable)):
        if classes[s] > 0:
            out[base == base_ids[b]] = classes[s]
            count += 1
    return out, count


def oracle_additions(base, base_type, x2, x2_type, gt, gt_type):
    """GT-assisted diagnostic ONLY: retain base, accept disjoint correctly typed IoU>.5 matches."""
    _validate(base, base_type, x2, x2_type, gt, gt_type)
    ov = overlap(gt, x2)
    _, gc = instance_classes(gt, gt_type)
    _, pc = instance_classes(x2, x2_type)
    ids = np.unique(x2); ids = ids[ids > 0]
    correct = (ov.iou > .5) & (gc[:, None] == pc[None, :]) & (gc[:, None] > 0)
    allowed = set(ids[np.any(correct, axis=0)].tolist())
    return fuse(base, base_type, x2, x2_type, {}, None, 0, False, allowed_ids=allowed)


def strict_dead_pq(stats):
    """Canonical class-absent-image treatment from light.image_stats (Dead is row 4)."""
    vals = []
    for s in np.asarray(stats)[:, 4]:
        if s[4] > 0:
            vals.append(_pq(*s[:4]))
        elif s[0] + s[1] > 0:
            vals.append(0.)
    return float(np.mean(vals)) if vals else float("nan")


STATE_NAMES = ["matched_correct", "matched_wrong", "merged", "split", "missed_bg", "missed_shape"]


def _gt_facts(gt, gt_type, pred, pred_type):
    from nucseg.metrics.errors import classify_errors
    ov = overlap(gt, pred)
    _, gc = instance_classes(gt, gt_type)
    _, pc = instance_classes(pred, pred_type)
    ids = np.unique(gt); ids = ids[ids > 0]
    edges = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]]))
    interior = ~np.isin(ids, edges)
    status, _, _ = classify_errors(ov)
    states = np.array([0, 2, 3, 4, 5], np.int8)[status]
    gi, pi = np.nonzero(ov.iou > .5)
    states[gi[gc[gi] != pc[pi]]] = 1
    dead = (gc == 4) & interior
    counters = {"interior_gt": int(dead.sum()), "interior_matched": int(((states <= 1) & dead).sum()),
                "negative_images": int(not np.any(gc == 4)),
                "dead_predictions_on_negative": int((pc == 4).sum()) if not np.any(gc == 4) else 0}
    return ov, gc, pc, interior, states, counters


def image_diagnostics(gt, gt_type, base, base_type, x2, x2_type):
    bo, gc, _, interior, bs, bd = _gt_facts(gt, gt_type, base, base_type)
    xo, _, pc, _, xs, xd = _gt_facts(gt, gt_type, x2, x2_type)
    strata = {"all": np.ones(len(gc), bool), "Dead": gc == 4,
              "Dead_interior": (gc == 4) & interior}
    for low, high in ((0, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 1000000)):
        strata[f"nonDead_{low}_{high}"] = (gc != 4) & (bo.area_t >= low) & (bo.area_t < high)
    transitions = {}
    for name, selected in strata.items():
        mat = np.zeros((6, 6), np.int64)
        np.add.at(mat, (bs[selected], xs[selected]), 1)
        transitions[name] = mat.tolist()
    candidates = {"all_disjoint": 0, "binary_tp": 0, "typed_tp": 0, "predicted_dead": 0,
                  "typed_dead_tp": 0, "typed_fp": 0}
    ids = np.unique(x2); ids = ids[ids > 0]
    positions = {int(i): j for j, i in enumerate(ids)}
    for c in candidate_info(base, x2, x2_type):
        if not c["disjoint"] or c["class"] == 0:
            continue
        matches = np.flatnonzero(xo.iou[:, positions[c["id"]]] > .5)
        typed = bool(len(matches) and gc[matches[0]] == c["class"])
        candidates["all_disjoint"] += 1
        candidates["binary_tp"] += int(bool(len(matches)))
        candidates["typed_tp"] += int(typed)
        candidates["typed_fp"] += int(not typed)
        candidates["predicted_dead"] += int(c["class"] == 4)
        candidates["typed_dead_tp"] += int(typed and c["class"] == 4)
    return {"transitions": transitions, "base_dead": bd, "x2_dead": xd, "candidates": candidates}


def select_fusion(rows):
    identity = [r for r in rows if r["name"] == "identity"]
    if len(identity) != 1:
        raise ValueError("one identity reference is required")
    base = identity[0]
    eligible = [r for r in rows if r["name"] not in ("identity", "unfiltered")
                and np.isfinite([r["mPQ"], r["bPQ"], r["strict_dead"]]).all()
                and r["mPQ"] >= base["mPQ"] - .002 and r["bPQ"] >= base["bPQ"] - .002
                and r["strict_dead"] >= base["strict_dead"]
                and r["interior_matched"] > base["interior_matched"]]
    return max(eligible, key=lambda r: (r["mPQ"], -r["added"], r["name"])) if eligible else base
