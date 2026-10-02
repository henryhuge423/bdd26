"""Descriptive typing-confidence audit on saved predictions.

Records per-instance predicted-class confidence against actual typing correctness, the
candidates a frozen P2 config added or rejected, x1/x2 agreement on stable pairs, and
map-reproduction sanity for re-predicted tables. Nothing here selects thresholds; every
output is a counter or a record for downstream reporting.
"""
from __future__ import annotations

import numpy as np

from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import instance_classes
from nucseg.postproc.scale_fusion import _validate, candidate_info, fuse

# Confidence bins: [0.5,0.6), ... [0.9,1.0]; values below 0.5 (map class != row argmax) get "<0.5".
CONF_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CLASSES = (1, 2, 3, 4, 5)


def _ids(map_):
    ids = np.unique(map_)
    return ids[ids > 0]


def _bin(value):
    if value is None:
        return None
    if value < CONF_EDGES[0]:
        return "<0.5"
    for lo, hi in zip(CONF_EDGES[:-1], CONF_EDGES[1:]):
        if value < hi or hi == CONF_EDGES[-1]:
            return f"{lo:.1f}-{hi:.1f}"
    return f"{CONF_EDGES[-2]:.1f}-{CONF_EDGES[-1]:.1f}"


def _gt_match(ov_gt_pred, gc, pred_pos):
    """Best-IoU GT for one predicted instance: (matched, gt_class, best_iou)."""
    col = ov_gt_pred.iou[:, pred_pos]
    if col.size == 0:
        return False, None, 0.0
    best = int(col.argmax())
    iou = float(col[best])
    return iou > .5, (int(gc[best]) if iou > .5 else None), iou


def _confidence_fields(row, cls):
    if row is None:
        return None, None, None
    top = np.sort(row)
    return (float(row[cls]) if 0 <= cls < len(row) else None, int(row.argmax()),
            float(top[-1] - top[-2]) if len(row) > 1 else None)


def instance_records(inst, typ, gt, gt_type, rows=None):
    """Per predicted instance: map class, table row view, best-IoU GT match and typing correctness.

    `rows` maps the actual instance ID to its 6-column probability row; without it the
    confidence fields are None. `typed_correct` is None for unmatched instances.
    """
    _validate(inst, typ, gt, gt_type)
    ids = _ids(inst)
    if not len(ids):
        return []
    _, classes = instance_classes(inst, typ)
    ov = overlap(gt, inst)
    _, gc = instance_classes(gt, gt_type)
    records = []
    for pos, (i, cls) in enumerate(zip(ids, classes)):
        matched, gt_class, best_iou = _gt_match(ov, gc, pos)
        row = rows.get(int(i)) if rows is not None else None
        conf, argmax, margin = _confidence_fields(row, int(cls))
        records.append({"id": int(i), "class": int(cls), "area": int((inst == i).sum()),
                        "border": bool((inst == i)[0].any() or (inst == i)[-1].any()
                                       or (inst == i)[:, 0].any() or (inst == i)[:, -1].any()),
                        "conf": conf, "row_argmax": argmax, "margin": margin,
                        "matched": matched, "gt_class": gt_class, "best_iou": best_iou,
                        "typed_correct": (matched and gt_class == int(cls)) if matched else None})
    return records


def _binned(records, field):
    out = {f"{lo:.1f}-{hi:.1f}": {str(c): {"n": 0, "matched": 0, "correct": 0}
                                  for c in CLASSES} for lo, hi in zip(CONF_EDGES[:-1], CONF_EDGES[1:])}
    out["<0.5"] = {str(c): {"n": 0, "matched": 0, "correct": 0} for c in CLASSES}
    for r in records:
        cls = r.get("class")
        if cls not in CLASSES:
            continue
        cell = out[_bin(r.get(field))][str(cls)]
        cell["n"] += 1
        cell["matched"] += int(bool(r["matched"]))
        cell["correct"] += int(bool(r["typed_correct"]))
    return out


def confidence_table(records):
    """Confidence/margin reliability counters plus map-class vs row-argmax agreement."""
    agree = {"agree": 0, "disagree": 0, "agree_correct": 0, "disagree_correct": 0}
    for r in records:
        if r.get("row_argmax") is None:
            continue
        same = r["row_argmax"] == r["class"]
        agree["agree" if same else "disagree"] += 1
        agree["agree_correct" if same else "disagree_correct"] += int(bool(r["typed_correct"]))
    return {"conf_bins": _binned(records, "conf"), "margin_bins": _binned(records, "margin"),
            "row_argmax_agreement": agree}


def agreement_records(base, base_type, x2, x2_type, gt, gt_type, x2_rows=None):
    """Type agreement on stable one-to-one IoU>.5 base/x2 pairs (the P1 swap population)."""
    _validate(base, base_type, x2, x2_type, gt, gt_type)
    base_ids, x2_ids = _ids(base), _ids(x2)
    if not len(base_ids) or not len(x2_ids):
        return []
    _, bc = instance_classes(base, base_type)
    _, xc = instance_classes(x2, x2_type)
    _, gc = instance_classes(gt, gt_type)
    ov = overlap(base, x2)
    stable = (ov.iou > .5) & ((ov.inter > 0).sum(1)[:, None] == 1) & ((ov.inter > 0).sum(0)[None, :] == 1)
    ovgb, ovgx = overlap(gt, base), overlap(gt, x2)
    records = []
    for b, s in zip(*np.nonzero(stable)):
        b_cls, x_cls = int(bc[b]), int(xc[s])
        b_m, b_gt, _ = _gt_match(ovgb, gc, b)
        x_m, x_gt, _ = _gt_match(ovgx, gc, s)
        row = x2_rows.get(int(x2_ids[s])) if x2_rows is not None else None
        conf, _, margin = _confidence_fields(row, x_cls)
        records.append({"base_class": b_cls, "x2_class": x_cls,
                        "agree": b_cls == x_cls, "base_correct": b_m and b_gt == b_cls,
                        "x2_correct": x_m and x_gt == x_cls,
                        "x2_conf": conf, "x2_margin": margin})
    return records


def agreement_table(records):
    """Stable-pair confusion, correctness conditioned on agreement, disagreement attribution."""
    confusion = [[0] * 6 for _ in range(6)]
    conditioned = {"agree": {"n": 0, "base_correct": 0, "x2_correct": 0},
                   "disagree": {"n": 0, "base_correct": 0, "x2_correct": 0}}
    attribution = {"base_right_x2_wrong": 0, "base_wrong_x2_right": 0, "both_wrong": 0,
                   "base_right_x2_wrong_conf_sum": 0.0, "base_wrong_x2_right_conf_sum": 0.0}
    for r in records:
        confusion[r["base_class"]][r["x2_class"]] += 1
        key = "agree" if r["agree"] else "disagree"
        conditioned[key]["n"] += 1
        conditioned[key]["base_correct"] += int(bool(r["base_correct"]))
        conditioned[key]["x2_correct"] += int(bool(r["x2_correct"]))
        if not r["agree"]:
            if r["base_correct"] and not r["x2_correct"]:
                attribution["base_right_x2_wrong"] += 1
                attribution["base_right_x2_wrong_conf_sum"] += r["x2_conf"] or 0.0
            elif r["x2_correct"] and not r["base_correct"]:
                attribution["base_wrong_x2_right"] += 1
                attribution["base_wrong_x2_right_conf_sum"] += r["x2_conf"] or 0.0
            else:
                attribution["both_wrong"] += 1
    return {"stable_pairs": len(records), "confusion": confusion, "conditioned": conditioned,
            "disagreement_attribution": attribution}


def p2_added_records(base, base_type, x2, x2_type, rows, config, gt, gt_type):
    """Audit what one frozen P2 config actually added, and which candidates it rejected.

    `rows` must cover exactly the surviving x2 instance IDs (same rule as the fusion runner).
    Rejected candidates are disjoint, typed x2 objects the config filtered out; `reason`
    mirrors the fuse() skip order (max_area, then border, then min_prob).
    """
    _validate(base, base_type, x2, x2_type, gt, gt_type)
    if config["name"] == "identity":
        return [], []
    ids = _ids(x2)
    if rows is None or set(rows) != {int(i) for i in ids}:
        raise ValueError("probability rows do not exactly match this image's surviving instance IDs")
    confidence = {c["id"]: float(rows[c["id"]][c["class"]]) for c in candidate_info(x2, x2, x2_type)}
    _, _, chosen = fuse(base, base_type, x2, x2_type, confidence, config["max_area"],
                        config["min_prob"], config["interior_only"])
    chosen_set = set(chosen)
    ov = overlap(gt, x2)
    _, gc = instance_classes(gt, gt_type)
    positions = {int(i): j for j, i in enumerate(ids)}
    added, rejected = [], []
    for c in candidate_info(base, x2, x2_type):
        if not c["disjoint"] or c["class"] == 0:
            continue
        matched, gt_class, _ = _gt_match(ov, gc, positions[c["id"]])
        rec = {"id": c["id"], "class": c["class"], "area": c["area"], "border": c["border"],
               "conf": confidence[c["id"]], "matched": matched, "gt_class": gt_class,
               "typed_correct": (gt_class == c["class"]) if matched else None}
        if c["id"] in chosen_set:
            added.append(rec)
        else:
            if config["max_area"] is not None and c["area"] > config["max_area"]:
                reason = "max_area"
            elif config["interior_only"] and c["border"]:
                reason = "interior_only"
            elif confidence[c["id"]] < config["min_prob"]:
                reason = "min_prob"
            else:
                reason = "unclassified"
            rec["reason"] = reason
            rejected.append(rec)
    return added, rejected


def repro_sanity(cached_inst, cached_type, new_inst, new_type):
    """Agreement between a cached prediction and its re-prediction (greedy IoU matching)."""
    _validate(cached_inst, cached_type, new_inst, new_type)
    c_ids, n_ids = _ids(cached_inst), _ids(new_inst)
    both = (cached_inst > 0) & (new_inst > 0)
    type_agreement = (float((cached_type[both] == new_type[both]).mean())
                      if bool(both.any()) else None)
    result = {"n_cached": int(len(c_ids)), "n_new": int(len(n_ids)), "n_matched": 0,
              "iou_gt_05": 0.0, "iou_gt_09": 0.0, "iou_gt_099": 0.0, "median_iou": None,
              "type_agreement_matched": None, "foreground_type_agreement": type_agreement}
    if not len(c_ids) or not len(n_ids):
        return result
    ov = overlap(cached_inst, new_inst)
    _, c_cls = instance_classes(cached_inst, cached_type)
    _, n_cls = instance_classes(new_inst, new_type)
    pairs = sorted(((float(ov.iou[g, p]), g, p) for g, p in zip(*np.nonzero(ov.iou > 0))),
                   reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matched.append((iou, g, p))
    result["n_matched"] = len(matched)
    if not matched:
        return result
    ious = np.array([m[0] for m in matched])
    result["iou_gt_05"] = float((ious > .5).mean())
    result["iou_gt_09"] = float((ious > .9).mean())
    result["iou_gt_099"] = float((ious > .99).mean())
    result["median_iou"] = float(np.median(ious))
    strong = [m for m in matched if m[0] > .9]
    result["type_agreement_matched"] = (float(np.mean([c_cls[g] == n_cls[p] for _, g, p in strong]))
                                        if strong else None)
    return result
