#!/usr/bin/env python
"""Descriptive typing-confidence audit on cached predictions (P0-P3 v2 follow-up).

Per split and role this records, without selecting anything:
  - calibration of per-instance predicted-class confidence/margin against actual typing
    correctness for the x2/du2 arm and, when the input carries probability rows, the raw x1 arm;
  - what the frozen P2 rule added and rejected on this fold (deployment reference = P0-selected
    x1+M/S base, candidates = original du2 rows), plus an unfiltered disjoint-candidate anchor
    against the raw x1 map for comparison with the P1 counters;
  - x1/x2 type agreement on stable one-to-one pairs with correctness conditioning;
  - optional re-prediction sanity (cached raw x1 vs re-predicted maps).

    python scripts/analyze_typing_confidence.py --split 1 --role val \
        --p0-base-ms runs/analysis/p0p3_20261002_v2/p0/split1/val/base/ms/pred.npz \
        --x2 runs/analysis/p0p3_20261002/inputs/x2_split1/pred_fold2_x2_du2.npz \
        [--x1 runs/cellvit_uni/split1/pred_fold2.npz] [--x1-ref CACHED_X1.npz] \
        [--p2-selection runs/analysis/p0p3_20261002_v2/p2/split1/selection.json] \
        --out runs/analysis/typing_audit_20261003/split1_val.json
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lever_postproc import file_provenance, write_json  # noqa: E402
from run_p0_controls import check_prediction, ground_truth_provenance  # noqa: E402
from nucseg.data.pannuke import PanNukeFold, split_folds  # noqa: E402
from nucseg.postproc.scale_fusion import probability_rows  # noqa: E402
from nucseg.postproc.typing_audit import (  # noqa: E402
    agreement_records, agreement_table, confidence_table, instance_records, p2_added_records,
    repro_sanity, _bin)

UNFILTERED = {"name": "unfiltered", "max_area": None, "min_prob": 0., "interior_only": False}
_G = {}


def class_summary(records):
    out = {str(c): {"n": 0, "matched": 0, "correct": 0, "conf_sum": 0.0, "n_conf": 0}
           for c in range(6)}
    for r in records:
        cell = out[str(r["class"])]
        cell["n"] += 1
        cell["matched"] += int(bool(r["matched"]))
        cell["correct"] += int(bool(r["typed_correct"]))
        if r["conf"] is not None:
            cell["conf_sum"] += r["conf"]
            cell["n_conf"] += 1
    return out


def candidate_summary(records):
    by_class = {str(c): {"n": 0, "matched": 0, "correct": 0} for c in range(6)}
    by_conf = {}
    reasons = {}
    for r in records:
        cell = by_class[str(r["class"])]
        cell["n"] += 1
        cell["matched"] += int(bool(r["matched"]))
        cell["correct"] += int(bool(r["typed_correct"]))
        b = _bin(r["conf"])
        by_conf.setdefault(b, {"n": 0, "matched": 0, "correct": 0})
        by_conf[b]["n"] += 1
        by_conf[b]["matched"] += int(bool(r["matched"]))
        by_conf[b]["correct"] += int(bool(r["typed_correct"]))
        if "reason" in r:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"by_class": by_class, "by_conf": by_conf, "reasons": reasons, "n": len(records)}


def _one(j):
    rows = _G["rows"].get(j, {})
    out = {"x2_records": instance_records(_G["xi"][j], _G["xt"][j], _G["gi"][j], _G["gt"][j], rows),
           "agreement": agreement_records(_G["bi"][j], _G["bt"][j], _G["xi"][j], _G["xt"][j],
                                          _G["gi"][j], _G["gt"][j], rows)}
    if _G.get("cfg") is not None:
        added, rejected = p2_added_records(_G["bi"][j], _G["bt"][j], _G["xi"][j], _G["xt"][j],
                                           rows, _G["cfg"], _G["gi"][j], _G["gt"][j])
        out["p2_added"], out["p2_rejected"] = added, rejected
    if _G.get("x1") is not None:
        out["x1_records"] = instance_records(_G["x1"][j], _G["x1t"][j], _G["gi"][j], _G["gt"][j],
                                             (_G["x1rows"] or {}).get(j, {}))
        out["anchor_added"], _ = p2_added_records(_G["x1"][j], _G["x1t"][j], _G["xi"][j],
                                                  _G["xt"][j], rows, UNFILTERED,
                                                  _G["gi"][j], _G["gt"][j])
    if _G.get("x1ref") is not None:
        out["sanity"] = repro_sanity(_G["x1ref"][j], _G["x1reft"][j], _G["x1"][j], _G["x1t"][j])
    return out


def load_maps(path, fold, label):
    z = np.load(path, allow_pickle=True)
    inst, typ = np.asarray(z["inst"]), np.asarray(z["type"])
    check_prediction(inst, typ, fold, z)
    rows = probability_rows(z) if "inst_prob" in z.files else None
    prov = {"path": str(path), **file_provenance(path)}
    manifest = Path(path).parent / "manifest.json"
    if manifest.exists():
        import json
        d = json.loads(manifest.read_text())
        if d.get("prediction", {}).get("sha256") != prov["sha256"]:
            raise ValueError(f"{label}: prediction hash does not match its provenance sidecar")
        prov["manifest"] = file_provenance(manifest)
    return inst, typ, rows, prov


def concat(values, key):
    return [r for v in values for r in v[key]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--role", choices=("val", "test"), required=True)
    p.add_argument("--p0-base-ms", type=Path, required=True)
    p.add_argument("--x2", type=Path, required=True)
    p.add_argument("--x1", type=Path, default=None,
                   help="raw x1 prediction with probability rows (calibration arm, anchor base)")
    p.add_argument("--x1-ref", type=Path, default=None,
                   help="cached raw x1 maps to sanity-check the --x1 re-prediction against")
    p.add_argument("--p2-selection", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite existing audit output {a.out}")
    if a.x1_ref is not None and a.x1 is None:
        raise SystemExit("--x1-ref requires --x1 (the re-predicted raw x1 maps)")

    import json
    _, val_fold, test_fold = split_folds(a.split)
    fold_id = val_fold if a.role == "val" else test_fold
    fold = PanNukeFold(fold_id)
    _G["gi"], _G["gt"] = fold.inst, fold.type
    _G["bi"], _G["bt"], _, p0prov = load_maps(a.p0_base_ms, fold, "p0-base-ms")
    _G["xi"], _G["xt"], _G["rows"], x2prov = load_maps(a.x2, fold, "x2")
    if _G["rows"] is None:
        raise SystemExit("the x2 input must carry per-instance probability rows")
    prov = {"p0_base_ms": p0prov, "x2": x2prov}
    if a.x1 is not None:
        _G["x1"], _G["x1t"], _G["x1rows"], prov["x1"] = load_maps(a.x1, fold, "x1")
    if a.x1_ref is not None:
        _G["x1ref"], _G["x1reft"], _, prov["x1_ref"] = load_maps(a.x1_ref, fold, "x1-ref")
    _G["cfg"] = None
    if a.p2_selection is not None:
        sel = json.loads(a.p2_selection.read_text())
        if sel["split"] != a.split or sel["fold"] != val_fold:
            raise SystemExit("selection.json split/val-fold does not match this invocation")
        _G["cfg"] = sel["selected"]
        prov["p2_selection"] = {"path": str(a.p2_selection), **file_provenance(a.p2_selection),
                                "selected": sel["selected"], "status": sel["status"]}

    with Pool(a.workers) as pool:
        values = pool.map(_one, range(len(_G["gi"])))

    result = {"split": a.split, "fold": fold_id, "role": a.role,
              "scope": "descriptive audit; no threshold selected on any fold",
              "x2_calibration": confidence_table(concat(values, "x2_records")),
              "x2_class_summary": class_summary(concat(values, "x2_records")),
              "agreement": agreement_table(concat(values, "agreement"))}
    if "x1_records" in values[0]:
        result["x1_calibration"] = confidence_table(concat(values, "x1_records"))
        result["x1_class_summary"] = class_summary(concat(values, "x1_records"))
        result["anchor_unfiltered_vs_raw_x1"] = candidate_summary(concat(values, "anchor_added"))
    if "p2_added" in values[0]:
        result["p2_added"] = candidate_summary(concat(values, "p2_added"))
        result["p2_rejected"] = candidate_summary(concat(values, "p2_rejected"))
    if "sanity" in values[0]:
        keys = ("n_cached", "n_new", "n_matched", "iou_gt_05", "iou_gt_09", "iou_gt_099",
                "type_agreement_matched", "foreground_type_agreement")
        agg = {}
        for k in keys:
            col = [v["sanity"][k] for v in values]
            if isinstance(col[0], (int, np.integer)) and not isinstance(col[0], bool):
                agg[k] = int(sum(col))
            else:
                nums = [np.nan if x is None else x for x in col]
                agg[k] = float(np.nanmean(nums)) if not np.all(np.isnan(nums)) else None
        worst = sorted(values, key=lambda v: v["sanity"]["iou_gt_099"])[:5]
        result["repro_sanity"] = {"aggregate": agg,
                                  "worst_images_by_iou099": [v["sanity"] for v in worst]}
    result["provenance"] = {"inputs": prov,
                            "ground_truth": ground_truth_provenance(fold),
                            "script": file_provenance(Path(__file__))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
