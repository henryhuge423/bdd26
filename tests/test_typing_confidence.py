"""Typing-confidence audit must stay descriptive: records and counters only, no selection."""
import importlib
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1] / "src/nucseg/postproc/typing_audit.py"


def module():
    assert FILE.exists(), "typing audit implementation missing"
    return importlib.import_module("nucseg.postproc.typing_audit")


def scene():
    """GT: two nuclei (classes 1 and 4). base: both found (types 1 and 4). x2: adds one disjoint
    class-4 candidate plus one overlapping the second GT with the wrong class."""
    gt = np.zeros((14, 14), np.int32)
    gt[1:4, 1:4] = 1   # class 1
    gt[6:10, 6:10] = 2  # class 4 (Dead)
    gt_type = np.where(gt == 1, 1, 4).astype(np.uint8)
    base = gt.copy()
    base_type = gt_type.copy()
    x2 = gt.copy()
    x2[11:13, 11:13] = 9  # disjoint candidate, class 4
    x2_type = np.where(x2 == 9, 4, gt_type).astype(np.uint8)
    # per-instance probability rows: class-4 rows confident, class-1 row less so
    rows = {1: np.array([.05, .55, .1, .1, .15, .05]), 2: np.array([.02, .05, .03, .05, .92, .05]),
            9: np.array([.01, .04, .02, .03, .9, 0.])}
    return gt, gt_type, base, base_type, x2, x2_type, rows


def test_instance_records_match_typing_and_confidence():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    recs = {r["id"]: r for r in module().instance_records(x2, x2_type, gt, gt_type, rows)}
    assert set(recs) == {1, 2, 9}
    r1, r2, r9 = recs[1], recs[2], recs[9]
    assert r1["class"] == 1 and r1["matched"] and r1["typed_correct"]
    assert r1["conf"] == pytest.approx(.55) and r1["row_argmax"] == 1
    assert r1["margin"] == pytest.approx(.55 - .15)
    assert r2["class"] == 4 and r2["matched"] and r2["typed_correct"]
    assert r9["matched"] is False and r9["typed_correct"] is None and r9["conf"] == pytest.approx(.9)
    assert r9["area"] == 4 and r9["border"] is False


def test_instance_records_flag_map_vs_row_argmax_disagreement():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    rows = dict(rows)
    rows[9] = np.array([.02, .5, .1, .1, .28, 0.])  # row argmax 1, map majority 4
    recs = {r["id"]: r for r in module().instance_records(x2, x2_type, gt, gt_type, rows)}
    assert recs[9]["row_argmax"] == 1 and recs[9]["class"] == 4
    assert recs[9]["conf"] == pytest.approx(.28)


def test_instance_records_without_rows_leaves_confidence_null():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    recs = module().instance_records(x2, x2_type, gt, gt_type, None)
    assert all(r["conf"] is None and r["margin"] is None and r["row_argmax"] is None for r in recs)
    assert all(r["matched"] in (True, False) for r in recs)


def test_confidence_table_counts_bins_classes_and_correctness():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    recs = module().instance_records(x2, x2_type, gt, gt_type, rows)
    tbl = module().confidence_table(recs)
    b4 = tbl["conf_bins"]["0.9-1.0"]["4"]
    assert b4["n"] == 2 and b4["matched"] == 1 and b4["correct"] == 1
    b1 = tbl["conf_bins"]["0.5-0.6"]["1"]
    assert b1["n"] == 1 and b1["matched"] == 1 and b1["correct"] == 1
    assert tbl["row_argmax_agreement"]["agree"] == 3 and tbl["row_argmax_agreement"]["disagree"] == 0


def test_agreement_confined_to_stable_pairs_with_conditioned_correctness():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    # make x2's second object disagree on type: retype GT-2 region as class 1 in x2
    x2d = x2.copy()
    x2d_type = x2_type.copy()
    x2d_type[x2d == 2] = 1
    x2d[6:8, 6:8] = 7  # fragment overlapping GT-2 -> GT-2 pair no longer one-to-one on x2 side? no:
    # object 7 overlaps base object 2 only; base side stays unique; x2 side: GT-2 region now two objects
    x2d_type[x2d == 7] = 4
    recs = module().agreement_records(base, base_type, x2d, x2d_type, gt, gt_type, rows)
    pairs = {(r["base_class"], r["x2_class"]) for r in recs}
    # object 2 (unstable: shares base-2's GT region with fragment 7 on neither side? base 2 overlaps
    # x2 objects 2 and 7 -> not stable); candidate 9 does not overlap base at all; only pair (1,1) stable
    assert pairs == {(1, 1)}
    tbl = module().agreement_table(recs)
    assert tbl["stable_pairs"] == 1 and tbl["confusion"][1][1] == 1
    assert tbl["conditioned"]["agree"]["n"] == 1


def test_p2_added_audit_reports_chosen_and_rejected():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    m = module()
    conf = {i: float(rows[i][c_]) for i, c_ in ((1, 1), (2, 4), (9, 4))}
    cfg = {"name": "a200_p0.85", "max_area": 200, "min_prob": .85, "interior_only": False}
    added, rejected = m.p2_added_records(base, base_type, x2, x2_type, rows, cfg, gt, gt_type)
    assert [r["id"] for r in added] == [9]
    assert added[0]["class"] == 4 and added[0]["conf"] == pytest.approx(.9)
    assert added[0]["matched"] is False and added[0]["typed_correct"] is None
    low = {"name": "a200_p0.95", "max_area": 200, "min_prob": .95, "interior_only": False}
    added2, rejected2 = m.p2_added_records(base, base_type, x2, x2_type, rows, low, gt, gt_type)
    assert added2 == [] and len(rejected2) == 1 and rejected2[0]["reason"] == "min_prob"
    identity = {"name": "identity", "max_area": None, "min_prob": 0., "interior_only": False}
    added3, rejected3 = m.p2_added_records(base, base_type, x2, x2_type, rows, identity, gt, gt_type)
    assert added3 == [] and rejected3 == []


def test_repro_sanity_identical_and_perturbed_maps():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    m = module()
    same = m.repro_sanity(base, base_type, base.copy(), base_type.copy())
    assert same["n_cached"] == same["n_new"] == 2 and same["iou_gt_099"] == 1.0
    assert same["type_agreement_matched"] == 1.0
    new = base.copy()
    new[6:8, 6:8] = 0  # erode one object
    pert = m.repro_sanity(base, base_type, new, base_type.copy())
    assert pert["iou_gt_099"] < 1.0 and 0.0 < pert["iou_gt_05"] <= 1.0


def test_empty_images_are_safe():
    m = module()
    z = np.zeros((8, 8), np.int32)
    tz = np.zeros((8, 8), np.uint8)
    assert m.instance_records(z, tz, z, tz, None) == []
    assert m.agreement_records(z, tz, z, tz, z, tz, None) == []
    added, rejected = m.p2_added_records(z, tz, z, tz, {}, {"name": "identity", "max_area": None,
                                                           "min_prob": 0., "interior_only": False},
                                         z, tz)
    assert added == [] and rejected == []
    s = m.repro_sanity(z, tz, z, tz)
    assert s["n_cached"] == 0 and s["n_new"] == 0


def test_empty_gt_with_predictions_is_unmatched_not_crash():
    gt, gt_type, base, base_type, x2, x2_type, rows = scene()
    z = np.zeros_like(gt)
    tz = np.zeros_like(gt_type)
    recs = module().instance_records(x2, x2_type, z, tz, rows)
    assert len(recs) == 3 and all(r["matched"] is False and r["typed_correct"] is None for r in recs)
    agr = module().agreement_records(base, base_type, x2, x2_type, z, tz, rows)
    assert agr and all(r["base_correct"] is False and r["x2_correct"] is False for r in agr)
    added, rejected = module().p2_added_records(base, base_type, x2, x2_type, rows,
                                                {"name": "a200_p0.85", "max_area": 200,
                                                 "min_prob": .85, "interior_only": False},
                                                z, tz)
    assert [r["id"] for r in added] == [9] and added[0]["matched"] is False
