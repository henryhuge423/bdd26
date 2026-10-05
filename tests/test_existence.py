import numpy as np
import pytest

from nucseg.postproc.existence import (candidate_geometry, distance_to_base, distance_to_base_map,
                                       existence_pass)


def test_existence_pass_matches_the_frozen_ef_menu_semantics():
    # a30: plain area floor at the boundary
    assert existence_pass({"area": 29, "class": 3}, min_area=30) is False
    assert existence_pass({"area": 30, "class": 3}, min_area=30) is True
    # a30d: Dead additions are exempt from the floor, non-Dead are not
    assert existence_pass({"area": 4, "class": 4}, min_area=30, dead_exempt=True) is True
    assert existence_pass({"area": 4, "class": 3}, min_area=30, dead_exempt=True) is False
    # off: no floor keeps everything
    assert existence_pass({"area": 1, "class": 1}, min_area=0) is True


def disk(centre, r, shape=(64, 64)):
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= r * r


def test_geometry_disk_is_round_sliver_is_not():
    d = candidate_geometry(disk((32, 32), 6))
    assert d["area"] == 113
    assert d["circularity"] >= 0.8
    assert d["extent"] >= 0.6  # rasterised disk fills ~2/3 of its bbox at this radius
    assert d["border"] is False
    sliver = np.zeros((64, 64), bool)
    sliver[np.arange(5, 25), np.arange(5, 25)] = True  # diagonal 1-px line: area 20, bbox 20x20
    s = candidate_geometry(sliver)
    assert s["area"] == 20
    assert s["circularity"] <= 0.65
    assert s["extent"] <= 0.1


def test_geometry_border_contact_and_single_pixel():
    edge = np.zeros((64, 64), bool)
    edge[0, 10:20] = True
    g = candidate_geometry(edge)
    assert g["border"] is True
    one = np.zeros((64, 64), bool)
    one[40, 40] = True
    p = candidate_geometry(one)
    assert p["area"] == 1 and 0.0 <= p["circularity"] <= 1.0


def test_distance_to_base_gaps_and_contact():
    base = np.zeros((64, 64), bool)
    base[:, :20] = True
    dist = distance_to_base_map(base)
    assert distance_to_base(disk((25, 32), 3), dist) == pytest.approx(3)  # leftmost pixel at x=22, base ends x=19
    assert distance_to_base(disk((18, 20), 3), dist) == 0.0  # centre inside the base band


def test_distance_map_error_on_empty_base():
    with pytest.raises(ValueError):
        distance_to_base_map(np.zeros((8, 8), bool))


@pytest.mark.parametrize("choice, expected_added", [("identity", 0), ("off", 1)])
def test_ef_apply_respects_identity_when_stage1_has_additions(choice, expected_added, monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_existence_ef as runner

    base = np.zeros((1, 16, 16), dtype=np.int32)
    candidate = base.copy()
    candidate[0, 5:9, 5:9] = 1
    types = candidate * 4
    runner._G.update(bi=base, bt=base.copy(), xi=candidate, xt=types,
                     prob={0: {1: np.array([0., 0., 0., 0., 1., 0.])}},
                     p2cfg={"max_area": 200, "min_prob": 0., "interior_only": False},
                     chosen={"name": choice})
    try:
        _, inst, typ, added = runner._apply(0)
        assert added == expected_added
        assert np.count_nonzero(inst) == 16 * expected_added
        assert np.count_nonzero(typ) == 16 * expected_added
    finally:
        runner._G.clear()


def test_existence_audit_border_bins_include_all_areas(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import audit_existence_candidates as audit

    rows = []
    for area, border, matched in [(1, True, False), (30, True, True), (80, False, True)]:
        rows.append(dict(pair="split1_seed19", area=area, border=border, matched=matched,
                         typed=matched, gt_already_by_base=False, extent=.7, circularity=.8,
                         dist=2., pmax=.9, margin=.8, image=0, id=len(rows)+1,
                         gt_class=4, added=True, iou_best=.8 if matched else 0., **{"class": 4}))
    # Isolate expensive prediction loading; exercise the real serialization/aggregation.
    monkeypatch.setattr(audit, "audit_pair", lambda *args: rows)
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(out), "--pairs", "split1_seed19"])
    audit.main()
    tables = json.loads((out / "summary.json").read_text())["tables"]
    assert [r["n"] for r in tables["added_by_border"]] == [2, 1]
    assert [r["matched"] for r in tables["added_by_border"]] == [1, 1]
    with np.load(out / "records.npz") as records:
        assert records["border"].tolist() == [True, True, False]
