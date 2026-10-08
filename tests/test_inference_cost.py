"""Pure-logic tests for the reduced-E0 inference-cost script (2026-10-08)."""

import importlib.util
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "inference_cost.py"
    spec = importlib.util.spec_from_file_location("inference_cost", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dist_ms():
    m = _load()
    d = m._dist_ms([10.0, 30.0])
    assert abs(d["mean_ms"] - 20.0) < 1e-9
    assert abs(d["p50_ms"] - 20.0) < 1e-9
    assert abs(d["p95_ms"] - (10 + 0.95 * 20)) < 1e-9


def test_chunk_bounds():
    m = _load()
    assert m.chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert m.chunk_bounds(8, 4) == [(0, 4), (4, 8)]
    assert m.chunk_bounds(0, 4) == []
    assert m.chunk_bounds(3, 4) == [(0, 3)]


def test_per_image_summary():
    m = _load()
    s = m.per_image_summary([(1.0, 100), (3.0, 100)])
    assert abs(s["mean_ms"] - 20.0) < 1e-9
    assert abs(s["p50_ms"] - 20.0) < 1e-9
    assert abs(s["p95_ms"] - (10 + 0.95 * 20)) < 1e-9     # linear percentile over {10, 30} ms
    assert abs(s["img_per_s"] - 200 / 4.0) < 1e-9


def test_frozen_ef_config():
    m = _load()
    p2_sel = {"selected": {"name": "a200_p0.9_interior0", "max_area": 200,
                           "min_prob": 0.9, "interior_only": False}}
    ef_sel = {"selected": {"name": "a30", "min_area": 30},
              "p2_selection": {"selected_name": "a200_p0.9_interior0", "sha256": "x"}}
    cfg = m.frozen_ef_config(p2_sel, ef_sel)
    assert cfg["p2"] == {"max_area": 200, "min_prob": 0.9, "interior_only": False}
    assert cfg["ef"] == {"min_area": 30, "dead_exempt": False}
    identity = {"selected": {"name": "identity"}}
    assert m.frozen_ef_config(identity, ef_sel)["p2"] is None


def test_ef_apply_matches_deployment_filters():
    m = _load()
    base = np.zeros((16, 16), np.int32); base[0:8, 0:8] = 1          # one base instance
    base_type = np.where(base > 0, 2, 0).astype(np.uint8)
    x2 = np.zeros((16, 16), np.int32)
    x2[0, 10:15] = 1                                                 # 5px candidate -> below a30
    x2[10:15, 0:5] = 2                                               # 25px candidate -> below a30
    x2[9:16, 9:16] = 3                                               # 49px disjoint candidate -> kept
    x2[0:6, 0:6] = 4                                                 # 36px but overlaps base -> never disjoint
    x2_type = np.where(x2 > 0, 2, 0).astype(np.uint8)
    rows = {i: np.full(6, 0.02) for i in (1, 2, 3, 4)}
    for i in rows:
        rows[i][2] = 0.9
    p2 = {"max_area": 200, "min_prob": 0.5, "interior_only": False}
    ef = {"min_area": 30, "dead_exempt": False}
    inst, typ, added = m.ef_apply(base, base_type, x2, x2_type, rows, p2, ef)
    assert added == [3]                                              # only the 49px disjoint candidate
    assert set(np.unique(inst)) == {0, 1, 2}
    assert set(typ[inst == 2].tolist()) == {2}
