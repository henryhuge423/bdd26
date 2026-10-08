"""Montage grid and stratified sampling for the DSB sonnet-review protocol (plan Task 7)."""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dsb_montage.py"
    spec = importlib.util.spec_from_file_location("dsb_montage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m = _load()
build_grid, outline, stratify = _m.build_grid, _m.outline, _m.stratify


def test_build_grid_layout_deterministic():
    patches = [np.full((8, 8, 3), i, np.uint8) for i in range(5)]
    grid = build_grid(patches, cols=2)
    assert grid.shape == (3 * 8, 2 * 8, 3)          # ceil(5/2) rows x 2 cols
    assert (grid[:8, :8] == 0).all() and (grid[8:16, 8:16] == 3).all()
    assert not grid[16:24, 8:16].any()              # trailing empty cell is blank


def test_stratify_exact_k_proportional_and_seeded():
    records = ([{"tissue": "Colon", "border": False, "area_bin": 0}] * 60 +
               [{"tissue": "Uterus", "border": True, "area_bin": 1}] * 40)
    keys = ("tissue", "border", "area_bin")
    idx = stratify(records, keys=keys, k=10, seed=7)
    assert len(idx) == len(set(idx)) == 10
    c = Counter(records[i]["tissue"] for i in idx)
    assert abs(c["Colon"] - 6) <= 1 and abs(c["Uterus"] - 4) <= 1
    assert stratify(records, keys=keys, k=10, seed=7) == idx   # seeded determinism


def test_stratify_rare_stratum_survives():
    records = ([{"tissue": "Colon", "border": False, "area_bin": 0}] * 99 +
               [{"tissue": "Uterus", "border": True, "area_bin": 1}])
    idx = stratify(records, keys=("tissue", "border", "area_bin"), k=10, seed=3)
    assert len(idx) == 10
    assert any(records[i]["tissue"] == "Uterus" for i in idx)  # largest remainder keeps it


def test_outline_marks_mask_border_only():
    img = np.zeros((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), bool)
    mask[4:10, 4:10] = True
    out = outline(img, mask, color=(0, 165, 255))
    assert out.shape == img.shape
    touched = (out[..., 0] == 0) & (out[..., 1] == 165) & (out[..., 2] == 255)
    assert touched[4, 4] and touched[9, 9]          # border ring colored
    assert not touched[6, 6]                        # interior untouched
    assert not touched[0, 0]                        # outside untouched
