"""Manifest and row-alignment guards for the validation-only scale probe."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_cellvit_scales.py"


def module():
    assert SCRIPT.exists(), "scale probe implementation is missing"
    spec = importlib.util.spec_from_file_location("scale_probe", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stratification_is_reproducible_unique_and_covers_tissues():
    m = module()
    tissue = np.array(["Lung"] * 20 + ["Colon"] * 12 + ["Uterus"] * 2)
    ids = m.stratified_indices(tissue, 12, 17)
    assert np.array_equal(ids, m.stratified_indices(tissue, 12, 17))
    assert len(ids) == len(set(ids)) == 12
    assert set(tissue[ids]) == set(tissue)
    assert np.all(np.diff(ids) > 0)


@pytest.mark.parametrize("ids", [[], [1, 1], [-1], [4], [1.5], [True]])
def test_manifest_rejects_invalid_indices(ids):
    with pytest.raises(ValueError):
        module().validate_indices(ids, 4)


def test_subset_preserves_original_manifest_order():
    m = module()
    f = SimpleNamespace(images=np.arange(24).reshape(4, 2, 3),
                        inst=np.arange(16).reshape(4, 2, 2),
                        type=np.ones((4, 2, 2), np.uint8),
                        tissue=np.array(["A", "B", "C", "D"]),
                        gt_channels=np.arange(32).reshape(4, 2, 2, 2))
    sub = m.IndexedFold(f, [3, 1])
    assert len(sub) == 2
    assert np.array_equal(sub.images, f.images[[3, 1]])
    assert np.array_equal(sub.inst, f.inst[[3, 1]])
    assert np.array_equal(sub.gt_channels, f.gt_channels[[3, 1]])
    assert sub.tissue.tolist() == ["D", "B"]


def test_manifest_checks_fold_tissue_and_does_not_overwrite(tmp_path):
    m = module()
    tissue = np.array(["A", "A", "B", "B"])
    p = tmp_path / "indices.json"
    m.make_manifest(p, tissue, 2, 7, fold=2)
    ids = m.read_manifest(p, tissue, fold=2)
    assert len(ids) == 2
    with pytest.raises(FileExistsError):
        m.make_manifest(p, tissue, 2, 7, fold=2)
    with pytest.raises(ValueError):
        m.read_manifest(p, tissue, fold=1)
    with pytest.raises(ValueError):
        m.read_manifest(p, tissue[::-1], fold=2)


def test_manifest_insufficient_budget_is_rejected():
    with pytest.raises(ValueError):
        module().stratified_indices(np.array(["A", "B", "C"]), 2, 0)


def test_saved_rows_must_match_manifest_exactly():
    m = module()
    m.validate_prediction_rows(np.array([3, 1]), np.array([3, 1]), 2)
    with pytest.raises(ValueError):
        m.validate_prediction_rows(np.array([1, 3]), np.array([3, 1]), 2)
    with pytest.raises(ValueError):
        m.validate_prediction_rows(np.array([3, 1]), np.array([3, 1]), 1)


def test_legacy_map_dump_rejects_x2_before_model_loading(tmp_path):
    import os
    import subprocess
    import sys
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"upscale": 2}))
    command = [sys.executable, str(SCRIPT.with_name("dump_cellvit_maps.py")),
               "--run", str(run), "--fold", "2", "--out", str(tmp_path / "maps")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0
    assert "native-scale checkpoints only" in result.stderr
    assert not (tmp_path / "maps").exists()
