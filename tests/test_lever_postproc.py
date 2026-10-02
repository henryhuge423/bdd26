"""Regression tests for the actual M/S geometry and class-ID contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "lever_postproc", Path(__file__).parents[1] / "scripts" / "lever_postproc.py")
lever = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lever)


def adjacent(dtype=np.int32, ids=(1, 2)):
    inst = np.zeros((6, 8), dtype)
    inst[2:4, 2:4] = ids[0]
    inst[2:4, 4:6] = ids[1]
    return inst


@pytest.mark.parametrize("dtype", [np.int32, np.uint16])
def test_background_does_not_inflate_last_instance_perimeter(dtype):
    assert lever.image_geometry(adjacent(dtype))["per"].tolist() == [8, 8]


def test_outer_image_edges_count_toward_perimeter():
    inst = np.ones((2, 3), np.int32)
    geo = lever.image_geometry(inst)
    assert geo["per"].tolist() == [10]
    assert geo["areas"].tolist() == [6]
    assert geo["border"].tolist() == [True]


@pytest.mark.parametrize("ids", [(1, 2), (2, 7)])
def test_same_class_lookup_uses_actual_instance_id(ids):
    inst = adjacent(ids=ids)
    classes = np.zeros(max(ids) + 1, np.int64)
    classes[list(ids)] = 4
    out = lever.apply_levers(inst, lever.image_geometry(inst), 0, 0, classes)
    assert np.unique(out[out > 0]).tolist() == [1]
    assert np.array_equal(out > 0, inst > 0)


def test_different_classes_do_not_merge_when_previous_classes_agree():
    inst = np.zeros((6, 11), np.int32)
    inst[2:4, 1:3] = 1
    inst[2:4, 5:7] = 2
    inst[2:4, 7:9] = 3
    out = lever.apply_levers(inst, lever.image_geometry(inst), 0, 0, np.array([0, 1, 1, 2]))
    assert out[2, 5] != out[2, 7]


def test_shared_boundary_fraction_uses_true_perimeter():
    inst = adjacent()
    classes = np.array([0, 1, 1])
    geo = lever.image_geometry(inst)
    assert geo["pairs"] == {(1, 2): 2}
    out = lever.apply_levers(inst, geo, .25, 0, classes)
    assert out[2, 2] == out[2, 4]
    out = lever.apply_levers(inst, geo, .26, 0, classes)
    assert out[2, 2] != out[2, 4]


@pytest.mark.parametrize("dtype", [np.int32, np.uint16])
def test_bent_shared_boundary_accumulates_both_directions(dtype):
    inst = np.pad(np.array([[1, 1, 0], [1, 2, 2], [0, 2, 0]], dtype=dtype), 1)
    geo = lever.image_geometry(inst)
    assert geo["per"].tolist() == [8, 8]
    assert geo["pairs"] == {(1, 2): 2}
    out = lever.apply_levers(inst, geo, .25, 0, np.array([0, 4, 4]))
    assert np.unique(out[out > 0]).tolist() == [1]


def test_shared_boundary_and_merging_are_transpose_invariant():
    inst = np.pad(np.array([[1, 1, 1], [1, 2, 2], [0, 2, 0]], np.int32), 1)
    geo, transposed = lever.image_geometry(inst), lever.image_geometry(inst.T)
    assert geo["pairs"] == transposed["pairs"] == {(1, 2): 3}
    cls = np.array([0, 4, 4])
    assert np.array_equal(lever.apply_levers(inst, geo, .25, 0, cls),
                          lever.apply_levers(inst.T, transposed, .25, 0, cls).T)


def test_sweep_preserves_overlapping_official_gt_channels():
    from types import SimpleNamespace
    from nucseg.metrics.pannuke_eval import evaluate
    gt = np.zeros((1, 6, 6, 5), np.uint16)
    gt[0, 1:3, 1:3, 0] = 1
    gt[0, 2:4, 2:4, 1] = 1
    inst = np.zeros((1, 6, 6), np.int32)
    inst[gt[..., 0] > 0] = 1
    inst[gt[..., 1] > 0] = 2
    typ = inst.astype(np.uint8)
    tissue = np.array(["Lung"])
    fold = SimpleNamespace(inst=inst, type=typ, tissue=tissue, gt_channels=gt)
    official = evaluate(gt, inst, typ, tissue, inst, typ, workers=1)["summary"]["official"]
    assert official["mPQ"] == pytest.approx(.875)
    row = lever.sweep_predictions(inst, typ, fold, [(None, 0)], workers=1)[0]
    assert row["mPQ"] == pytest.approx(official["mPQ"], abs=1e-12)
    assert row["bPQ"] == pytest.approx(official["bPQ"], abs=1e-12)


def test_drop_only_small_border_instances():
    inst = np.zeros((6, 8), np.int32)
    inst[0, 1:3] = 1
    inst[2, 2:4] = 2
    inst[5, 4:8] = 3
    out = lever.apply_levers(inst, lever.image_geometry(inst), None, 3, np.array([0, 1, 1, 1]))
    assert not out[0].any()
    assert out[2, 2] > 0
    assert out[5, 4] > 0


def test_noop_preserves_partition_with_sparse_ids():
    inst = adjacent(ids=(2, 7))
    out = lever.apply_levers(inst, lever.image_geometry(inst), None, 0,
                             np.array([0, 0, 1, 0, 0, 0, 0, 2]))
    assert np.array_equal(out == 1, inst == 2)
    assert np.array_equal(out == 2, inst == 7)


def test_background_only_image():
    inst = np.zeros((4, 4), np.int32)
    geo = lever.image_geometry(inst)
    out = lever.apply_levers(inst, geo, .25, 40, np.zeros(1, np.int64))
    assert geo["n"] == 0
    assert not out.any()


@pytest.mark.parametrize("inst", [np.array([[0, -1]]), np.zeros((2, 2, 2), int),
                                   np.array([[0., 1.]]), np.zeros((0, 2), int)])
def test_invalid_instance_maps_rejected(inst):
    with pytest.raises(ValueError):
        lever.image_geometry(inst)


@pytest.mark.parametrize("frac,a_min", [(float("nan"), 0), (-.1, 0),
                                         (None, -1), (0, float("inf"))])
def test_invalid_thresholds_rejected(frac, a_min):
    inst = adjacent()
    with pytest.raises(ValueError):
        lever.apply_levers(inst, lever.image_geometry(inst), frac, a_min, np.array([0, 1, 1]))


def test_class_lut_without_background_slot_rejected():
    inst = adjacent()
    with pytest.raises(ValueError):
        lever.apply_levers(inst, lever.image_geometry(inst), .25, 0, np.array([1, 1]))


def test_transformed_payload_drops_stale_instance_tables_and_slices_image_metadata():
    inst = np.stack([adjacent(), adjacent()])
    typ = np.where(inst > 0, 1, 0).astype(np.uint8)
    original = dict(inst=inst, type=typ, inst_img=np.array([0, 0, 1, 1]),
                    inst_id=np.array([1, 2, 1, 2]), inst_prob=np.ones((4, 6)),
                    tissue_prob=np.ones((2, 19)), image_indices=np.array([7, 9]))
    transformed = np.zeros_like(inst[:1])
    payload = lever.prediction_payload(original, transformed)
    assert set(payload) == {"inst", "type", "tissue_prob", "image_indices"}
    assert payload["type"].shape == (1, 6, 8)
    assert not payload["type"].any()
    assert payload["tissue_prob"].shape == (1, 19)
    assert payload["image_indices"].tolist() == [7]


def test_apply_cli_refuses_overwrite_and_records_provenance(tmp_path):
    import json
    import os
    import subprocess
    import sys
    inst = adjacent()[None]
    pred = tmp_path / "input.npz"
    np.savez_compressed(pred, inst=inst, type=np.where(inst > 0, 1, 0).astype(np.uint8),
                        inst_prob=np.ones((2, 6)), inst_id=np.array([1, 2]), inst_img=np.array([0, 0]))
    out = tmp_path / "output.npz"
    command = [sys.executable, str(Path(lever.__file__)), "--pred", str(pred), "--fold", "2",
               "--mode", "apply", "--frac", "0", "--a-min", "0", "--workers", "1", "--out", str(out)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    first = subprocess.run(command, capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    with np.load(out) as result:
        assert "inst_prob" not in result
        assert np.unique(result["inst"][result["inst"] > 0]).tolist() == [1]
    meta = json.loads(out.with_suffix(".npz.json").read_text())
    assert meta["input"]["sha256"]
    assert meta["code"]["files"]
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    assert second.returncode != 0
