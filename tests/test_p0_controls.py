"""P0 frozen validation selection and end-to-end artifact contract."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_p0_controls.py"


def runner():
    spec = importlib.util.spec_from_file_location("run_p0_controls", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(frac, area, mpq, bpq):
    return dict(frac=frac, a_min=area, mPQ=mpq, bPQ=bpq)


def test_selection_respects_bpq_noninferiority():
    rows = [row(None, 0, .50, .66), row(.25, 40, .51, .65), row(.5, 20, .505, .659)]
    selected = runner().select_config(rows)
    assert selected["frac"] == .5
    assert selected["a_min"] == 20


def test_selection_ties_prefer_identity_independent_of_menu_order():
    rows = [row(.25, 40, .50, .66), row(None, 0, .50, .66)]
    assert runner().select_config(rows)["frac"] is None
    assert runner().select_config(rows[::-1])["a_min"] == 0


def test_selection_rejects_missing_identity_and_nonfinite_baseline():
    with pytest.raises(ValueError):
        runner().select_config([row(.25, 40, .51, .66)])
    with pytest.raises(ValueError):
        runner().select_config([row(None, 0, float("nan"), .66)])


def make_inputs(tmp_path):
    data = tmp_path / "data"
    inst = np.zeros((2, 8, 8), np.int32)
    inst[:, 2:4, 2:4] = 3
    typ = np.where(inst > 0, 1, 0).astype(np.uint8)
    gt = np.zeros((2, 8, 8, 5), np.uint16)
    gt[..., 0] = inst
    for fold in (2, 3):
        d = data / f"fold{fold}"
        d.mkdir(parents=True)
        np.save(d / "inst.npy", inst)
        np.save(d / "type.npy", typ)
        np.save(d / "tissue.npy", np.array(["Lung", "Colon"]))
        np.savez_compressed(d / "gt_channels.npz", gt=gt)
    paths = {}
    for arm, scale in (("base", 1), ("x2", 2)):
        d = tmp_path / arm
        d.mkdir()
        (d / "config.json").write_text(json.dumps(dict(split=1, seed=19, upscale=scale)))
        paths[f"{arm}-run"] = d
        for role in ("val", "test"):
            p = d / f"{role}.npz"
            np.savez_compressed(p, inst=inst, type=typ,
                                inst_prob=np.ones((2, 6)), inst_id=np.array([3, 3]),
                                inst_img=np.array([0, 1]))
            paths[f"{arm}-{role}"] = p
    return data, paths


def command(paths, out, stage):
    cmd = [sys.executable, str(SCRIPT), "--split", "1", "--workers", "1", "--out", str(out), "--stage", stage]
    for key, value in paths.items():
        cmd += ["--" + key, str(value)]
    return cmd


def test_runner_freezes_validation_before_test_and_keeps_historical_inputs(tmp_path):
    data, paths = make_inputs(tmp_path)
    env = dict(os.environ, PANNUKE_ROOT=str(data), CUDA_VISIBLE_DEVICES="")
    out = tmp_path / "results"
    val = subprocess.run(command(paths, out, "val"), capture_output=True, text=True, env=env)
    assert val.returncode == 0, val.stderr
    selection_path = out / "selection.json"
    frozen = selection_path.read_bytes()
    selection = json.loads(frozen)
    assert selection["split"] == 1
    assert selection["validation_fold"] == 2
    assert selection["arms"]["base"]["selected"]["frac"] is None
    assert selection["arms"]["base"]["actual_seed"] == 19
    assert not (out / "test").exists()
    test = subprocess.run(command(paths, out, "test"), capture_output=True, text=True, env=env)
    assert test.returncode == 0, test.stderr
    assert selection_path.read_bytes() == frozen
    for arm in ("base", "x2"):
        for variant in ("identity", "ms"):
            summary = json.loads((out / "test" / arm / variant / "eval" / "summary.json").read_text())
            assert summary["official"]["mPQ"] == pytest.approx(1., abs=1e-6)
            meta = json.loads((out / "test" / arm / variant / "manifest.json").read_text())
            assert meta["role"] == "test"
            assert meta["fold"] == 3
            assert meta["selection_sha256"]
    with np.load(paths["base-test"]) as original:
        assert "inst_prob" in original
        assert int(original["inst"].max()) == 3
    again = subprocess.run(command(paths, out, "test"), capture_output=True, text=True, env=env)
    assert again.returncode != 0


@pytest.mark.parametrize("name", ["inst.npy", "type.npy", "tissue.npy", "gt_channels.npz"])
def test_test_stage_rejects_changed_validation_ground_truth(tmp_path, name):
    data, paths = make_inputs(tmp_path)
    env = dict(os.environ, PANNUKE_ROOT=str(data), CUDA_VISIBLE_DEVICES="")
    out = tmp_path / "results"
    val = subprocess.run(command(paths, out, "val"), capture_output=True, text=True, env=env)
    assert val.returncode == 0, val.stderr
    gt_path = data / "fold2" / name
    with gt_path.open("ab") as f:
        f.write(b"changed validation snapshot")
    test = subprocess.run(command(paths, out, "test"), capture_output=True, text=True, env=env)
    assert test.returncode != 0
    assert "validation ground truth" in test.stderr
    assert not (out / "test").exists()


def test_sweep_cli_keeps_official_channels_with_limit(tmp_path):
    data, paths = make_inputs(tmp_path)
    gt = np.zeros((2, 8, 8, 5), np.uint16)
    gt[:, 1:3, 1:3, 0] = 1
    gt[:, 2:4, 2:4, 1] = 1
    inst = np.zeros((2, 8, 8), np.int32)
    inst[gt[..., 0] > 0] = 1
    inst[gt[..., 1] > 0] = 2
    typ = inst.astype(np.uint8)
    d = data / "fold2"
    np.save(d / "inst.npy", inst)
    np.save(d / "type.npy", typ)
    np.savez_compressed(d / "gt_channels.npz", gt=gt)
    np.savez_compressed(paths["base-val"], inst=inst, type=typ)
    out = tmp_path / "sweep.json"
    cmd = [sys.executable, str(SCRIPT.with_name("lever_postproc.py")), "--pred", str(paths["base-val"]),
           "--fold", "2", "--mode", "sweep", "--limit", "1", "--workers", "1",
           "--fracs", "none", "--a-mins", "0", "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=dict(os.environ, PANNUKE_ROOT=str(data), CUDA_VISIBLE_DEVICES=""))
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["rows"][0]["mPQ"] == pytest.approx(.875)


def test_test_stage_requires_frozen_validation(tmp_path):
    data, paths = make_inputs(tmp_path)
    env = dict(os.environ, PANNUKE_ROOT=str(data), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(command(paths, tmp_path / "results", "test"), capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert not (tmp_path / "results" / "test").exists()
