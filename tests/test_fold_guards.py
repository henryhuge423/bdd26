"""Dev/probe CLIs must refuse their run split's TEST fold (audit 2026-10-08 hardening).

These run each script as a subprocess with a minimal fake run dir; the guard must fire
before any heavy artifact (maps/preds/GPU) is touched.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PY = sys.executable


def _run(script, *args):
    return subprocess.run([PY, str(SCRIPTS / f"{script}.py"), *args],
                          capture_output=True, text=True, timeout=180)


def _fake_run(tmp_path, split=1):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"split": split}))
    return run


def test_dsb_dev_eval_refuses_test_fold(tmp_path):
    run = _fake_run(tmp_path, split=1)                       # split 1 tests fold 3
    r = _run("dsb_dev_eval", "--run", str(run), "--fold", "3")
    assert r.returncode != 0
    assert "TEST fold" in r.stderr


def test_dsb_dev_eval_requires_split_provenance(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    r = _run("dsb_dev_eval", "--run", str(run), "--fold", "2")
    assert r.returncode != 0
    assert "config.json" in r.stderr


def test_dsb_oracle_ceiling_refuses_test_fold(tmp_path):
    run = _fake_run(tmp_path, split=3)                       # split 3 tests fold 1
    r = _run("dsb_oracle_ceiling", "--run", str(run), "--fold", "1",
             "--out", str(tmp_path / "gate_d.json"))
    assert r.returncode != 0
    assert "TEST fold" in r.stderr


def test_dsb_gate_c_refuses_test_fold(tmp_path):
    r = _run("dsb_gate_c", "--maps", str(tmp_path), "--fold", "3",
             "--out", str(tmp_path / "gate_c.json"))
    assert r.returncode != 0
    assert "TEST fold" in r.stderr


def test_inference_cost_refuses_test_fold(tmp_path):
    run = _fake_run(tmp_path, split=1)
    r = _run("inference_cost", "--x1-run", str(run), "--x2-run", str(run), "--fold", "3",
             "--ef-selection", "missing.json", "--p2-selection", "missing.json",
             "--base-pred", "missing.npz", "--x2-pred", "missing.npz",
             "--out", str(tmp_path / "cost.json"))
    assert r.returncode != 0
    assert "TEST fold" in r.stderr


def test_inference_cost_requires_split_provenance(tmp_path):
    run = tmp_path / "x1"
    run.mkdir()
    r = _run("inference_cost", "--x1-run", str(run), "--x2-run", str(run), "--fold", "2",
             "--ef-selection", "missing.json", "--p2-selection", "missing.json",
             "--base-pred", "missing.npz", "--x2-pred", "missing.npz",
             "--out", str(tmp_path / "cost.json"))
    assert r.returncode != 0
    assert "config.json" in r.stderr
