"""Audit inference freezes arguments and publishes provenance only after success."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from test_hv_control_numbers import study

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hv_inference_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("hv_inference_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scale,pred_name", [(1, "pred_fold2.npz"), (2, "pred_fold2_x2_du2.npz")])
def test_audit_commands_use_final_fold2_no_tta_and_scaled_decode(tmp_path, scale, pred_name):
    predict, evaluate, filename = _module().frozen_commands(tmp_path, scale)
    assert filename == pred_name
    assert predict[predict.index("--run")+1] == str(tmp_path)
    assert predict[predict.index("--fold")+1] == "2"
    assert predict[predict.index("--ckpt")+1] == "final.pth"
    assert predict[predict.index("--decode-u")+1] == str(scale)
    assert predict[predict.index("--marker-u")+1] == "1"
    assert predict[predict.index("--batch-size")+1] == "8"
    assert "--tta" not in predict and "--no-eval" in predict
    assert evaluate[evaluate.index("--fold")+1] == "2"
    assert evaluate[evaluate.index("--pred")+1] == str(tmp_path / pred_name)
    assert evaluate[evaluate.index("--out")+1] == str(tmp_path / "eval")


def test_existing_audit_is_not_overwritten(tmp_path):
    run = tmp_path / "x1_hv30_seed19"
    (run / "audit_val_fold2").mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite"):
        _module().audit_run(run)


def test_unknown_scale_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="scale"):
        _module().frozen_commands(tmp_path, 3)


def test_successful_audit_records_actual_inputs_and_evaluation_hashes(study, monkeypatch):
    module = _module()
    root, data = study
    run = root / "x2_hv30_seed1"
    shutil.rmtree(run / "audit_val_fold2")

    def execute(command, **kwargs):
        if "--no-eval" in command:
            out = Path(command[command.index("--run")+1])
            assert (out / "final.pth").resolve() == run / "final.pth"
            (out / "pred_fold2_x2_du2.npz").write_bytes(b"actual test invocation output")
        else:
            out = Path(command[command.index("--out")+1])
            shutil.copytree(run / "eval_val_fold2", out)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    out = module.audit_run(run, data)
    manifest = json.loads((out / "inference_provenance.json").read_text())
    assert manifest["complete"] is True
    assert manifest["checkpoint_sha256"] == module.file_sha256(run / "final.pth")
    assert manifest["prediction_sha256"] == module.file_sha256(out / "pred_fold2_x2_du2.npz")
    assert manifest["evaluation_sha256"]["summary.json"] == module.file_sha256(out / "eval/summary.json")
    assert manifest["decode_u"] == 2 and manifest["tta"] is False


def test_failed_prediction_does_not_publish_complete_audit(study, monkeypatch):
    module = _module()
    root, data = study
    run = root / "x1_hv30_seed19"
    shutil.rmtree(run / "audit_val_fold2")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.audit_run(run, data)
    assert not (run / "audit_val_fold2").exists()
