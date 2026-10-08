"""The external CLI must not label unsupported HoVer-NeXt inference as TTA."""
import subprocess
import sys
from pathlib import Path


def test_hovernext_rejects_unsupported_tta_before_loading_weights():
    script = Path(__file__).resolve().parents[1] / "scripts/predict_external.py"
    result = subprocess.run(
        [sys.executable, str(script), "--model", "hovernext", "--tta",
         "--run", "/nonexistent/bdd26-run", "--data", "conic"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode != 0
    assert "does not support --tta" in result.stderr
