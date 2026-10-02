"""The diagnostic CLI must accept paths without importing GPU dependencies for help."""
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kongnet_quantiles.py"


def test_help_without_site_packages(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--repo" in result.stdout
    assert "--checkpoint" in result.stdout


def test_paths_are_required_before_loading_model(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stderr
    assert "--repo" in result.stderr
    assert "--checkpoint" in result.stderr
