"""Report figures must remain readable by standard PDF text extractors."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_report3_cjk_pdf_fonts_are_valid(tmp_path):
    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    extractor = shutil.which("pdftotext")
    if not font.is_file() or extractor is None:
        pytest.skip("Requires Droid Sans Fallback and Poppler for PDF validation")

    endpoints = ("mPQ", "bPQ", "Dead PQ")
    baseline = {key: 0.5 for key in endpoints}
    delta = {key: 0.001 for key in endpoints}
    pairs = {f"split{split}_seed{seed}": {"baseline": baseline, "delta": delta}
             for split in (1, 2, 3) for seed in (19, 1, 2)}
    interval = {key: {"point": 0.001, "ci95": [-0.001, 0.003]}
                for key in endpoints}
    for name in ("matched_seed_20261004", "existence_ef_20261005"):
        root = tmp_path / name / "stats"
        root.mkdir(parents=True)
        (root / "stats.json").write_text(json.dumps({"pairs": pairs}))
        for split in (1, 2, 3):
            (root / f"bootstrap_split{split}.json").write_text(json.dumps(interval))

    script = Path(__file__).resolve().parents[1] / "scripts/make_report_figures3.py"
    output = tmp_path / "figs"
    subprocess.run([sys.executable, str(script), "--analysis-root", str(tmp_path),
                    "--out", str(output)],
                   check=True, capture_output=True, text=True)
    for name in ("paired_gains_zh", "split_intervals_zh"):
        result = subprocess.run([extractor, str(output / f"{name}.pdf"), "-"],
                                check=True, capture_output=True, text=True)
        assert not result.stderr, result.stderr
        assert "变化" in result.stdout
        assert "Dead PQ" in result.stdout
