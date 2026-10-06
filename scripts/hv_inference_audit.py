#!/usr/bin/env python
"""Rerun one completed HV control on validation fold2 with hash-bound provenance.

Uses the unchanged prediction/evaluation CLIs and the frozen final/no-TTA/decode
settings. Original predictions and eval_val_fold2 are never overwritten. Failed
attempts remain in a hidden staging directory and cannot count as complete audits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hv_control_numbers import (ARMS, COMMON, INFERENCE_SOURCES, SEEDS, SOURCE_COMMIT,
                                _training_record, file_sha256, prediction_name, require_values)


def frozen_commands(directory, scale):
    if scale not in (1, 2):
        raise ValueError("unsupported scale")
    directory = Path(directory)
    pred = prediction_name(scale)
    predict = [sys.executable, str(ROOT / "scripts/predict_cellvit.py"), "--run", str(directory),
               "--fold", "2", "--ckpt", "final.pth", "--upscale", str(scale),
               "--decode-u", str(scale), "--marker-u", "1", "--batch-size", "8", "--no-eval"]
    evaluate = [sys.executable, str(ROOT / "scripts/eval_pannuke.py"), "--pred", str(directory / pred),
                "--fold", "2", "--out", str(directory / "eval"), "--workers", "8"]
    return predict, evaluate, pred


def audit_run(run, data_root=None):
    run = Path(run).resolve()
    out = run / "audit_val_fold2"
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    cfg = json.loads((run / "config.json").read_text())
    scale, cutoff, seed = cfg["upscale"], cfg["hv_min_size"], cfg["seed"]
    arm = f"x{scale}_hv{cutoff}"
    if ARMS.get(arm) != (scale, cutoff) or seed not in SEEDS or run.name != f"{arm}_seed{seed}":
        raise ValueError("run is not in the frozen HV-control matrix")
    require_values(cfg, {**COMMON, "upscale": scale, "hv_min_size": cutoff, "seed": seed}, run.name)
    provenance = json.loads((run / "provenance.json").read_text())
    require_values(provenance, dict(source_commit=SOURCE_COMMIT, arm=arm, seed=seed,
                                   split=1, train_fold=1, val_fold=2, test_access=False), run.name)
    _training_record(run)
    before = file_sha256(run / "final.pth")
    code_hashes = {name: file_sha256(ROOT / name) for name in INFERENCE_SOURCES}
    stage = Path(tempfile.mkdtemp(prefix=".audit_val_fold2.", dir=run))
    (stage / "final.pth").symlink_to(run / "final.pth")
    (stage / "config.json").symlink_to(run / "config.json")
    predict, evaluate, pred = frozen_commands(stage, scale)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if data_root is not None:
        env["PANNUKE_ROOT"] = str(Path(data_root).resolve())
    for name, command in (("predict", predict), ("evaluate", evaluate)):
        with (stage / f"{name}.stdout.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    after = file_sha256(run / "final.pth")
    if before != after:
        raise ValueError("final checkpoint changed during audit inference")
    if any(file_sha256(ROOT / name) != sha for name, sha in code_hashes.items()):
        raise ValueError("inference code changed during audit")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=True,
                    training_source_commit=SOURCE_COMMIT, fold=2, tta=False, upscale=scale,
                    decode_u=scale, marker_u=1, batch_size=8, checkpoint="final.pth",
                    checkpoint_sha256=after, prediction_file=pred,
                    prediction_sha256=file_sha256(stage / pred),
                    evaluation_sha256={name: file_sha256(stage / "eval" / name) for name in
                                       ("summary.json", "per_image.npz", "gt_records.csv.gz")},
                    code_sha256=code_hashes, commands={"predict": predict, "evaluate": evaluate},
                    versions={name: version(name) for name in ("torch", "numpy")})
    (stage / "inference_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    stage.rename(out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    a = p.parse_args()
    out = audit_run(a.run, a.data_root)
    print(json.dumps({"run": a.run.name, "audit": str(out), "complete": True}))


if __name__ == "__main__":
    main()
