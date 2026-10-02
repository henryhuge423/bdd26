#!/usr/bin/env python
"""P0 fair controls: freeze validation-only M/S choices, then evaluate untouched test folds.

Example (all prediction inputs are explicit; no test file is read during --stage val):
  python scripts/run_p0_controls.py --split 1 --base-run runs/cellvit_uni/split1 \
    --x2-run runs/cellvit_uni_x2/split1 --base-val BVAL.npz --x2-val XVAL.npz \
    --base-test BTEST.npz --x2-test XTEST.npz --out runs/analysis/p0p3_20261002/p0/split1 \
    --stage all --workers 2

A completed validation stage may be followed by --stage test against the same output
root. Existing stages are never overwritten. A failed stage needs a new versioned root.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lever_postproc import (code_provenance, file_provenance, prediction_payload,
                            sweep_predictions, transform_predictions, write_json)
from nucseg.constants import SPLITS
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate, save_report

GRID = [(frac, area) for frac in (None, 0., .25, .5, .75) for area in (0, 20, 40, 80)]
BPQ_MARGIN = .002


def select_config(rows: list[dict]) -> dict:
    """Max validation mPQ under the bPQ guard; exact ties prefer no intervention."""
    identity = [r for r in rows if r["frac"] is None and r["a_min"] == 0]
    if len(identity) != 1 or not np.isfinite([identity[0]["mPQ"], identity[0]["bPQ"]]).all():
        raise ValueError("selection requires one finite identity row")
    eligible = [r for r in rows if np.isfinite([r["mPQ"], r["bPQ"]]).all()
                and r["bPQ"] >= identity[0]["bPQ"] - BPQ_MARGIN]
    return max(eligible, key=lambda r: (r["mPQ"], r["frac"] is None and r["a_min"] == 0,
                                       -r["a_min"], float("inf") if r["frac"] is None else r["frac"]))


def run_info(path: Path, split: int, scale: int) -> dict:
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("split") != split or config.get("upscale", 1) != scale:
        raise ValueError(f"run config split/scale mismatch: {config_path}")
    if not isinstance(config.get("seed"), int):
        raise ValueError(f"actual seed missing from {config_path}")
    return {"run": str(path.resolve()), "actual_seed": config["seed"],
            "train_scale": scale, "run_config": file_provenance(config_path),
            "checkpoint_policy": "final checkpoint required; legacy NPZ does not embed checkpoint hash"}


def check_prediction(inst, typ, fold, original) -> None:
    if inst.shape != typ.shape or inst.shape != fold.inst.shape:
        raise ValueError("P0 requires full-fold instance/type arrays in original fold order")
    if not np.issubdtype(inst.dtype, np.integer) or not np.issubdtype(typ.dtype, np.integer):
        raise ValueError("instance and type maps must be integer arrays")
    if np.any(inst < 0) or np.any(typ < 0) or np.any(typ > 5):
        raise ValueError("invalid instance or type IDs")
    if "image_indices" in original and not np.array_equal(original["image_indices"], np.arange(len(inst))):
        raise ValueError("subset/reordered predictions are not valid P0 full-fold inputs")


def ground_truth_provenance(fold: PanNukeFold) -> dict:
    return {name: file_provenance(fold.dir / name) for name in
            ("inst.npy", "type.npy", "tissue.npy", "gt_channels.npz")}


def evaluate_variants(original_path, original, inst, typ, fold, selected, destination, metadata, workers):
    """Process one arm at a time; do not retain transformed full folds across variants."""
    for variant in ("identity", "ms"):
        out = destination / variant
        out.mkdir(parents=True, exist_ok=False)
        if variant == "identity":
            pred_i, pred_t = inst, typ
            shutil.copyfile(original_path, out / "pred.npz")
            params = {"frac": None, "a_min": 0}
        else:
            params = {k: selected[k] for k in ("frac", "a_min")}
            pred_i = transform_predictions(inst, typ, **params, workers=workers)
            payload = prediction_payload(original, pred_i)
            pred_t = payload["type"]
            with (out / "pred.npz").open("xb") as stream:
                np.savez_compressed(stream, **payload)
        write_json(out / "manifest.json", {**metadata, "variant": variant, "parameters": params,
                                          "n_images": len(inst), "index_order": "full fold, original order",
                                          "prediction": file_provenance(out / "pred.npz")})
        result = evaluate(fold.gt_channels, fold.inst, fold.type, fold.tissue,
                          pred_i, pred_t, workers=workers)
        save_report(result, out / "eval")
        print(f"{out}: mPQ={result['summary']['official']['mPQ']:.6f} "
              f"bPQ={result['summary']['official']['bPQ']:.6f}", flush=True)
        del result, pred_i, pred_t
        if variant == "ms":
            del payload
        gc.collect()


def run_stage(args, role: str) -> None:
    selection_path = args.out / "selection.json"
    code = code_provenance()
    if (args.out / role).exists() or (role == "val" and selection_path.exists()):
        raise FileExistsError(f"refusing to overwrite {role} stage under {args.out}")
    if role == "test":
        selection = json.loads(selection_path.read_text())
        if selection["split"] != args.split or selection["code"]["files"] != code["files"]:
            raise ValueError("frozen selection split or code fingerprint changed")
        validation_gt = ground_truth_provenance(PanNukeFold(SPLITS[args.split][1]))
        if {k: v["sha256"] for k, v in validation_gt.items()} != {
                k: v["sha256"] for k, v in selection.get("validation_ground_truth", {}).items()}:
            raise ValueError("validation ground truth differs from frozen selection")
        for arm in ("base", "x2"):
            val_hash = file_provenance(getattr(args, arm + "_val"))["sha256"]
            if val_hash != selection["arms"][arm]["input"]["sha256"]:
                raise ValueError(f"{arm} validation input differs from frozen selection")
    else:
        selection = {"split": args.split, "validation_fold": SPLITS[args.split][1],
                     "created_utc": datetime.now(timezone.utc).isoformat(), "code": code,
                     "selection_rule": "max val mPQ; bPQ >= identity-.002; conservative exact ties",
                     "grid": [{"frac": f, "a_min": a} for f, a in GRID], "arms": {}}
    fold_id = SPLITS[args.split][1 if role == "val" else 2]
    fold = PanNukeFold(fold_id)
    gt_provenance = ground_truth_provenance(fold)
    if role == "val":
        selection["validation_ground_truth"] = gt_provenance
    # Preflight all run configs and inputs before creating any stage output.
    infos = {arm: run_info(getattr(args, arm + "_run"), args.split, scale)
             for arm, scale in (("base", 1), ("x2", 2))}
    inputs = {arm: file_provenance(getattr(args, arm + "_" + role)) for arm in infos}
    if role == "test":
        for arm in infos:
            if infos[arm]["run_config"]["sha256"] != selection["arms"][arm]["run_config"]["sha256"]:
                raise ValueError(f"{arm} run config differs from frozen selection")
    (args.out / role).mkdir(parents=True, exist_ok=False)
    for arm in ("base", "x2"):
        path = getattr(args, arm + "_" + role)
        with np.load(path) as original:
            inst, typ = original["inst"], original["type"]
            check_prediction(inst, typ, fold, original)
            destination = args.out / role / arm
            metadata = {**infos[arm], "input": inputs[arm], "code": code, "ground_truth": gt_provenance,
                        "split": args.split, "fold": fold_id, "role": role, "arm": arm}
            if role == "val":
                rows = sweep_predictions(inst, typ, fold, GRID, args.workers)
                selected = select_config(rows)
                selection["arms"][arm] = {**infos[arm], "input": inputs[arm], "selected": selected}
                write_json(destination / "sweep.json", {**metadata, "rows": rows, "selected": selected,
                                                        "selection_bpq_margin": BPQ_MARGIN})
            else:
                selected = selection["arms"][arm]["selected"]
                metadata["selection_sha256"] = file_provenance(selection_path)["sha256"]
            evaluate_variants(path, original, inst, typ, fold, selected, destination, metadata, args.workers)
            del inst, typ
        gc.collect()
    if role == "val":
        write_json(selection_path, selection)
        print(f"frozen validation selection: {selection_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
    for arm in ("base", "x2"):
        p.add_argument(f"--{arm}-run", type=Path, required=True)
        p.add_argument(f"--{arm}-val", type=Path, required=True)
        p.add_argument(f"--{arm}-test", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("val", "test", "all"), default="all")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    if args.stage != "val" and (args.base_test is None or args.x2_test is None):
        p.error("test/all stages require --base-test and --x2-test")
    for role in ("val", "test") if args.stage == "all" else (args.stage,):
        run_stage(args, role)


if __name__ == "__main__":
    main()
