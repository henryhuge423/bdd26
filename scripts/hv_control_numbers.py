#!/usr/bin/env python
"""Verify and summarize the frozen four-arm, two-seed HV-control development study.

Run on the artifact host after all eight final checkpoints and hash-bound fold2
audit evaluations exist (see hv_inference_audit.py). Never predicts, changes parameters, or reads fold3 data. Official mPQ/bPQ/
Dead are rebuilt from per-image arrays; strict and pooled endpoints are taken
from evaluator summaries. Bootstrap intervals condition on the two trained seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from matched_seed_stats import ENDPOINTS, bootstrap, load_eval
from nucseg.constants import TISSUES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import instance_classes

SOURCE_COMMIT = "c6e7e92aceea9a9bc28f85d77611f302af9b7387"
SEEDS = (19, 1)
ARMS = {"x1_hv30": (1, 30), "x1_hv8": (1, 8), "x2_hv30": (2, 30), "x2_hv120": (2, 120)}
CONTRASTS = {"lower_support_x1": ("x1_hv8", "x1_hv30"),
             "lower_support_x2": ("x2_hv30", "x2_hv120"),
             "scale_at_high_support": ("x2_hv120", "x1_hv30"),
             "scale_at_low_support": ("x2_hv30", "x1_hv8")}
COMMON = dict(split=1, epochs=130, unfreeze_epoch=25, batch_size=16, lr=.0003,
              weight_decay=.0001, gamma=.85, sampling_gamma=.85, workers=8, val_every=5,
              np_wce=0., dead_w=0., small_w=0., cp_prob=0., synth=None, synth_frac=0.)
REQUIRED = ("config.json", "provenance.json", "log.jsonl", "final.pth", "last.pth",
            "eval_val_fold2/summary.json", "eval_val_fold2/per_image.npz",
            "eval_val_fold2/gt_records.csv.gz", "audit_val_fold2/inference_provenance.json",
            "audit_val_fold2/eval/summary.json", "audit_val_fold2/eval/per_image.npz",
            "audit_val_fold2/eval/gt_records.csv.gz")
INFERENCE_SOURCES = ("scripts/predict_cellvit.py", "scripts/eval_pannuke.py",
                     "src/nucseg/cellvit/model.py", "src/nucseg/cellvit/engine.py",
                     "src/nucseg/hovernet/engine.py", "src/nucseg/metrics/pannuke_eval.py",
                     "src/nucseg/metrics/instance.py", "third_party/hover_net/models/hovernet/post_proc.py")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_name(scale):
    return "pred_fold2.npz" if scale == 1 else "pred_fold2_x2_du2.npz"


def require_values(data, expected, where):
    for key, value in expected.items():
        actual = data.get(key)
        if key not in data or actual != value or (isinstance(actual, bool) != isinstance(value, bool)):
            raise ValueError(f"{where}: unexpected {key}: {actual!r}, expected {value!r}")


def verify_inference(run, scale):
    audit = run / "audit_val_fold2"
    manifest = json.loads((audit / "inference_provenance.json").read_text())
    require_values(manifest, dict(fold=2, tta=False, upscale=scale, decode_u=scale, marker_u=1,
                                  batch_size=8, checkpoint="final.pth", complete=True,
                                  prediction_file=prediction_name(scale)), run.name)
    if file_sha256(run / "final.pth") != manifest.get("checkpoint_sha256"):
        raise ValueError(f"{run.name}: checkpoint hash mismatch")
    if file_sha256(audit / prediction_name(scale)) != manifest.get("prediction_sha256"):
        raise ValueError(f"{run.name}: prediction hash mismatch")
    for name in ("summary.json", "per_image.npz", "gt_records.csv.gz"):
        if file_sha256(audit / "eval" / name) != manifest.get("evaluation_sha256", {}).get(name):
            raise ValueError(f"{run.name}: evaluation hash mismatch: {name}")
    for name in INFERENCE_SOURCES:
        if file_sha256(ROOT / name) != manifest.get("code_sha256", {}).get(name):
            raise ValueError(f"{run.name}: inference code hash mismatch: {name}")
    return manifest


def gt_metadata(fold):
    """Use exactly the evaluator's instance order; preserve empty-image indices."""
    tables = []
    for image in range(len(fold)):
        labels, classes = instance_classes(np.asarray(fold.inst[image]), np.asarray(fold.type[image]))
        n = len(classes)
        if not n:
            continue
        edge_ids = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        tables.append(pd.DataFrame({"image": image, "cls": classes,
                                    "area": np.bincount(labels.ravel())[1:],
                                    "tissue": str(fold.tissue[image]),
                                    "border": np.isin(np.arange(1, n+1), edge_ids)}))
    if not tables:
        raise ValueError("No GT instances in validation fold")
    return pd.concat(tables, ignore_index=True)


def gt_views(records, metadata):
    cols = ["image", "cls", "area", "tissue"]
    if len(records) != len(metadata) or not np.array_equal(records[cols].to_numpy(), metadata[cols].to_numpy()):
        raise ValueError("GT alignment mismatch: order/count/class/area/tissue differs from validation GT")
    dead = metadata["cls"].to_numpy() == 4
    border = metadata["border"].to_numpy()
    area = metadata["area"].to_numpy()
    masks = {"all": np.ones(len(records), bool), "dead": dead,
             "interior_dead": dead & ~border, "border_dead": dead & border,
             "native_area_lt30": area < 30, "native_area_30to59": (area >= 30) & (area < 60),
             "native_area_ge60": area >= 60}
    result = {}
    for name, mask in masks.items():
        g = records.loc[mask]
        n = len(g)
        counts = {"matched": int((g.status == "matched").sum()),
                  "missed_bg": int((g.status == "missed_bg").sum()),
                  "matched_typed_correct": int(((g.status == "matched") & (g.pred_cls == g.cls)).sum())}
        result[name] = {"n": n, **counts,
                        **{f"{key}_rate": value/n if n else None for key, value in counts.items()}}
    return result


def _training_record(run):
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    epochs = [row["epoch"] for row in rows]
    if not rows or set(epochs) != set(range(130)) or epochs[-1] != 129:
        raise ValueError(f"{run.name}: incomplete or unexpected epoch log")
    if not all(np.isfinite(row["train/loss"]) for row in rows):
        raise ValueError(f"{run.name}: nonfinite training loss")
    return {"logged_epochs": len(rows), "unique_epochs": len(set(epochs)),
            "duplicate_epoch_records": len(rows)-len(set(epochs)),
            "training_loop_hours": sum(row["time"] for row in rows)/3600,
            "final_checkpoint_bytes": (run / "final.pth").stat().st_size,
            "last_checkpoint_bytes": (run / "last.pth").stat().st_size}


def analyze(root, data_root=None, boot=2000, bootstrap_seed=20261006):
    root = Path(root)
    if boot < 1:
        raise ValueError("boot must be positive")
    # Check completeness before reading any partial performance table.
    for arm, (scale, _) in ARMS.items():
        for seed in SEEDS:
            run = root / f"{arm}_seed{seed}"
            required = (*REQUIRED, f"audit_val_fold2/{prediction_name(scale)}")
            missing = [name for name in required if not (run / name).is_file() or (run / name).stat().st_size == 0]
            if missing:
                raise ValueError(f"{run.name}: incomplete artifacts: {missing}")
            if any(run.rglob("*fold3*")):
                raise ValueError(f"{run.name}: forbidden fold3 artifact in validation-only run")
    fold = PanNukeFold(2, root=Path(data_root) if data_root is not None else None)
    tissue = np.asarray(fold.tissue)
    tissue_idx = [np.where(tissue == t)[0] for t in TISSUES if (tissue == t).any()]
    if sum(map(len, tissue_idx)) != len(tissue):
        raise ValueError("Unknown validation tissue label")
    metadata = gt_metadata(fold)
    runs, arrays, hashes = {}, {}, {}
    for arm, (scale, cutoff) in ARMS.items():
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            run = root / name
            cfg = json.loads((run / "config.json").read_text())
            expected = {**COMMON, "seed": seed, "upscale": scale, "hv_min_size": cutoff}
            require_values(cfg, expected, name)
            provenance = json.loads((run / "provenance.json").read_text())
            require_values(provenance, dict(source_commit=SOURCE_COMMIT, arm=arm, seed=seed, split=1,
                                           train_fold=1, val_fold=2, test_access=False, upscale=scale,
                                           hv_min_size=cutoff, batch_size=16, epochs=130), name)
            training = _training_record(run)
            inference = verify_inference(run, scale)
            directory = run / "audit_val_fold2/eval"
            points, per_image = load_eval(directory, tissue_idx)
            if per_image.shape != (len(tissue), 3):
                raise ValueError(f"{name}: per-image shape mismatch")
            dead = float(np.nanmean(per_image[:, 2]))
            if not np.isclose(dead, points["Dead PQ"], atol=1e-10, rtol=0):
                raise ValueError(f"{name}: Dead summary disagrees with per-image aggregation")
            if not all(np.isfinite(value) for value in points.values()):
                raise ValueError(f"{name}: nonfinite endpoint")
            records = pd.read_csv(directory / "gt_records.csv.gz")
            views = gt_views(records, metadata)
            original_points, _ = load_eval(run / "eval_val_fold2", tissue_idx)
            runs[name] = {"arm": arm, "seed": seed, "endpoints": points, "gt_views": views,
                          "training": training, "scientific_config": expected,
                          "inference": {key: inference[key] for key in
                                        ("fold", "tta", "upscale", "decode_u", "marker_u", "batch_size",
                                         "checkpoint_sha256", "prediction_sha256")},
                          "audit_minus_original": {ep: points[ep]-original_points[ep] for ep in ENDPOINTS}}
            arrays[name] = per_image
            for relative in REQUIRED:
                if not relative.endswith(".pth"):
                    path = run / relative
                    hashes[f"{name}/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    arms = {arm: {"mean": {ep: float(np.mean([runs[f"{arm}_seed{s}"]["endpoints"][ep] for s in SEEDS]))
                           for ep in ENDPOINTS}} for arm in ARMS}
    contrasts = {}
    for label, (method, base) in CONTRASTS.items():
        paired = {str(s): {ep: runs[f"{method}_seed{s}"]["endpoints"][ep] - runs[f"{base}_seed{s}"]["endpoints"][ep]
                           for ep in ENDPOINTS} for s in SEEDS}
        delta_arrays = {str(s): arrays[f"{method}_seed{s}"] - arrays[f"{base}_seed{s}"] for s in SEEDS}
        contrasts[label] = {"method": method, "reference": base, "per_seed": paired,
                            "mean": {ep: float(np.mean([row[ep] for row in paired.values()])) for ep in ENDPOINTS},
                            "image_bootstrap": bootstrap(delta_arrays, tissue, boot, bootstrap_seed)}
    return {"protocol": {"split": 1, "training_fold": 1, "validation_fold": 2, "seeds": list(SEEDS),
                         "source_commit": SOURCE_COMMIT, "bootstrap_replicates": boot, "bootstrap_seed": bootstrap_seed,
                         "scope": "development only; no independent test or training-seed significance claim",
                         "rebuilt_endpoints": ["mPQ", "bPQ", "Dead PQ"],
                         "summary_only_endpoints": ["mPQ+", "strict mPQ", "strict Dead PQ"],
                         "checkpoint_check": "SHA256-bound audit inference from final.pth and complete epoch logs",
                         "primary_evaluation": "audit_val_fold2/eval; original queue evaluations retained for comparison"},
            "runs": runs, "arms": arms, "contrasts": contrasts, "input_sha256": hashes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261006)
    a = p.parse_args()
    if a.out.exists():
        p.error(f"refusing to overwrite {a.out}")
    result = analyze(a.root, a.data_root, a.boot, a.bootstrap_seed)
    text = json.dumps(result, indent=2, allow_nan=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as handle:
        handle.write(text + "\n")
    print(json.dumps({"arms": result["arms"], "contrasts": result["contrasts"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
