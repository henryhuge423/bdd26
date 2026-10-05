#!/usr/bin/env python
"""P1 descriptive error transitions and GT-assisted/type-swap diagnostics on saved predictions."""
from __future__ import annotations
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lever_postproc import file_provenance, write_json as _write_json
from run_p0_controls import ground_truth_provenance, check_prediction
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics import light
from nucseg.postproc.scale_fusion import (STATE_NAMES, image_diagnostics, oracle_additions,
                                         transfer_types, strict_dead_pq)

ARMS = ("base", "x2", "oracle_add", "base_geometry_x2_types", "x2_geometry_base_types")
_G = {}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def write_json(path, data):
    _write_json(path, json_safe(data))


def run_image(gt, gt_type, gt_ch, base, base_type, x2, x2_type):
    diag = image_diagnostics(gt, gt_type, base, base_type, x2, x2_type)
    oi, ot, chosen = oracle_additions(base, base_type, x2, x2_type, gt, gt_type)
    bt, bn = transfer_types(base, base_type, x2, x2_type)
    xt, xn = transfer_types(x2, x2_type, base, base_type)
    predictions = ((base, base_type), (x2, x2_type), (oi, ot), (base, bt), (x2, xt))
    stats = np.stack([light.image_stats(gt_ch, gt, gt_type, i, t) for i, t in predictions])
    return diag, stats, [len(chosen), bn, xn]


def _one(j):
    return run_image(_G["gi"][j], _G["gt"][j], _G["gc"][j],
                     _G["bi"][j], _G["bt"][j], _G["xi"][j], _G["xt"][j])


def sum_diagnostics(values):
    first = values[0]
    if isinstance(first, dict):
        return {k: sum_diagnostics([v[k] for v in values]) for k in first}
    if isinstance(first, list):
        return np.sum(np.asarray(values), axis=0).tolist()
    return sum(values)


def prediction_context(path):
    path = Path(path)
    manifest = path.parent / "manifest.json"
    config = path.parent / "config.json"
    if manifest.exists():
        d = json.loads(manifest.read_text())
        if d["prediction"]["sha256"] != file_provenance(path)["sha256"]:
            raise ValueError("prediction hash does not match its provenance sidecar")
        return {"actual_seed": d.get("actual_seed"), "split": d.get("split"),
                "sidecar": file_provenance(manifest)}
    if config.exists():
        d = json.loads(config.read_text())
        return {"actual_seed": d.get("seed"), "split": d.get("split"),
                "config": file_provenance(config), "checkpoint_binding": "legacy run provenance only"}
    return {"actual_seed": None, "split": None, "status": "legacy context unavailable"}


def provenance(paths, fold, script):
    return {"inputs": {k: file_provenance(p) for k, p in paths.items()},
            "run_context": {k: prediction_context(p) for k, p in paths.items()},
            "ground_truth": ground_truth_provenance(fold),
            "source": {str(p.relative_to(ROOT)): file_provenance(p) for p in
                       sorted(set([Path(script).resolve(), Path(__file__).resolve(), ROOT / "src/nucseg/postproc/scale_fusion.py",
                        ROOT / "src/nucseg/metrics/pannuke_eval.py", ROOT / "src/nucseg/metrics/light.py",
                        ROOT / "src/nucseg/metrics/instance.py", ROOT / "src/nucseg/metrics/errors.py"]))}}


def validate_run_context(evidence, split, expected_seed=19):
    for name, context in evidence["run_context"].items():
        if context.get("split") != split or context.get("actual_seed") != expected_seed:
            raise ValueError(f"{name}: P1/P2 require verified seed{expected_seed} predictions for the declared split")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--x2", type=Path, required=True)
    p.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--role", choices=("val", "test"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    fold_id = split_folds(a.split)[1 if a.role == "val" else 2]
    f = PanNukeFold(fold_id)
    evidence = provenance({"base": a.base, "x2": a.x2}, f, __file__)
    validate_run_context(evidence, a.split)
    with np.load(a.base) as b, np.load(a.x2) as x:
        bi, bt, xi, xt = b["inst"], b["type"], x["inst"], x["type"]
        check_prediction(bi, bt, f, b); check_prediction(xi, xt, f, x)
    _G.update(gi=f.inst, gt=f.type, gc=f.gt_channels, bi=bi, bt=bt, xi=xi, xt=xt)
    with Pool(a.workers) as pool:
        results = pool.map(_one, range(len(f)), chunksize=8)
    stats = np.stack([r[1] for r in results])
    summaries = {name: {**light.summarize(stats[:, k], f.tissue),
                        "strict_dead": strict_dead_pq(stats[:, k])} for k, name in enumerate(ARMS)}
    data = {"split": a.split, "fold": fold_id, "role": a.role,
            "scope": "descriptive; oracle uses GT and is not a deployable result; swaps cover stable one-to-one matches only",
            "state_order": STATE_NAMES, "n_images": len(f),
            "diagnostics": sum_diagnostics([r[0] for r in results]), "scores": summaries,
            "oracle_added_and_swap_coverage": dict(zip(("oracle_added", "base_swap_instances", "x2_swap_instances"),
                                                       np.sum([r[2] for r in results], axis=0).tolist())),
            "by_tissue": {str(t): sum_diagnostics([results[j][0] for j in np.flatnonzero(f.tissue == t)])
                          for t in np.unique(f.tissue)},
            "provenance": evidence}
    write_json(a.out, data)
    print(json.dumps({"output": str(a.out), "scores": summaries,
                      "candidates": data["diagnostics"]["candidates"]}), flush=True)


if __name__ == "__main__":
    main()
