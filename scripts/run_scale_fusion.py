#!/usr/bin/env python
"""P2 fixed-menu, validation-gated additions to a frozen base; test uses only the frozen choice."""
from __future__ import annotations
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from lever_postproc import file_provenance
from run_p0_controls import check_prediction
from analyze_scale_complementarity import provenance, validate_run_context, write_json
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics import light
from nucseg.metrics.pannuke_eval import evaluate, save_report
from nucseg.postproc.scale_fusion import (candidate_info, fuse, probability_rows, strict_dead_pq,
                                         select_fusion, _gt_facts)

GRID = [{"name": "identity", "max_area": None, "min_prob": 0., "interior_only": False},
        {"name": "unfiltered", "max_area": None, "min_prob": 0., "interior_only": False}]
GRID += [{"name": f"a{area}_p{prob:g}_interior{int(interior)}", "max_area": area,
          "min_prob": prob, "interior_only": interior}
         for area in (100, 200) for prob in (.5, .7, .9) for interior in (True, False)]
_G = {}


def image_confidence(inst, typ, rows):
    ids = set(np.unique(inst).tolist()) - {0}
    if ids != set(rows):
        raise ValueError("original probability rows do not exactly match this image's surviving instance IDs")
    return {c["id"]: float(rows[c["id"]][c["class"]]) for c in candidate_info(inst, inst, typ)}


def apply_config(bi, bt, xi, xt, confidence, config):
    if config["name"] == "identity":
        return bi, bt, []
    return fuse(bi, bt, xi, xt, confidence, config["max_area"], config["min_prob"], config["interior_only"])


def _one(j):
    bi, bt, xi, xt = (_G[k][j] for k in ("bi", "bt", "xi", "xt"))
    confidence = image_confidence(xi, xt, _G["prob"].get(j, {}))
    rows = []
    for cfg in _G["grid"]:
        pi, pt, added = apply_config(bi, bt, xi, xt, confidence, cfg)
        stats = light.image_stats(_G["gc"][j], _G["gi"][j], _G["gt"][j], pi, pt)
        counts = _gt_facts(_G["gi"][j], _G["gt"][j], pi, pt)[-1]
        rows.append((stats, counts["interior_matched"], len(added)))
    return rows


def _apply(j):
    bi, bt, xi, xt = (_G[k][j] for k in ("bi", "bt", "xi", "xt"))
    conf = image_confidence(xi, xt, _G["prob"].get(j, {}))
    pi, pt, added = apply_config(bi, bt, xi, xt, conf, _G["grid"][0])
    return j, pi, pt, len(added)


def run(a, role):
    destination = a.out / role
    if destination.exists():
        raise FileExistsError(destination)
    fold = split_folds(a.split)[1 if role == "val" else 2]
    f = PanNukeFold(fold)
    base, x2 = getattr(a, "base_" + role), getattr(a, "x2_" + role)
    evidence = provenance({"base": base, "x2": x2}, f, __file__)
    validate_run_context(evidence, a.split, a.expect_seed)
    choice_path = a.out / "selection.json"
    if role == "test":
        frozen = json.loads(choice_path.read_text())
        if frozen["split"] != a.split or frozen["provenance"]["source"] != evidence["source"]:
            raise ValueError("frozen split or code changed")
        for name, path in (("base", a.base_val), ("x2", a.x2_val)):
            if file_provenance(path)["sha256"] != frozen["provenance"]["inputs"][name]["sha256"]:
                raise ValueError("validation predictions changed")
        # Bind the validation dataset as well as the predictions and code.
        vf = PanNukeFold(split_folds(a.split)[1])
        from run_p0_controls import ground_truth_provenance
        if ground_truth_provenance(vf) != frozen["provenance"]["ground_truth"]:
            raise ValueError("validation annotations changed")
        configs = [frozen["selected"]]
    else:
        if choice_path.exists():
            raise FileExistsError(choice_path)
        configs = GRID
    with np.load(base) as b, np.load(x2) as x:
        bi, bt, xi, xt = b["inst"], b["type"], x["inst"], x["type"]
        check_prediction(bi, bt, f, b); check_prediction(xi, xt, f, x)
        prob = probability_rows(x)
        if any(j >= len(f) for j in prob):
            raise ValueError("probability table references an image outside the fold")
    _G.update(bi=bi, bt=bt, xi=xi, xt=xt, gi=f.inst, gt=f.type, gc=f.gt_channels, prob=prob, grid=configs)
    destination.mkdir(parents=True, exist_ok=False)
    if role == "val":
        with Pool(a.workers) as pool:
            results = pool.map(_one, range(len(f)), chunksize=4)
        rows = []
        for k, cfg in enumerate(configs):
            stats = np.stack([r[k][0] for r in results])
            s = light.summarize(stats, f.tissue)
            rows.append({**cfg, "mPQ": s["mPQ"], "bPQ": s["bPQ"],
                         "Dead_PQ": s["per_class_PQ"]["Dead"], "strict_dead": strict_dead_pq(stats),
                         "interior_matched": int(sum(r[k][1] for r in results)),
                         "added": int(sum(r[k][2] for r in results))})
        selected = select_fusion(rows)
        record = {"split": a.split, "fold": fold, "role": "validation selection",
                  "expected_seed": a.expect_seed,
                  "status": "NO_GO" if selected["name"] == "identity" else "GO",
                  "reference": "P0 validation-selected corrected base M/S; original du2 candidates",
                  "rule": "mPQ,bPQ >= base-.002; strict Dead PQ >= base; interior Dead matched count increases; max mPQ, fewer additions tie-break",
                  "rows": rows, "selected": selected, "provenance": evidence}
        write_json(choice_path, record)
        write_json(destination / "sweep.json", record)
        print(f"split{a.split} P2 {record['status']}: {selected}", flush=True)
    else:
        pi = np.empty_like(bi); pt = np.empty_like(bt); added = 0
        with Pool(a.workers) as pool:
            for j, one_i, one_t, n in pool.imap(_apply, range(len(f)), chunksize=4):
                pi[j] = one_i; pt[j] = one_t; added += n
        pred = destination / "pred.npz"
        with pred.open("xb") as out:
            np.savez_compressed(out, inst=pi, type=pt)
        result = evaluate(f.gt_channels, f.inst, f.type, f.tissue, pi, pt, workers=a.workers)
        save_report(result, destination / "eval")
        write_json(destination / "manifest.json", {"split": a.split, "fold": fold,
                   "role": "frozen test; identity fallback if validation NO_GO",
                   "selected": configs[0], "added_instances": added, "provenance": evidence,
                   "selection_sha256": file_provenance(choice_path)["sha256"],
                   "prediction": file_provenance(pred)})
        print(f"split{a.split} P2 test mPQ={result['summary']['official']['mPQ']:.6f}", flush=True)
    _G.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=int, required=True, choices=(1, 2, 3))
    for model in ("base", "x2"):
        for role in ("val", "test"):
            p.add_argument(f"--{model}-{role}", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expect-seed", type=int, default=19, dest="expect_seed",
                   help="declared training seed of the matched pair; both arms must verify to it")
    p.add_argument("--stage", choices=("val", "test", "all"), default="all")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    for role in ("val", "test") if a.stage == "all" else (a.stage,):
        run(a, role)


if __name__ == "__main__":
    main()
