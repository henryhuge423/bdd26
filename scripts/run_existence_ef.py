#!/usr/bin/env python
"""EF-P2: existence filter on the frozen P2 additions (plan 2026-10-05, menu frozen).

Stage 1 = the pair's frozen matched-seed P2 selection, unchanged and bound by input
sha256; stage 2 = one EF filter from the frozen menu {identity, off, a30, a30d, a60}
selected on the validation fold with the P2 gate/rule verbatim. The `off` row must
reproduce the frozen stage-1 row exactly, guarding the stage-1 reimplementation.
Test uses only the frozen choice, as in run_scale_fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from lever_postproc import file_provenance
from run_p0_controls import check_prediction
from analyze_scale_complementarity import provenance, validate_run_context, write_json
from run_scale_fusion import image_confidence
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics import light
from nucseg.metrics.pannuke_eval import evaluate, save_report
from nucseg.postproc.existence import existence_pass
from nucseg.postproc.scale_fusion import (candidate_info, fuse, probability_rows, select_fusion,
                                          strict_dead_pq, _gt_facts)

MENU = [{"name": "identity"},
        {"name": "off"},
        {"name": "a30", "min_area": 30},
        {"name": "a30d", "min_area": 30, "dead_exempt": True},
        {"name": "a60", "min_area": 60}]
_G = {}


def _ef_source(evidence):
    extra = sorted([ROOT / "src/nucseg/postproc/existence.py", ROOT / "scripts/run_scale_fusion.py"])
    evidence["source"].update({str(p.relative_to(ROOT)): file_provenance(p) for p in extra})
    return evidence


def _sweep(j):
    bi, bt, xi, xt = (_G[k][j] for k in ("bi", "bt", "xi", "xt"))
    conf = image_confidence(xi, xt, _G["prob"].get(j, {}))
    p2 = _G["p2cfg"]
    cands = candidate_info(bi, xi, xt)
    rows = []
    for m in _G["menu"]:
        if m["name"] == "identity" or p2 is None:
            pi, pt, added = bi, bt, []
        else:
            allowed = None
            if m.get("min_area", 0) > 0 or m.get("dead_exempt", False):
                allowed = {c["id"] for c in cands
                           if existence_pass(c, m.get("min_area", 0), m.get("dead_exempt", False))}
            pi, pt, added = fuse(bi, bt, xi, xt, conf, p2["max_area"], p2["min_prob"],
                                 p2["interior_only"], allowed_ids=allowed)
        stats = light.image_stats(_G["gc"][j], _G["gi"][j], _G["gt"][j], pi, pt)
        counts = _gt_facts(_G["gi"][j], _G["gt"][j], pi, pt)[-1]
        rows.append((stats, counts["interior_matched"], len(added)))
    return rows


def _apply(j):
    bi, bt, xi, xt = (_G[k][j] for k in ("bi", "bt", "xi", "xt"))
    conf = image_confidence(xi, xt, _G["prob"].get(j, {}))
    p2, m = _G["p2cfg"], _G["chosen"]
    if m["name"] == "identity" or p2 is None:
        pi, pt, added = bi, bt, []
    else:
        allowed = None
        if m.get("min_area", 0) > 0 or m.get("dead_exempt", False):
            allowed = {c["id"] for c in candidate_info(bi, xi, xt)
                       if existence_pass(c, m.get("min_area", 0), m.get("dead_exempt", False))}
        pi, pt, added = fuse(bi, bt, xi, xt, conf, p2["max_area"], p2["min_prob"],
                             p2["interior_only"], allowed_ids=allowed)
    return j, pi, pt, len(added)


def load_stage1(selection_path: Path):
    record = json.loads(selection_path.read_text())
    if record.get("split") is None or "selected" not in record:
        sys.exit(f"{selection_path}: not a frozen P2 selection")
    selected = record["selected"]
    stage1 = None if selected["name"] == "identity" else selected
    return record, stage1, selected


def run(a, role):
    destination = a.out / role
    if destination.exists():
        raise FileExistsError(destination)
    fold = split_folds(a.split)[1 if role == "val" else 2]
    f = PanNukeFold(fold)
    base, x2 = getattr(a, "base_" + role), getattr(a, "x2_" + role)
    p2_record, stage1, frozen_row = load_stage1(a.p2_selection)
    if p2_record["split"] != a.split:
        sys.exit(f"p2 selection split {p2_record['split']} does not match --split {a.split}")
    evidence = _ef_source(provenance({"base": base, "x2": x2}, f, __file__))
    validate_run_context(evidence, a.split, a.expect_seed)
    if p2_record.get("expected_seed", 19) != a.expect_seed:
        sys.exit(f"p2 selection expected_seed {p2_record.get('expected_seed')} != {a.expect_seed}")
    for name, path in (("base", a.base_val), ("x2", a.x2_val)):
        frozen_sha = p2_record["provenance"]["inputs"][name]["sha256"]
        if file_provenance(path)["sha256"] != frozen_sha:
            sys.exit(f"{name} validation prediction does not match the frozen stage-1 selection")
    choice_path = a.out / "selection.json"
    if role == "test":
        frozen = json.loads(choice_path.read_text())
        if frozen["split"] != a.split or frozen["provenance"]["source"] != evidence["source"]:
            raise ValueError("frozen split or code changed")
        if file_provenance(a.p2_selection)["sha256"] != frozen["p2_selection"]["sha256"]:
            raise ValueError("the stage-1 selection changed after the EF selection was frozen")
        for name, path in (("base", a.base_val), ("x2", a.x2_val)):
            if file_provenance(path)["sha256"] != frozen["provenance"]["inputs"][name]["sha256"]:
                raise ValueError("validation predictions changed")
        from run_p0_controls import ground_truth_provenance
        vf = PanNukeFold(split_folds(a.split)[1])
        if ground_truth_provenance(vf) != frozen["provenance"]["ground_truth"]:
            raise ValueError("validation annotations changed")
        menu = [frozen["selected"]]
    else:
        if choice_path.exists():
            raise FileExistsError(choice_path)
        menu = MENU
    with np.load(base) as b, np.load(x2) as x:
        bi, bt, xi, xt = b["inst"], b["type"], x["inst"], x["type"]
        check_prediction(bi, bt, f, b)
        check_prediction(xi, xt, f, x)
        prob = probability_rows(x)
        if any(j >= len(f) for j in prob):
            raise ValueError("probability table references an image outside the fold")
    _G.update(bi=bi, bt=bt, xi=xi, xt=xt, gi=f.inst, gt=f.type, gc=f.gt_channels,
              prob=prob, p2cfg=stage1, menu=menu)
    destination.mkdir(parents=True, exist_ok=False)
    if role == "val":
        with Pool(a.workers) as pool:
            results = pool.map(_sweep, range(len(f)), chunksize=4)
        rows = []
        for k, cfg in enumerate(menu):
            stats = np.stack([r[k][0] for r in results])
            s = light.summarize(stats, f.tissue)
            rows.append({**cfg, "mPQ": s["mPQ"], "bPQ": s["bPQ"],
                         "Dead_PQ": s["per_class_PQ"]["Dead"], "strict_dead": strict_dead_pq(stats),
                         "interior_matched": int(sum(r[k][1] for r in results)),
                         "added": int(sum(r[k][2] for r in results))})
        off = next(r for r in rows if r["name"] == "off")
        if stage1 is not None:
            for k in ("mPQ", "bPQ", "strict_dead", "interior_matched", "added"):
                if abs(off[k] - frozen_row[k]) > 1e-12:
                    sys.exit(f"stage-1 reproduction drifted on '{k}': off {off[k]} vs frozen "
                             f"{frozen_row[k]} — inputs or code do not match the frozen selection")
        selected = select_fusion(rows)
        record = {"split": a.split, "fold": fold, "role": "EF validation selection",
                  "expected_seed": a.expect_seed,
                  "status": "NO_GO" if selected["name"] == "identity" else "GO",
                  "reference": "P0 validation-selected corrected base M/S; frozen matched-seed P2 "
                               "additions filtered by the frozen EF menu",
                  "rule": "mPQ,bPQ >= base-.002; strict Dead PQ >= base; interior Dead matched count "
                          "increases; max mPQ, fewer additions tie-break (P2 rule verbatim)",
                  "p2_selection": {"path": str(a.p2_selection.resolve()),
                                   "sha256": file_provenance(a.p2_selection)["sha256"],
                                   "selected_name": p2_record["selected"]["name"]},
                  "rows": rows, "selected": selected, "provenance": evidence}
        write_json(choice_path, record)
        write_json(destination / "sweep.json", record)
        print(f"split{a.split} EF {record['status']}: {selected['name']}", flush=True)
    else:
        _G["chosen"] = menu[0]
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
                   "role": "frozen EF test; identity fallback if validation NO_GO",
                   "selected": menu[0], "added_instances": added, "provenance": evidence,
                   "selection_sha256": file_provenance(choice_path)["sha256"],
                   "p2_selection_sha256": file_provenance(a.p2_selection)["sha256"],
                   "prediction": file_provenance(pred)})
        print(f"split{a.split} EF test mPQ={result['summary']['official']['mPQ']:.6f}", flush=True)
    _G.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=int, required=True, choices=(1, 2, 3))
    for model in ("base", "x2"):
        for role in ("val", "test"):
            p.add_argument(f"--{model}-{role}", type=Path, required=True)
    p.add_argument("--p2-selection", type=Path, required=True, dest="p2_selection",
                   help="frozen matched-seed P2 selection.json for this pair (stage 1)")
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
