#!/usr/bin/env python
"""Fusion-gating dev endpoints (spec docs/superpowers/specs/2026-10-15-fusion-gating-design.md).

Fold2 original-image endpoints for arbitrary run dirs plus the frozen paired comparison against the
same-recipe A0 baselines. Reads saved predictions and summaries; no training, no model selection,
no test-fold access, no threshold search.

    python scripts/analyze_fusion_arms.py --baseline runs/visibility_supervision_20261014 \
        --arm-name G --arm runs/fusion_gating_20261015 --seeds 19 1 \
        --out runs/analysis/fusion_gating_20261015
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT_SEED = 20261015
N_BOOT = 1000
DEAD = 4
GUARD = -0.002          # mPQ / bPQ / strict mPQ / strict Dead PQ floor, relative to the baseline
MERGE_GUARD = 0.002     # merge rate may not rise by more than this
DEAD_GAIN = 0.005       # required mean official Dead PQ improvement
MATCH_GAIN = 0.02       # required mean interior-Dead matched-rate improvement


def provenance(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def fold_diagnostics(fold, pred_inst: np.ndarray) -> dict:
    """GT merge rate and prediction split rate from the existing error taxonomy; no new rules."""
    from nucseg.metrics.errors import classify_errors
    from nucseg.metrics.instance import overlap
    merged = split = n_gt = n_pred = 0
    for j in range(len(fold)):
        gt, pi = np.asarray(fold.inst[j]), pred_inst[j]
        status, _, _ = classify_errors(overlap(gt, pi))
        merged += int((status == 1).sum())
        split += int((status == 2).sum())
        n_gt += int((np.unique(gt) > 0).sum())
        n_pred += int((np.unique(pi) > 0).sum())
    return {"n_gt": n_gt, "n_pred": n_pred, "merged": merged, "split": split,
            "merge_rate": merged / max(n_gt, 1), "split_rate": split / max(n_pred, 1)}


def load_run(run: Path, fold, expected_fusion=None) -> dict:
    """Per-GT-object rows (class, interior/border, matched >.5, miss <.1) plus official endpoints."""
    from nucseg.visibility_probe import max_iou_table, original_full
    sys.path.insert(0, str(ROOT / "scripts"))
    from probe_visibility import gt_class_map
    cfg = json.loads((run / "config.json").read_text())
    if cfg["split"] != 1 or cfg.get("upscale", 1) != 1 or cfg.get("dead_expert") or cfg.get("widen"):
        raise ValueError(f"{run.name}: not a split1 / native-resolution baseline-form run")
    if expected_fusion is not None and cfg.get("fusion", "plain") != expected_fusion:
        raise ValueError(f"{run.name}: fusion is {cfg.get('fusion', 'plain')!r}, expected {expected_fusion!r}")
    base = dict(np.load(run / "pred_fold2.npz"))
    n = len(fold)
    if base["inst"].shape != (n, 256, 256):
        raise ValueError(f"{run.name}: predictions are not native fold2 instance maps")
    rows = []
    for j in range(n):
        gt = np.asarray(fold.inst[j])
        classes, full = gt_class_map(gt, np.asarray(fold.type[j])), original_full(gt)
        for i, u in max_iou_table(gt, base["inst"][j]).items():
            rows.append({"image": j, "tissue": str(fold.tissue[j]), "gt_id": int(i), "cls": int(classes[i]),
                         "interior": bool(full[i]), "matched": bool(u > .5), "miss": bool(u < .1),
                         "max_iou": float(u)})
    summary = json.loads((run / "eval_fold2/summary.json").read_text())
    endpoints = {"mPQ": summary["official"]["mPQ"], "bPQ": summary["official"]["bPQ"],
                 "strict_mPQ": summary["strict"]["mPQ"],
                 "dead_PQ": summary["per_class_PQ"]["Dead"],
                 "strict_dead_PQ": summary["per_class_PQ_strict"]["Dead"]}
    dead = [r for r in rows if r["cls"] == DEAD and r["interior"]]
    endpoints["interior_dead_n"] = len(dead)
    endpoints["interior_dead_matched"] = float(np.mean([r["matched"] for r in dead]))
    endpoints["interior_dead_miss"] = float(np.mean([r["miss"] for r in dead]))
    endpoints.update(fold_diagnostics(fold, base["inst"]))
    return {"endpoints": endpoints, "rows": rows, "wall_config": cfg}


def paired(rows_by_run: dict, left: str, right: str, seeds: tuple) -> dict:
    """Mean-of-seeds interior-Dead matched-rate improvement, paired by GT object across arms and
    seeds, with a tissue-stratified image-cluster bootstrap that uses ONE shared draw per step."""
    from nucseg.visibility_probe import resample_rows
    tables, keys = {}, None
    for s in seeds:
        for arm in (left, right):
            rr = [r for r in rows_by_run[(arm, s)] if r["cls"] == DEAD and r["interior"]]
            table = {(r["image"], r["gt_id"]): r for r in rr}
            if len(table) != len(rr):
                raise ValueError("duplicate object identity")
            tables[(arm, s)] = table
            keys = set(table) if keys is None else keys
            if set(table) != keys:
                raise ValueError("paired object identities differ across arms/seeds")
    keys = sorted(keys)
    img = np.array([k[0] for k in keys], int)
    tissue = np.array([tables[(left, seeds[0])][k]["tissue"] for k in keys])
    diff_matched = {s: np.array([int(tables[(left, s)][k]["matched"]) - int(tables[(right, s)][k]["matched"])
                                 for k in keys], float) for s in seeds}
    diff_miss = {s: -diff_matched[s] for s in seeds}
    point = {str(s): float(diff_matched[s].mean()) for s in seeds}
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(N_BOOT)
    n = np.empty(N_BOOT, int)
    for k in range(N_BOOT):
        picked = resample_rows(img, tissue, rng)
        draws[k] = float(np.mean([diff_matched[s][picked].mean() for s in seeds])) if len(picked) else np.nan
        n[k] = len(picked)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    mean_point = float(np.mean(list(point.values())))
    return {"left": left, "right": right, "n_objects": len(keys),
            "n_images": int(len(np.unique(img))), "per_seed_matched_improvement": point,
            "matched_improvement": mean_point, "lo": float(lo), "hi": float(hi),
            "miss_improvement": -mean_point,
            "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
            "bootstrap_rows_mean": float(np.mean(n)),
            "pass_clause1": bool(all(v > 0 for v in point.values()) and mean_point >= MATCH_GAIN and lo > 0)}


def guardrails(arm_eps: dict, base_eps: dict, seeds: tuple) -> dict:
    deltas = {}
    for key in ("mPQ", "bPQ", "strict_mPQ", "strict_dead_PQ", "dead_PQ", "merge_rate"):
        per_seed = {str(s): arm_eps[s][key] - base_eps[s][key] for s in seeds}
        deltas[key] = {"per_seed": per_seed, "mean": float(np.mean(list(per_seed.values())))}
    dead_ok = all(deltas["dead_PQ"]["per_seed"][str(s)] >= 0 for s in seeds) and deltas["dead_PQ"]["mean"] >= DEAD_GAIN
    floors = all(deltas[key]["per_seed"][str(s)] >= GUARD for key in ("mPQ", "bPQ", "strict_mPQ", "strict_dead_PQ")
                 for s in seeds)
    merge_ok = all(deltas["merge_rate"]["per_seed"][str(s)] <= MERGE_GUARD for s in seeds)
    return {"deltas": deltas, "pass_clause2_dead_gain": bool(dead_ok),
            "pass_clause3_guardrails": bool(floors and merge_ok),
            "guard_floor": GUARD, "merge_guard": MERGE_GUARD}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="run root holding <baseline-prefix>_seed<seed>")
    p.add_argument("--baseline-prefix", default="A0")
    p.add_argument("--arm", type=Path, required=True, help="run root holding <arm-name>_seed<seed>")
    p.add_argument("--arm-name", required=True)
    p.add_argument("--arm-fusion", default="gate")
    p.add_argument("--seeds", type=int, nargs="+", default=[19, 1])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        raise SystemExit("output exists; no overwrite/replay")
    args.out.mkdir(parents=True)
    from nucseg.data.pannuke import PanNukeFold
    fold = PanNukeFold(2)
    start = time.time()
    rows_by_run, endpoints, sources = {}, {}, {}
    for label, root, prefix, fusion in (("baseline", args.baseline, args.baseline_prefix, "plain"),
                                        ("arm", args.arm, args.arm_name, args.arm_fusion)):
        for seed in args.seeds:
            name = f"{prefix}_seed{seed}"
            run = root / name
            arm = "A0" if label == "baseline" else args.arm_name
            loaded = load_run(run, fold, expected_fusion=fusion)
            rows_by_run[(arm, seed)] = loaded["rows"]
            endpoints[f"{name}"] = loaded["endpoints"]
            sources[name] = {"config": provenance(run / "config.json"),
                             "pred": provenance(run / "pred_fold2.npz"),
                             "eval": provenance(run / "eval_fold2/summary.json"),
                             "final": provenance(run / "final.pth") if (run / "final.pth").exists() else None}
            (args.out / "status.json").write_text(json.dumps({"state": "running", "run": name}))
            print(f"{name}: interior Dead n={loaded['endpoints']['interior_dead_n']} "
                  f"matched={loaded['endpoints']['interior_dead_matched']:.6f} "
                  f"mPQ={loaded['endpoints']['mPQ']:.7f}", flush=True)
    base_eps = {s: endpoints[f"{args.baseline_prefix}_seed{s}"] for s in args.seeds}
    arm_eps = {s: endpoints[f"{args.arm_name}_seed{s}"] for s in args.seeds}
    comparison = paired(rows_by_run, args.arm_name, "A0", tuple(args.seeds))
    guards = guardrails(arm_eps, base_eps, tuple(args.seeds))
    verdict = {"clause1_interior_dead_matched": comparison["pass_clause1"],
               "clause2_official_dead_pq": guards["pass_clause2_dead_gain"],
               "clause3_quality_guardrails": guards["pass_clause3_guardrails"]}
    verdict["dev_admission"] = bool(all(verdict.values()))
    payload = {"arm": args.arm_name, "arm_fusion": args.arm_fusion, "seeds": list(args.seeds),
               "fold": 2, "interior_dead_endpoint": {"definition": "max-IoU>.5 vs any predicted instance, "
               "GT Dead objects not touching the patch border; miss = max-IoU<.1",
               "baseline": {str(s): {k: base_eps[s][k] for k in ("interior_dead_n", "interior_dead_matched",
                                                                "interior_dead_miss")} for s in args.seeds},
               "arm": {str(s): {k: arm_eps[s][k] for k in ("interior_dead_n", "interior_dead_matched",
                                                           "interior_dead_miss")} for s in args.seeds}},
               "comparison": comparison, "guardrails": guards, "verdict": verdict,
               "endpoints": endpoints, "sources": sources,
               "script": provenance(Path(__file__)), "wall_seconds": time.time() - start}
    (args.out / "endpoints.json").write_text(json.dumps(payload, indent=2))
    (args.out / "status.json").write_text(json.dumps({"state": "complete", "verdict": verdict}))
    print(json.dumps({"comparison": comparison, "guardrails": guards["deltas"], "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
