#!/usr/bin/env python
"""Paired-by-seed statistics for the matched-seed P0/P2 round (pre-registered plan §5).

    python scripts/matched_seed_stats.py --root runs/analysis/matched_seed_20261004 \
        --out runs/analysis/matched_seed_20261004/stats

Per (split, seed) pair the endpoints come from the frozen test evals:
baseline = p0/<pair>/test/base/ms/eval, P2 = p2/<pair>/test/eval, and the M/S
secondary delta uses p0/<pair>/test/base/identity/eval. Deltas are paired by
seed; per split we report the seed mean, population std over 3 seeds and sign
counts, the 3-split means feed the pre-registered decision rule verbatim, and
each split gets a tissue-stratified paired-image bootstrap (fixed seed, same
resample for every seed) of the 3-seed pooled mPQ/bPQ/Dead-PQ delta. Strict and
+ endpoints have no per-image arrays and stay point estimates, as in the v2
round. The official aggregation (image -> tissue nanmean -> tissue mean for
mPQ/bPQ; image nanmean for class PQ) is copied from scripts/compare_runs.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.constants import CLASS_NAMES, SPLITS, TISSUES
from nucseg.data.pannuke import PanNukeFold

SEEDS = (19, 1, 2)
ENDPOINTS = ("mPQ", "bPQ", "mPQ+", "strict mPQ", "Dead PQ", "strict Dead PQ")
DEAD = CLASS_NAMES.index("Dead")


def endpoints_of(summary: dict) -> dict:
    return {"mPQ": summary["official"]["mPQ"], "bPQ": summary["official"]["bPQ"],
            "mPQ+": summary["plus"]["mPQ+"], "strict mPQ": summary["strict"]["mPQ"],
            "Dead PQ": summary["per_class_PQ"]["Dead"],
            "strict Dead PQ": summary["per_class_PQ_strict"]["Dead"]}


def load_eval(directory: Path, tissue_idx: list[np.ndarray]):
    """Endpoints from summary.json plus (n, 3) per-image mPQ/bPQ/Dead-PQ arrays."""
    summary = json.loads((directory / "summary.json").read_text())
    z = np.load(directory / "per_image.npz")
    per_image = np.column_stack([z["mPQ"], z["bPQ"], z["class_PQ"][:, DEAD]])
    rebuilt = {"mPQ": float(np.mean([np.nanmean(per_image[i, 0]) for i in tissue_idx])),
               "bPQ": float(np.mean([np.nanmean(per_image[i, 1]) for i in tissue_idx]))}
    for key in ("mPQ", "bPQ"):
        if not np.isclose(rebuilt[key], summary["official"][key], atol=1e-10, rtol=0):
            sys.exit(f"{directory}: summary {key} disagrees with per-image aggregation "
                     f"({summary['official'][key]} vs {rebuilt[key]})")
    return endpoints_of(summary), per_image


def bootstrap(deltas: dict[str, np.ndarray], tissue: np.ndarray, boot: int, seed: int) -> dict:
    """Tissue-stratified paired-image bootstrap of the 3-seed pooled delta, same resample per seed."""
    rng = np.random.default_rng(seed)
    n = len(tissue)
    idx_all = np.empty((boot, n), dtype=int)
    for t in TISSUES:
        rows = np.where(tissue == t)[0]
        if len(rows):
            idx_all[:, rows] = rows[rng.integers(0, len(rows), (boot, len(rows)))]
    pooled = np.mean([d[idx_all] for d in deltas.values()], axis=0)  # (B, n, 3)
    tissue_rows = [np.where(tissue == t)[0] for t in TISSUES if (tissue == t).any()]
    mPQ = np.mean([np.nanmean(pooled[:, i, 0], axis=1) for i in tissue_rows], axis=0)  # (B,)
    bPQ = np.mean([np.nanmean(pooled[:, i, 1], axis=1) for i in tissue_rows], axis=0)
    dead = np.nanmean(pooled[:, :, 2], axis=1)
    return {k: {"ci95": [float(x) for x in np.nanpercentile(s, [2.5, 97.5])]}
            for k, s in (("mPQ", mPQ), ("bPQ", bPQ), ("Dead PQ", dead))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", default="p2",
                   help="round-local directory name of the comparison arm (default p2)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261004)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    a.out.mkdir(parents=True)
    pairs, splits = {}, {}
    for split in (1, 2, 3):
        fold = SPLITS[split][2]
        tissue = np.asarray(PanNukeFold(fold).tissue)
        tissue_idx = [np.where(tissue == t)[0] for t in TISSUES if (tissue == t).any()]
        per_seed_delta = {}
        for seed in SEEDS:
            pair = f"split{split}_seed{seed}"
            try:
                base_eps, base_pi = load_eval(a.root / "p0" / pair / "test" / "base" / "ms" / "eval", tissue_idx)
                ident_eps, _ = load_eval(a.root / "p0" / pair / "test" / "base" / "identity" / "eval", tissue_idx)
                p2_eps, p2_pi = load_eval(a.root / a.arm / pair / "test" / "eval", tissue_idx)
            except (FileNotFoundError, KeyError) as e:
                sys.exit(f"missing artifacts for pair {pair}: {e}")
            delta = {k: p2_eps[k] - base_eps[k] for k in ENDPOINTS}
            ms_delta = {k: base_eps[k] - ident_eps[k] for k in ENDPOINTS}
            pairs[pair] = {"split": split, "seed": seed, "baseline": base_eps, "p2": p2_eps,
                           "delta": delta, "ms_delta": ms_delta}
            per_seed_delta[seed] = p2_pi - base_pi
        splits[split] = {"seed_mean": {k: float(np.mean([pairs[f"split{split}_seed{s}"]["delta"][k] for s in SEEDS]))
                                       for k in ENDPOINTS},
                         "seed_std": {k: float(np.std([pairs[f"split{split}_seed{s}"]["delta"][k] for s in SEEDS]))
                                      for k in ENDPOINTS},
                         "sign_counts": {k: {"nonneg": sum(pairs[f"split{split}_seed{s}"]["delta"][k] >= 0 for s in SEEDS),
                                             "neg": sum(pairs[f"split{split}_seed{s}"]["delta"][k] < 0 for s in SEEDS)}
                                         for k in ENDPOINTS}}
        boot_out = bootstrap(per_seed_delta, tissue, a.boot, a.seed)
        for k, v in boot_out.items():
            v["point"] = float(splits[split]["seed_mean"][k])
            v["boot"], v["seed"] = a.boot, a.seed
        (a.out / f"bootstrap_split{split}.json").write_text(json.dumps(boot_out, indent=1))

    three_split_mean = {k: float(np.mean([splits[s]["seed_mean"][k] for s in (1, 2, 3)])) for k in ENDPOINTS}
    max_std = {k: max(splits[s]["seed_std"][k] for s in (1, 2, 3)) for k in ENDPOINTS}
    overall_signs = {k: {"nonneg": sum(pairs[f"split{s}_seed{e}"]["delta"][k] >= 0 for s in (1, 2, 3) for e in SEEDS),
                         "neg": sum(pairs[f"split{s}_seed{e}"]["delta"][k] < 0 for s in (1, 2, 3) for e in SEEDS)}
                     for k in ENDPOINTS}
    failed = []
    if three_split_mean["Dead PQ"] < max_std["Dead PQ"]:
        failed.append(f"Dead PQ 3-split mean {three_split_mean['Dead PQ']:.5f} < 1x largest per-split seed std "
                      f"{max_std['Dead PQ']:.5f}")
    if overall_signs["Dead PQ"]["nonneg"] < 7:
        failed.append(f"Dead PQ non-negative pairs {overall_signs['Dead PQ']['nonneg']}/9 < 7/9")
    for k in ("mPQ", "bPQ"):
        if three_split_mean[k] < -max_std[k]:
            failed.append(f"{k} 3-split mean {three_split_mean[k]:.5f} < -1x largest per-split seed std "
                          f"{max_std[k]:.5f}")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "root": str(a.root.resolve()),
              "endpoints": ENDPOINTS, "pairs": pairs, "splits": {str(s): splits[s] for s in splits},
              "overall": {"three_split_mean": three_split_mean, "max_split_seed_std": max_std,
                          "sign_counts": overall_signs},
              "decision": {"rule": "Dead PQ mean >= 1x max split seed std; >= 7/9 pairs Dead >= 0; "
                                  "mPQ/bPQ mean >= -1x their max split seed std",
                           "robust_improvement": not failed, "failed": failed}}
    (a.out / "stats.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"robust_improvement": not failed, "failed": failed,
                      "three_split_mean": three_split_mean}, indent=1))


if __name__ == "__main__":
    main()
