#!/usr/bin/env python
"""Compare evaluated runs on the same fold: seed spread + paired, tissue-stratified image bootstrap.

    python scripts/compare_runs.py --fold 3 \
        --group base runs/cellvit_uni/split1/eval_test_fold3 runs/cellvit_abl/split1_seed1/eval_test_fold3 \
        --group M2 runs/cellvit_abl/split1_npwce_small5/eval_test_fold3

The first group is the reference. A group may hold several runs (seeds); its metric is the mean over
runs. Metrics are recomputed from each eval dir's per_image.npz with the official aggregation (image
-> tissue nanmean -> mean over tissues for mPQ/bPQ; image nanmean for per-class PQ) and asserted equal
to its summary.json. The bootstrap resamples test images within each tissue (same resample for all
runs), so its CI reflects test-set sampling only; training noise is the across-seed std.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from nucseg.constants import CLASS_NAMES, TISSUES
from nucseg.data.pannuke import PanNukeFold

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True)
p.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "EVAL_DIR"))
p.add_argument("--boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json", type=Path)
a = p.parse_args()
warnings.simplefilter("ignore", RuntimeWarning)

tissue = np.asarray(PanNukeFold(a.fold).tissue)
tidx = [np.where(tissue == t)[0] for t in TISSUES if (tissue == t).any()]
METRICS = ["mPQ", "bPQ"] + CLASS_NAMES


def per_image(d: Path) -> np.ndarray:
    """(n_images, 7): image mPQ, image bPQ, 5 per-class image PQs."""
    z = np.load(d / "per_image.npz")
    return np.column_stack([z["mPQ"], z["bPQ"], z["class_PQ"]])


def aggregate(v: np.ndarray, idx: list[np.ndarray]) -> np.ndarray:
    """v (n, 7) or (B, n, 7) with idx = per-tissue image index arrays (each (m,) or (B, m))."""
    tis = [np.nanmean(v[..., i, :2] if i.ndim == 1 else np.take_along_axis(v[..., :2], i[..., None], -2), -2)
           for i in idx]
    allidx = np.concatenate(idx, -1)
    cls = np.nanmean(v[..., allidx, 2:] if allidx.ndim == 1 else np.take_along_axis(v[..., 2:], allidx[..., None], -2), -2)
    return np.concatenate([np.nanmean(np.stack(tis), 0), cls], -1)


groups = []
for g in a.group:
    name, dirs = g[0], [Path(x) for x in g[1:]]
    assert dirs, f"group {name} has no eval dirs"
    vals = np.stack([per_image(d) for d in dirs])  # (R, n, 7)
    assert vals.shape[1] == len(tissue), f"{name}: eval dirs are not for fold {a.fold}"
    point = np.stack([aggregate(v, tidx) for v in vals])  # (R, 7)
    for d, pt in zip(dirs, point):
        s = json.loads((d / "summary.json").read_text())
        ref = [s["official"]["mPQ"], s["official"]["bPQ"]] + [s["per_class_PQ"][c] for c in CLASS_NAMES]
        assert np.allclose(pt, ref, atol=1e-12, equal_nan=True), f"{d}: recomputed metrics != summary.json"
    groups.append((name, dirs, vals, point))

rng = np.random.default_rng(a.seed)
bidx = [i[rng.integers(0, len(i), (a.boot, len(i)))] for i in tidx]  # same resample for all runs
boot = {name: np.mean([aggregate(np.broadcast_to(v, (a.boot,) + v.shape), bidx) for v in vals], 0)
        for name, _, vals, _ in groups}  # (B, 7) group mean per bootstrap sample

ref_name = groups[0][0]
out = {"fold": a.fold, "reference": ref_name, "boot": a.boot, "groups": {}}
cols = ["mPQ", "bPQ", "Dead"]
ci = {"mPQ": 0, "bPQ": 1, "Dead": 2 + CLASS_NAMES.index("Dead")}
print(f"fold {a.fold}; mean +- std over runs; delta vs '{ref_name}' with 95% paired image-bootstrap CI")
print(f"{'group':14s} {'n':>2s}  " + "  ".join(f"{c:>15s}" for c in cols) + "  " + "  ".join(f"{'d' + c + ' [95% CI]':>24s}" for c in cols))
for name, dirs, vals, point in groups:
    mean, std = point.mean(0), point.std(0, ddof=1) if len(point) > 1 else np.full(point.shape[1], np.nan)
    row = f"{name:14s} {len(dirs):2d}  " + "  ".join(f"{mean[ci[c]]:.4f} +- {std[ci[c]]:.4f}" for c in cols)
    rec = {"runs": [str(d) for d in dirs], "per_run": point.tolist(), "mean": dict(zip(METRICS, mean.tolist())),
           "std": dict(zip(METRICS, std.tolist()))}
    if name != ref_name:
        diff = boot[name] - boot[ref_name]
        dmean = mean - groups[0][3].mean(0)
        lo, hi = np.nanpercentile(diff, 2.5, 0), np.nanpercentile(diff, 97.5, 0)
        row += "  " + "  ".join(f"{dmean[ci[c]]:+.4f} [{lo[ci[c]]:+.4f},{hi[ci[c]]:+.4f}]" for c in cols)
        rec["delta"] = dict(zip(METRICS, dmean.tolist()))
        rec["ci95"] = {m: [float(l), float(h)] for m, l, h in zip(METRICS, lo, hi)}
    print(row)
    out["groups"][name] = rec
if a.json:
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=2))
