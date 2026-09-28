#!/usr/bin/env python
"""Benchmark-artifact decomposition of PanNuke test-fold errors (2026-09-28 analysis).

Reproduces the tables in docs/findings.md (2026-09-28 evening entry) from existing eval dirs:
  1. patch-border fragment statistics (what share of GT / FN touch the 256 px border),
  2. interior-only missed_bg by class and GT-area bin,
  3. cross-architecture consensus (how many models miss the same nucleus),
  4. official vs Uterus-excluded Dead PQ per split (official per-class PQ averages
     per-image PQ over the images that CONTAIN the class),
  5. Dead F_c (HoVer-Net definition) per split,
  6. Dead co-missing / clustering context stats.

Inputs are the `eval_test_fold*` directories written by scripts/eval_pannuke.py (summary.json,
per_image.npz, gt_records.csv.gz). CellViT-UNI and HoVer-Net live in runs/ on LM1; HoVer-NeXt-T
evals live on LM2 under runs/hovernext_t/ -- rsync them into one local root, e.g.
  /tmp/hnxt/split1/eval_test_fold3, /tmp/hnxt/split2/eval_test_fold3, /tmp/hnxt/split3/eval_test_fold1
and pass --hovernext /tmp/hnxt. Splits with missing eval dirs are skipped with a warning.

    python scripts/analyze_artifacts.py [--hovernext /tmp/hnxt] [--out-gt runs/analysis/artifacts_gt.csv.gz]

--out-gt is opt-in (no default): pass it to cache the per-GT consensus frame for ad-hoc analysis.

The consensus section uses the two splits with DISTINCT test folds (1 and 3 -> test folds 3 and 1)
so no GT nucleus is counted twice; the per-model section uses all three splits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.instance import centroids, relabel

TEST_FOLD = {1: 3, 2: 3, 3: 1}  # official split -> test fold
BINS = [(0, 60), (60, 100), (100, 256), (256, 10**6)]


def eval_dir(root: Path, split: int) -> Path | None:
    d = root / f"split{split}" / f"eval_test_fold{TEST_FOLD[split]}"
    return d if (d / "summary.json").exists() else None


def load_gt(root: Path, split: int) -> pd.DataFrame | None:
    d = eval_dir(root, split)
    if d is None:
        print(f"[warn] {root}: split {split} eval dir missing, skipped")
        return None
    g = pd.read_csv(d / "gt_records.csv.gz")
    g["split"] = split
    return g


def border_and_centroids(fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-GT border flag and centroid (cy, cx), in the evaluator's relabelled instance order."""
    f = PanNukeFold(fold)
    borders, cen = [], []
    for i in range(len(f)):
        inst = np.asarray(f.inst[i])
        lab, n = relabel(inst)
        if n == 0:
            continue
        edge_ids = set()
        for edge in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
            edge_ids.update(np.unique(edge).tolist())
        b = np.zeros(n + 1, bool)
        b[list(edge_ids)] = True
        _, c = centroids(inst)  # relabels identically; returns (n, 2) in xy order
        borders.append(b[1:])
        cen.append(c[:, ::-1])  # -> (cy, cx)
    return np.concatenate(borders), np.concatenate(cen)


def fmt_pq(z: np.ndarray, tissue: np.ndarray, exclude: str) -> float:
    dead = z["class_PQ"][:, 3]
    has = ~np.isnan(dead)
    keep = has & (tissue != exclude) if exclude else has
    return float(np.nanmean(dead[keep]))


def per_model(name: str, root: Path) -> dict:
    rows, uter = [], {}
    for split in (1, 2, 3):
        d = eval_dir(root, split)
        if d is None:
            print(f"[warn] {name}: split {split} missing")
            continue
        s = json.load(open(d / "summary.json"))
        z = np.load(d / "per_image.npz")
        # per-image tissue straight from the eval dir's gt_records (no PanNuke re-load needed);
        # images without GT nuclei get NaN but their class_PQ is NaN anyway, so `has` masks them
        tis = (pd.read_csv(d / "gt_records.csv.gz").drop_duplicates("image")
               .set_index("image")["tissue"].reindex(range(z["class_PQ"].shape[0])).values)
        dead = z["class_PQ"][:, 3]
        has = ~np.isnan(dead)
        ut = has & (tis == "Uterus")
        rows.append((split, s["official"]["mPQ"], np.nanmean(dead), fmt_pq(z, tis, "Uterus"),
                     int(has.sum()), int(ut.sum()),
                     float(np.nanmean(dead[ut])) if ut.any() else float("nan"),
                     s["detection"]["F_c"]["Dead"]))
    print(f"\n== {name}")
    print(f"{'split':>5} {'mPQ':>6} {'DeadPQ':>7} {'-Uterus':>8} {'n_img':>5} {'n_ut':>4} {'ut_DeadPQ':>9} {'DeadF_c':>7}")
    for r in rows:
        print(f"{r[0]:>5} {r[1]:>6.3f} {r[2]:>7.3f} {r[3]:>8.3f} {r[4]:>5} {r[5]:>4} {r[6]:>9.3f} {r[7]:>7.3f}")
    if rows:
        print(f"{'mean':>5} {np.mean([r[1] for r in rows]):>6.3f} {np.mean([r[2] for r in rows]):>7.3f} "
              f"{np.mean([r[3] for r in rows]):>8.3f} {'':>5} {'':>4} {'':>9} {np.mean([r[7] for r in rows]):>7.3f}"
              f"   ({len(rows)} splits)")
    return {"rows": rows}


def consensus(models: dict[str, Path], splits=(1, 3)) -> pd.DataFrame:
    frames = {}
    for split in splits:
        fold = TEST_FOLD[split]
        gs, miss = {}, {}
        for name, root in models.items():
            g = load_gt(root, split)
            if g is None:
                raise SystemExit(f"consensus needs all models on split {split}")
            gs[name], miss[name] = g, None
        base = gs[next(iter(gs))]
        for name, g in gs.items():
            assert (g[["image", "cls", "area"]].values == base[["image", "cls", "area"]].values).all(), \
                f"GT rows not aligned: {name} split {split}"
            miss[name] = (g.status == "missed_bg").values
        border, cen = border_and_centroids(fold)
        assert len(border) == len(base), (len(border), len(base))
        df = base[["image", "cls", "area", "n_touch", "tissue"]].copy()
        df["k"] = np.stack(list(miss.values()), 1).sum(1)
        df["border"] = border
        df[["cy", "cx"]] = cen
        df["fold"] = fold
        for name, m in miss.items():
            df[f"miss_{name}"] = m
        frames[split] = df
    return pd.concat(frames.values(), ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cellvit", type=Path, default=Path("runs/cellvit_uni"))
    ap.add_argument("--hovernet", type=Path, default=Path("runs/hovernet"))
    ap.add_argument("--hovernext", type=Path, default=None,
                    help="local root with HoVer-NeXt-T eval dirs (rsync from LM2 runs/hovernext_t)")
    ap.add_argument("--out-gt", type=Path, default=None, help="write the per-GT consensus frame here")
    a = ap.parse_args()

    models = {"CellViT-UNI": a.cellvit, "HoVer-Net": a.hovernet}
    if a.hovernext:
        models["HoVer-NeXt-T"] = a.hovernext
    for name, root in models.items():
        per_model(name, root)

    df = consensus(models)
    cls = {i + 1: n for i, n in enumerate(CLASS_NAMES)}
    df["c"] = df.cls.map(cls)
    n_models = len(models)

    print(f"\n== consensus over {n_models} models, test folds 3+1 (n={len(df)} GT)")
    print(f"GT border share overall {df.border.mean():.3f}")
    for lo, hi in BINS:
        s = df[(df.area >= lo) & (df.area < hi)]
        ks = s.k.value_counts(normalize=True).sort_index()
        print(f"area {lo:>3}-{hi if hi < 10**6 else 'inf'}: border {s.border.mean():.2f} | "
              + " ".join(f"missed-by-{i}: {ks.get(i, 0):.2f}" for i in range(n_models + 1))
              + f" (n={len(s)})")

    print("\n== CellViT-UNI missed_bg by class x area, ALL vs INTERIOR (no border contact)")
    for c in CLASS_NAMES:
        s = df[df.c == c]
        cells = []
        for lo, hi in BINS:
            b = s[(s.area >= lo) & (s.area < hi)]
            cells.append(f"{b['miss_CellViT-UNI'].mean():.2f}/{b[~b.border]['miss_CellViT-UNI'].mean():.2f}")
        interior = s[~s.border]
        print(f"{c:<14}" + "  ".join(f"{c9:>13}" for c9 in cells)
              + f"   interior_all {interior['miss_CellViT-UNI'].mean():.3f} (n={len(interior)})")

    print("\n== Dead interior (no border) by tissue, CellViT-UNI miss / all-%d-miss" % n_models)
    d = df[(df.c == "Dead") & (~df.border)]
    t = d.groupby("tissue").agg(n=("k", "size"), cv=("miss_CellViT-UNI", "mean"),
                                allmiss=("k", lambda k: (k == n_models).mean()), med_area=("area", "median"))
    print(t[t.n >= 20].sort_values("n", ascending=False).round(2).to_string())

    # co-missing context: non-Dead interior nuclei near a consensus-missed Dead
    res = []
    for (_, _), s in df.groupby(["fold", "image"]):
        dm = s[(s.c == "Dead") & (s.k == n_models) & (~s.border)]
        ot = s[(s.c != "Dead") & (~s.border)]
        if len(dm) == 0 or len(ot) == 0:
            continue
        dist, _ = cKDTree(dm[["cy", "cx"]].values).query(ot[["cy", "cx"]].values)
        res.append(pd.DataFrame({"near": dist < 40, "miss3": ot.k == n_models}))
    r = pd.concat(res)
    base = df[(df.c != "Dead") & (~df.border)]
    print(f"\n== context: interior non-Dead all-{n_models}-missed near (<40px) vs far from a consensus-missed Dead: "
          f"{r[r.near].miss3.mean():.3f} vs {r[~r.near].miss3.mean():.3f} (baseline {float((base.k == n_models).mean()):.3f})")
    clustered = []
    for (_, _), s in df.groupby(["fold", "image"]):
        dead = s[s.c == "Dead"]
        if len(dead) == 0:
            continue
        t2 = cKDTree(dead[["cy", "cx"]].values)
        nn = np.array([len(t2.query_ball_point(p, 40)) - 1 for p in dead[["cy", "cx"]].values])
        clustered.append(pd.DataFrame({"clustered": nn >= 3, "k": dead.k.values, "tissue": dead.tissue.values,
                                       "interior": ~dead.border.values}))
    C = pd.concat(clustered)
    Ci = C[C.interior]
    print(f"Dead in clusters (>=3 Dead within 40px): all-{n_models}-missed {C[C.clustered].k.eq(n_models).mean():.3f} "
          f"vs scattered {C[~C.clustered].k.eq(n_models).mean():.3f}; interior-only: "
          f"{Ci[Ci.clustered].k.eq(n_models).mean():.3f} vs {Ci[~Ci.clustered].k.eq(n_models).mean():.3f}")

    if a.out_gt:
        a.out_gt.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["cy", "cx"]).to_csv(a.out_gt, index=False)
        print(f"[written] {a.out_gt}")


if __name__ == "__main__":
    main()
