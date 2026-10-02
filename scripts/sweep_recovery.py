#!/usr/bin/env python
"""Sweep training-free nucleus-recovery post-processing on saved CellViT-UNI outputs (CPU stage).

    python scripts/sweep_recovery.py --split 1 --maps-dir data/cache/maps/split1 \
        --run runs/cellvit_uni/split1 --out runs/recovery/cellvit_uni_split1 --workers 30

Needs maps_fold{val,test}.npy from dump_cellvit_maps.py. Every configuration is scored on the
VALIDATION fold (light official mPQ, tests/test_recovery.py); per family the best-by-val configuration
is then scored once on the TEST fold. Families (nucseg.postproc.recovery.Recovery):
  official  default post-processing (reference; should reproduce the pipeline result up to fp16 maps)
  thr       blob threshold only                          -- generic post-processing tuning control
  orphans   keep blobs without a watershed marker (no tuning)
  beta      mix NP with the TP-branch foreground (+ threshold)
  dead      boost the foreground with P_TP(Dead) (+ threshold)
  all       best over the full grid
Does not import torch, saving address space; forked workers share the parent's mappings.
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics.light import image_stats, summarize
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report
from nucseg.postproc.recovery import Recovery, postprocess

p = argparse.ArgumentParser()
p.add_argument("--split", type=int, required=True)
p.add_argument("--maps-dir", type=Path, required=True)
p.add_argument("--run", type=Path, help="segmenter run dir (for the pipeline-result sanity check)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--workers", type=int, default=16)
p.add_argument("--tta", action="store_true")
p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
p.add_argument("--ks", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
p.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
p.add_argument("--limit", type=int, default=0, help="smoke test: first N patches of each fold only")
a = p.parse_args()
assert "torch" not in sys.modules

tr, va, te = split_folds(a.split)
tag = "_tta" if a.tta else ""
GRID = [Recovery(b, k, t, o) for b, k, t, o in itertools.product(a.betas, a.ks, a.thrs, (False, True))]
assert Recovery() in GRID
G = {}


def _stats_job(i):
    m = np.asarray(G["maps"][i], dtype=np.float32)
    gt_ch, gi, gt = G["gt"][i], np.asarray(G["inst"][i]), np.asarray(G["type"][i])
    out = np.zeros((len(G["cfgs"]), 6, 5))
    for c, cfg in enumerate(G["cfgs"]):
        inst, typ = postprocess(m[..., 0], m[..., 1:3], m[..., 3:], cfg)
        out[c] = image_stats(gt_ch, gi, gt, inst, typ)
    return out


def _pred_job(i):
    m = np.asarray(G["maps"][i], dtype=np.float32)
    return postprocess(m[..., 0], m[..., 1:3], m[..., 3:], G["cfgs"][0])


class Subset:
    """First `n` patches of a fold (smoke tests)."""

    def __init__(self, f, n):
        self.gt_channels, self.inst, self.type = f.gt_channels[:n], f.inst[:n], f.type[:n]
        self.tissue = np.asarray(f.tissue)[:n]

    def __len__(self):
        return len(self.tissue)


def load(fold):
    f = PanNukeFold(fold)
    G.clear()
    maps = np.load(a.maps_dir / f"maps_fold{fold}{tag}.npy", mmap_mode="r")
    assert maps.shape[0] == len(f)
    if a.limit:
        f = Subset(f, a.limit)
    G.update(gt=f.gt_channels, inst=f.inst, type=f.type, maps=maps)
    return f


def score(f, cfgs):
    G["cfgs"] = cfgs
    with Pool(a.workers) as pool:
        st = np.stack(pool.map(_stats_job, range(len(f)), chunksize=4), 1)  # (C, N, 6, 5)
    return [summarize(st[c], f.tissue) for c in range(len(cfgs))], st


a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
fv = load(va)
val, val_stats = score(fv, GRID)
np.savez_compressed(a.out / f"val_stats{tag}.npz", stats=val_stats.astype(np.float32),
                    grid=np.array([list(c.as_dict().values()) for c in GRID], np.float32))
print(f"val fold {va}: {len(GRID)} configs in {(time.time() - t0) / 60:.1f} min", flush=True)
vm = {c: r["mPQ"] for c, r in zip(GRID, val)}

families = {
    "official": [Recovery()],
    "thr": [c for c in GRID if c.beta == 0 and c.k_dead == 0 and not c.orphans],
    "orphans": [Recovery(orphans=True)],
    "beta": [c for c in GRID if c.k_dead == 0 and not c.orphans],
    "dead": [c for c in GRID if c.beta == 0 and not c.orphans],
    "all": GRID,
}
chosen = {name: max(cands, key=lambda c: vm[c]) for name, cands in families.items()}
uniq = list(dict.fromkeys(chosen.values()))

fte = load(te)
test, test_stats = score(fte, uniq)
tm = dict(zip(uniq, test))
np.savez_compressed(a.out / f"test_stats{tag}.npz", stats=test_stats.astype(np.float32),
                    grid=np.array([list(c.as_dict().values()) for c in uniq], np.float32))

results = {"split": a.split, "val_fold": va, "test_fold": te, "tta": a.tta, "maps": str(a.maps_dir),
           "grid_val": [{**c.as_dict(), **r} for c, r in zip(GRID, val)], "families": {}}
print(f"{'family':9s} {'config':22s} {'val mPQ':>8s} {'test mPQ':>9s} {'test bPQ':>9s}  test PQ Neo/Inf/Con/Dead/Epi")
for name, c in chosen.items():
    r = tm[c]
    results["families"][name] = {"config": c.as_dict(), "val": dict(zip(GRID, val))[c], "test": r}
    print(f"{name:9s} {c.name():22s} {vm[c]:8.4f} {r['mPQ']:9.4f} {r['bPQ']:9.4f}  "
          + " ".join(f"{v:.3f}" for v in r["per_class_PQ"].values()))

if a.run is not None:
    ref = a.run / f"eval_test_fold{te}{tag}" / "summary.json"
    if ref.exists():
        s = json.loads(ref.read_text())["official"]
        d = tm[Recovery()]
        results["pipeline_reference"] = {"mPQ": s["mPQ"], "bPQ": s["bPQ"],
                                         "diff_mPQ": d["mPQ"] - s["mPQ"], "diff_bPQ": d["bPQ"] - s["bPQ"]}
        print(f"sanity: official config from fp16 maps vs pipeline: mPQ {d['mPQ']:.4f} vs {s['mPQ']:.4f}, "
              f"bPQ {d['bPQ']:.4f} vs {s['bPQ']:.4f}")
(a.out / f"results{tag}.json").write_text(json.dumps(results, indent=2))

for name in ("official", "all"):
    G["cfgs"] = [chosen[name]]
    with Pool(a.workers) as pool:
        res = pool.map(_pred_job, range(len(fte)), chunksize=4)
    full = evaluate(fte.gt_channels, fte.inst, fte.type, fte.tissue, np.stack([r[0] for r in res]), np.stack([r[1] for r in res]),
                    workers=a.workers)
    assert abs(full["summary"]["official"]["mPQ"] - tm[chosen[name]]["mPQ"]) < 1e-9
    save_report(full, a.out / f"eval_test_{name}{tag}")
    print(f"--- {name} {chosen[name].name()} (full eval, test fold {te})\n" + format_summary(full["summary"]), flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")
