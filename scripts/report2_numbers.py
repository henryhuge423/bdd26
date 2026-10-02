"""Historical stage-report-2 extraction (pre-2026-10-02 audit outputs).

The old du2lev outputs and run-grid summaries are retained for audit, not corrected
M/S or distinct-seed claims. Use scripts/p0p3_numbers.py for the corrected follow-up.

Dumps (stdout, save to runs/analysis/report2_numbers.txt):
  [A] strict-protocol architecture table: KongNet / LKCell-L / CellViT-UNI base / x2 arms,
      same endpoint set as x2_grid_sum.views()
  [B] du2 recovery: 3-split seed-19 means of base / x2 / x2+du2, TTA + noTTA
  [C] B1 initial split-1 (x2 vs 3-seed base): printed JSONs (paired bootstrap CIs)
  [D] lever: per-split du2lev - du2 from lever_s*.json + per-run replication from grid dirs
  [E] size-binned mechanism (split-1 TTA, x2 vs 3-seed base): match-rate / matched-IoU deltas
  [F] error composition (split-1 TTA): GT split-rate by class, pred counts, <80px share,
      npred/image

Run: ~/.conda/envs/nuclei/bin/python scripts/report2_numbers.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from nucseg.metrics.instance import relabel
from nucseg.data.pannuke import PanNukeFold

SPLITS = {1: 3, 2: 3, 3: 1}  # split -> test fold

# ---------------------------------------------------------------- views (as x2_grid_sum)
_bc = {}
def border_flags(fold):
    if fold not in _bc:
        f = PanNukeFold(fold); bs = []
        for i in range(len(f)):
            lab, n = relabel(np.asarray(f.inst[i]))
            if n == 0: continue
            edge = set()
            for e in (lab[0,:], lab[-1,:], lab[:,0], lab[:,-1]): edge.update(np.unique(e).tolist())
            b = np.zeros(n+1, bool); b[list(edge)] = True; bs.append(b[1:])
        _bc[fold] = np.concatenate(bs)
    return _bc[fold]

def views(d, tf):
    sm = json.load(open(d + "/summary.json"))
    z = np.load(d + "/per_image.npz")
    g = pd.read_csv(d + "/gt_records.csv.gz")
    tis = g.drop_duplicates("image").set_index("image")["tissue"].reindex(range(z["class_PQ"].shape[0])).values
    dead = z["class_PQ"][:, 3]; has = ~np.isnan(dead)
    gi = g[(g.cls == 4).values & ~border_flags(tf)[:len(g)]]
    miss = gi.status.str.startswith("missed").values
    return dict(mPQ=sm["official"]["mPQ"], bPQ=sm["official"]["bPQ"], mpq=sm["plus"]["mPQ+"],
                dead=sm["per_class_PQ"]["Dead"], dutex=float(np.nanmean(dead[has & (tis != "Uterus")])),
                fc=sm["detection"]["F_c"]["Dead"], dq=sm["per_class_DQ+"]["Dead"],
                imiss=miss.mean(), nmiss=int(miss.sum()), nint=len(gi))

KEYS = ["mPQ", "bPQ", "mpq", "dead", "dutex", "fc", "dq", "imiss"]
CLASSES = {1: "Neo", 2: "Inf", 3: "Conn", 4: "Dead", 5: "Epi"}

def fmt(v): return f"{v:.4f}"

# ---------------------------------------------------------------- [A] architecture table
print("========== [A] strict-protocol architecture endpoints (3-split seed-19) ==========")
ARCH = {
    "KongNet":   {s: f"runs/kongnet/split{s}/eval_test_fold{tf}" for s, tf in SPLITS.items()},
    "LKCell-L":  {s: f"runs/lkcell/split{s}/eval_fold{tf}" for s, tf in SPLITS.items()},
    "base x1":   {s: f"runs/cellvit_uni/split{s}/eval_test_fold{tf}" for s, tf in SPLITS.items()},
    "x2+du2":    {s: f"runs/cellvit_uni_x2/split{s}/eval_test_fold{tf}_du2" for s, tf in SPLITS.items()},
    "x2+du2lev": {s: f"runs/cellvit_uni_x2/split{s}/eval_test_fold{tf}_du2lev" for s, tf in SPLITS.items()},
}
for name, per in ARCH.items():
    rows = {}
    for s, tf in SPLITS.items():
        d = per[s]
        assert os.path.exists(d + "/summary.json"), d
        rows[s] = views(d, tf)
        print(f" {name:10s} split{s} " + " ".join(f"{k}={rows[s][k]:.4f}" for k in KEYS))
    m = {k: float(np.mean([rows[s][k] for s in rows])) for k in KEYS}
    print(f" {name:10s} MEAN   " + " ".join(f"{k}={m[k]:.4f}" for k in KEYS))
print()

# ---------------------------------------------------------------- [B] du2 recovery
print("========== [B] du2 recovery, 3-split seed-19 means (base / x2 / x2+du2) ==========")
for tta, tag in [(False, ""), (True, "_tta")]:
    # split1's x2 evals are untagged (in-flight run); splits 2/3 tagged _x2 (pulled from LM2).
    # du2: noTTA = eval_test_fold{tf}_du2, TTA = eval_test_fold{tf}_du2_tta.
    X2 = {1: "runs/cellvit_uni_x2/split1/eval_test_fold3" + tag,
          2: "runs/cellvit_uni_x2/split2/eval_test_fold3_x2" + tag,
          3: "runs/cellvit_uni_x2/split3/eval_test_fold1_x2" + tag}
    # du2 TTA lives under the watcher's built-in name eval_fold{tf}_x2_du2_tta
    DU2 = {1: "runs/cellvit_uni_x2/split1/eval_fold3_x2_du2_tta",
           2: "runs/cellvit_uni_x2/split2/eval_fold3_x2_du2_tta",
           3: "runs/cellvit_uni_x2/split3/eval_fold1_x2_du2_tta"}
    arms = {
        "base": {s: f"runs/cellvit_uni/split{s}/eval_test_fold{tf}{tag}" for s, tf in SPLITS.items()},
        "x2":   X2,
        "du2":  {s: (f"runs/cellvit_uni_x2/split{s}/eval_test_fold{tf}_du2" if not tta else DU2[s])
                 for s, tf in SPLITS.items()},
    }
    line = f" {'TTA ' if tta else 'noTTA'} "
    for aname, per in arms.items():
        vals = {}
        for k in ["mPQ", "bPQ", "mpq", "dead"]:
            vv = []
            for s, tf in SPLITS.items():
                d = per[s]
                assert os.path.exists(d + "/summary.json"), (aname, d)
                sm = json.load(open(d + "/summary.json"))
                vv.append({"mPQ": sm["official"]["mPQ"], "bPQ": sm["official"]["bPQ"],
                           "mpq": sm["plus"]["mPQ+"], "dead": sm["per_class_PQ"]["Dead"]}[k])
            vals[k] = float(np.mean(vv))
        line += f" {aname}: mPQ={fmt(vals['mPQ'])} bPQ={fmt(vals['bPQ'])} mPQ+={fmt(vals['mpq'])} Dead={fmt(vals['dead'])} |"
    print(line)
print()

# ---------------------------------------------------------------- [C] B1 initial split-1
print("========== [C] B1 initial (split 1, x2 vs 3-seed base), saved bootstrap JSONs ==========")
for tag in ["notta", "tta"]:
    j = json.load(open(f"runs/analysis/x2_split1_{tag}_vs_base.json"))
    print(f" --{tag}--")
    print(json.dumps(j, indent=1)[:2000])
print()

# ---------------------------------------------------------------- [D] lever
print("========== [D] lever du2lev - du2 (paired image bootstrap) ==========")
for s in (1, 2, 3):
    j = json.load(open(f"runs/analysis/lever_s{s}_du2_vs_du2lev.json"))
    print(f" split{s}: " + json.dumps(j)[:600])
print(" per-run replication (all seed dirs, frozen 0.25/40):")
for s, tf in SPLITS.items():
    for d in sorted(glob.glob(f"runs/cellvit_uni_x2/split{s}*/eval_test_fold{tf}_du2lev")):
        a = views(d.replace("_du2lev", "_du2"), tf); b = views(d, tf)
        print(f"  {d.split('/')[-2]:14s} dmPQ={b['mPQ']-a['mPQ']:+.4f} dbPQ={b['bPQ']-a['bPQ']:+.4f} dDead={b['dead']-a['dead']:+.4f}")
print()

# ---------------------------------------------------------------- [E] size-binned mechanism (split-1 TTA)
print("========== [E] size bins, split-1 TTA, x2 (seed19) minus 3-seed base mean ==========")
base_dirs = ["runs/cellvit_uni/split1/eval_test_fold3_tta",
             "runs/cellvit_abl/split1_seed1/eval_test_fold3_tta",
             "runs/cellvit_abl/split1_seed2/eval_test_fold3_tta"]
x2_dir = "runs/cellvit_uni_x2/split1/eval_test_fold3_tta"
BINS = [(0, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400)]

def bin_stats(g, lo, hi, dead_only=False):
    m = (g.area >= lo) & (g.area < hi)
    if dead_only: m &= g.cls == 4
    else: m &= g.cls != 4
    sub = g[m.values]
    if len(sub) == 0: return None
    matched = sub.status == "matched"
    rate = matched.mean()
    iou_all = (sub.max_iou_x1000 / 1000.0).mean()
    iou_m = (sub.max_iou_x1000[matched] / 1000.0).mean() if matched.any() else np.nan
    return rate, iou_all, iou_m, len(sub)

for label, dead_only in [("nonDead", False), ("Dead", True)]:
    for lo, hi in BINS:
        bs = [bin_stats(pd.read_csv(d + "/gt_records.csv.gz"), lo, hi, dead_only) for d in base_dirs]
        xs = bin_stats(pd.read_csv(x2_dir + "/gt_records.csv.gz"), lo, hi, dead_only)
        bs = [b for b in bs if b]
        if not bs or xs is None: continue
        br = float(np.mean([b[0] for b in bs])); ball = float(np.mean([b[1] for b in bs])); bm = float(np.nanmean([b[2] for b in bs]))
        n = xs[3]
        print(f" {label} {lo}-{hi}px: n={n:6d} dMatchRate={xs[0]-br:+.3f} dBestIoU(all)={xs[1]-ball:+.3f} dIoU(matched)={xs[2]-bm:+.3f}")
print()

# ---------------------------------------------------------------- [F] error composition (split-1 TTA)
print("========== [F] error composition, split-1 TTA (x2 vs 3-seed base) ==========")
for name, dirs in [("base", base_dirs), ("x2", [x2_dir])]:
    gts = [pd.read_csv(d + "/gt_records.csv.gz") for d in dirs]
    prs = [pd.read_csv(d + "/pred_records.csv.gz") for d in dirs]
    g = pd.concat(gts); p = pd.concat(prs)
    per_run_npred = [len(pr) / gt.image.nunique() for gt, pr in zip(gts, prs)]
    print(f" --{name}-- npred/image per run: {[f'{v:.1f}' for v in per_run_npred]}")
    small = [(pr.area < 80).mean() for pr in prs]
    print(f"    <80px pred share per run: {[f'{v:.3f}' for v in small]}")
    for c in (1, 2, 3, 4, 5):
        sub = pd.concat([gt[gt.cls == c] for gt in gts])
        if name == "x2":
            sub = pd.concat([gt[gt.cls == c] for gt in gts])  # single run already
        n = len(sub)
        rates = {}
        for st in ("matched", "merged", "split"):
            rates[st] = (sub.status == st).mean()
        rates["missed_bg"] = sub.status.str.startswith("missed_bg").mean()
        rates["missed_sh"] = sub.status.str.startswith("missed_shape").mean()
        print(f"    {CLASSES[c]:5s} n={n:6d} " + " ".join(f"{k}={v:.3f}" for k, v in rates.items()))
