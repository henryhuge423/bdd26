"""Historical B1 run-grid summation (pre-audit du2/du2lev outputs vs x1 base).

The intended x2 seeds were {19,1,2}, but split2 actually ran {19,1,1}; base coverage
is 3/1/1. Values below are descriptive means/stds over RUNS, not a symmetric
three-distinct-seed experiment. Actual seeds come from configs, never directory
names; missing configs remain unknown. Corrected M/S results live under the
versioned P0–P3 analysis directory and are summarized by p0p3_numbers.py.
"""
import glob, json, os, sys
from pathlib import Path
from p0p3_numbers import seed_from_run
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from nucseg.metrics.instance import relabel
from nucseg.data.pannuke import PanNukeFold

SPLITS = {1: 3, 2: 3, 3: 1}  # split -> test fold
def dirs(pat, s, tf):
    out = []
    for d in pat.format(s=s, tf=tf).split():
        if os.path.exists(d + "/summary.json"):
            out.append(d)
    return out

ARMS = {
    # globs: split{s}* also picks up split{s}_seed{1,2} as those evals land
    "base":   lambda s, tf: " ".join(filter(None, [" ".join(sorted(glob.glob(f"runs/cellvit_uni/split{s}/eval_test_fold{tf}"))),
                                                  " ".join(sorted(glob.glob(f"runs/cellvit_abl/split{s}_seed*/eval_test_fold{tf}")))])),
    "du2":    lambda s, tf: " ".join(sorted(glob.glob(f"runs/cellvit_uni_x2/split{s}*/eval_test_fold{tf}_du2"))),
    "du2lev": lambda s, tf: " ".join(sorted(glob.glob(f"runs/cellvit_uni_x2/split{s}*/eval_test_fold{tf}_du2lev"))),
}

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
arm_means = {}
print("AUDIT: historical M/S outputs; run SD is not independent-seed uncertainty. "
      "Split2 has a repeated seed1; base seed coverage is asymmetric.")
for arm, pat in ARMS.items():
    print(f"===== {arm} =====")
    means = {}
    for s, tf in SPLITS.items():
        ds = dirs(pat(s, tf), s, tf)  # drops half-synced dirs without summary.json
        rows = [views(d, tf) | {"dir": d.split("/")[-2] + "/" + d.split("/")[-1],
                               "actual_seed": seed_from_run(Path(d).parent)} for d in ds]
        if not rows:
            print(f" split{s}: (none yet)"); continue
        for r in rows:
            print(f"  split{s} {r['dir']:32s} actual_seed={r['actual_seed']} " + " ".join(f"{k}={r[k]:.4f}" for k in KEYS) + f" ({r['nmiss']}/{r['nint']})")
        m = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
        sd = {k: (float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else float("nan")) for k in KEYS}
        m["imiss_pooled"] = sum(r["nmiss"] for r in rows) / sum(r["nint"] for r in rows)
        print(f"  split{s} RUN MEAN/SD (n_runs={len(rows)}, actual_seeds={[r['actual_seed'] for r in rows]}) " + " ".join(f"{k}={m[k]:.4f}±{sd[k]:.4f}" for k in KEYS))
        means[s] = m
    have = sorted(means)
    if len(have) == 3:
        gm = {k: float(np.mean([means[s][k] for s in have])) for k in KEYS}
        gp = float(np.mean([means[s]["imiss_pooled"] for s in have]))
        arm_means[arm] = gm
        print(f"  3-SPLIT MEAN " + " ".join(f"{k}={gm[k]:.4f}" for k in KEYS) + f" imiss_pooled~{gp:.4f}")
    print()

if "base" in arm_means and "du2" in arm_means:
    print("===== deltas vs base (3-split means) =====")
    for arm in ("du2", "du2lev"):
        if arm in arm_means:
            print(f" {arm:7s} " + " ".join(f"d{k}={arm_means[arm][k]-arm_means['base'][k]:+.4f}" for k in KEYS))
    if "du2lev" in arm_means:
        print(f" du2lev-du2 " + " ".join(f"d{k}={arm_means['du2lev'][k]-arm_means['du2'][k]:+.4f}" for k in KEYS))
