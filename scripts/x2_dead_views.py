"""B1 x2 split1: pre-registered Dead views + bPQ regression localisation."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from nucseg.metrics.instance import relabel
from nucseg.data.pannuke import PanNukeFold

RUNS = {
    "base_s19_tta": "runs/cellvit_uni/split1/eval_test_fold3_tta",
    "base_s1_tta":  "runs/cellvit_abl/split1_seed1/eval_test_fold3_tta",
    "base_s2_tta":  "runs/cellvit_abl/split1_seed2/eval_test_fold3_tta",
    "x2_s19_tta":   "runs/cellvit_uni_x2/split1/eval_test_fold3_tta",
    "base_s19":     "runs/cellvit_uni/split1/eval_test_fold3",
    "base_s1":      "runs/cellvit_abl/split1_seed1/eval_test_fold3",
    "base_s2":      "runs/cellvit_abl/split1_seed2/eval_test_fold3",
    "x2_s19":       "runs/cellvit_uni_x2/split1/eval_test_fold3",
}

# border flag per GT nucleus, evaluator order (same for every run: fold-3 GT is fixed)
f = PanNukeFold(3)
borders = []
for i in range(len(f)):
    lab, n = relabel(np.asarray(f.inst[i]))
    if n == 0: continue
    edge = set()
    for e in (lab[0,:], lab[-1,:], lab[:,0], lab[:,-1]): edge.update(np.unique(e).tolist())
    b = np.zeros(n+1, bool); b[list(edge)] = True
    borders.append(b[1:])
border = np.concatenate(borders)

print(f"{'run':14s} {'DeadPQ':>7s} {'-Uterus':>8s} {'ut_DeadPQ':>9s} {'DQ+':>6s} {'F_c':>6s} {'int_missbg':>10s} {'int_match':>9s} {'n_int':>6s}")
for name, d in RUNS.items():
    s = json.load(open(f"{d}/summary.json"))
    z = np.load(f"{d}/per_image.npz")
    g = pd.read_csv(f"{d}/gt_records.csv.gz")
    tis = g.drop_duplicates("image").set_index("image")["tissue"].reindex(range(z["class_PQ"].shape[0])).values
    dead = z["class_PQ"][:, 3]; has = ~np.isnan(dead)
    ut = has & (tis == "Uterus"); ex = has & (tis != "Uterus")
    deadmask = (g.cls == 4).values
    gi = g[deadmask & ~border[:len(g)]]  # interior Dead GT
    print(f"{name:14s} {np.nanmean(dead[has]):7.3f} {np.nanmean(dead[ex]):8.3f} "
          f"{np.nanmean(dead[ut]):9.3f} {s['per_class_DQ+']['Dead']:6.3f} {s['detection']['F_c']['Dead']:6.3f} "
          f"{(gi.status=='missed_bg').mean():10.3f} {(gi.status=='matched').mean():9.3f} {len(gi):6d}")

print("\n== per-tissue (TTA): x2 - base_mean, bPQ / mPQ ==")
bt = [json.load(open(RUNS[k] + "/summary.json"))["per_tissue"] for k in ("base_s19_tta","base_s1_tta","base_s2_tta")]
xt = json.load(open(RUNS["x2_s19_tta"] + "/summary.json"))["per_tissue"]
rows = []
for t in xt:
    db = xt[t]["bPQ"] - np.mean([b[t]["bPQ"] for b in bt])
    dm = xt[t]["mPQ"] - np.mean([b[t]["mPQ"] for b in bt])
    rows.append((t, dm, db, xt[t]["bPQ"]))
for t, dm, db, xb in sorted(rows, key=lambda r: r[2]):
    print(f"{t:16s} dmPQ {dm:+.3f}  dbPQ {db:+.3f}  (x2 bPQ {xb:.3f})")

print("\n== Dead PQ by tissue (TTA): x2 vs base mean ==")
btc = [json.load(open(RUNS[k] + "/summary.json"))["tissue_class_PQ"] for k in ("base_s19_tta","base_s1_tta","base_s2_tta")]
xtc = json.load(open(RUNS["x2_s19_tta"] + "/summary.json"))["tissue_class_PQ"]
for t in xtc:
    bd = xtc[t].get("Dead", float("nan"))
    base_vals = [b[t].get("Dead", float("nan")) for b in btc if t in b]
    m = np.nanmean(base_vals)
    if bd == bd and m == m:
        print(f"{t:16s} x2 {bd:6.3f}  base {m:6.3f}  d {bd-m:+.3f}")
