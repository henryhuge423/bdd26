"""Pre-registered lever verdict (2026-10-01): du2lev vs du2 on the 3 TEST folds.

Same Dead views as x2_dead_views.py (official Dead PQ, Uterus-excluded, interior-Dead
missed_bg / matched share, Dead F_c, DQ+) plus npred and fragment load, per split and
3-split mean. The lever constants were chosen VAL-only (frac 0.25, a_min 40); this is
their one-shot test readout."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from nucseg.metrics.instance import relabel
from nucseg.data.pannuke import PanNukeFold

SPLITS = {  # split -> (test fold, eval suffix)
    1: (3, "eval_test_fold3"),
    2: (3, "eval_test_fold3"),
    3: (1, "eval_test_fold1"),
}
ARMS = {"base": "runs/cellvit_uni/split{s}/{e}", "du2": "runs/cellvit_uni_x2/split{s}/{e}_du2",
        "du2lev": "runs/cellvit_uni_x2/split{s}/{e}_du2lev"}

# border flag per GT nucleus, evaluator order (GT is fixed per fold)
_border_cache = {}
def border_flags(fold):
    if fold not in _border_cache:
        f = PanNukeFold(fold)
        borders = []
        for i in range(len(f)):
            lab, n = relabel(np.asarray(f.inst[i]))
            if n == 0: continue
            edge = set()
            for e in (lab[0,:], lab[-1,:], lab[:,0], lab[:,-1]): edge.update(np.unique(e).tolist())
            b = np.zeros(n+1, bool); b[list(edge)] = True
            borders.append(b[1:])
        _border_cache[fold] = np.concatenate(borders)
    return _border_cache[fold]

rows = {}
print(f"{'split':>5s} {'arm':7s} {'mPQ':>6s} {'bPQ':>6s} {'mPQ+':>6s} {'DeadPQ':>7s} {'-Uterus':>8s} "
      f"{'F_c':>6s} {'DQ+':>6s} {'int_mbg':>8s} {'int_miss':>9s} {'int_mch':>8s} {'npred':>6s}")
for s, (fold, e) in SPLITS.items():
    border = border_flags(fold)
    for arm, pat in ARMS.items():
        d = pat.format(s=s, e=e)
        sm = json.load(open(f"{d}/summary.json"))
        z = np.load(f"{d}/per_image.npz")
        g = pd.read_csv(f"{d}/gt_records.csv.gz")
        tis = g.drop_duplicates("image").set_index("image")["tissue"].reindex(range(z["class_PQ"].shape[0])).values
        dead = z["class_PQ"][:, 3]; has = ~np.isnan(dead)
        ex = has & (tis != "Uterus")
        gi = g[(g.cls == 4).values & ~border[:len(g)]]
        miss_any = float(gi.status.str.startswith("missed").mean())
        row = dict(mPQ=sm["official"]["mPQ"], bPQ=sm["official"]["bPQ"], mpq=sm["plus"]["mPQ+"],
                   dead=sm["per_class_PQ"]["Dead"], dutex=float(np.nanmean(dead[ex])),
                   fc=sm["detection"]["F_c"]["Dead"], dq=sm["per_class_DQ+"]["Dead"],
                   imbg=float((gi.status == "missed_bg").mean()), imiss=miss_any,
                   imch=float((gi.status == "matched").mean()),
                   npred=float(pd.read_csv(f"{d}/pred_records.csv.gz").groupby("image").size().mean()))
        rows[(s, arm)] = row
        print(f"{s:5d} {arm:7s} {row['mPQ']:.4f} {row['bPQ']:.4f} {row['mpq']:.4f} {row['dead']:7.3f} "
              f"{row['dutex']:8.3f} {row['fc']:6.3f} {row['dq']:6.3f} {row['imbg']:8.3f} {row['imiss']:8.3f} "
              f"{row['imch']:8.3f} {row['npred']:6.1f}")

print("\n3-split means + delta (lev - du2):")
for arm in ARMS:
    m = {k: float(np.mean([rows[(s, arm)][k] for s in SPLITS])) for k in next(iter(rows.values()))}
    print(f"{arm:7s} " + " ".join(f"{k}={m[k]:.4f}" for k in ("mPQ","bPQ","mpq","dead","dutex","fc","dq","imbg","imiss","imch","npred")))
for k in ("mPQ","bPQ","mpq","dead","dutex","fc","dq","imbg","imiss","imch"):
    d = np.mean([rows[(s,"du2lev")][k] - rows[(s,"du2")][k] for s in SPLITS])
    print(f"  d{k:5s} {d:+.4f}")
