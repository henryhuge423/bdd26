#!/usr/bin/env python
"""DSB dev-fold evaluation: apply the frozen area-floor menu to dead-expert candidates (spec §4 gate v0).

    python scripts/dsb_dev_eval.py --run runs/dsb_dev_20261007/dsb_seed19 --fold 2
-> <run>/dsb_dev_eval/menu.json: every menu row plus the frozen selection.

Menu (frozen before any run): identity = no additions; a30 / a60 = area floors 30 / 60 px on
whole, base-disjoint candidates. Selection rule: argmax dDead_PQ subject to dBPQ >= -0.001,
ties to the smaller floor; identity when nothing beats it. Dev folds only — never a test fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate
from nucseg.postproc.dead_merge import merge_dead

MENU = (("identity", None), ("a30", 30), ("a60", 60))
GUARD_BPQ = -0.001
_FLOOR = {"identity": 0, "a30": 30, "a60": 60}


def select_menu(rows):
    """Frozen selection: argmax d_dead subject to d_bpq >= guard, tie -> smaller floor, else identity."""
    guard = [r for r in rows if float(r["d_bpq"]) >= GUARD_BPQ]
    best = max(guard, key=lambda r: (float(r["d_dead"]), -_FLOOR[r["name"]]))
    return best["name"] if float(best["d_dead"]) > 0 else "identity"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--tag", default=None, help="prediction tag (default: fold{k})")
    a = p.parse_args(argv)
    tag = a.tag or f"fold{a.fold}"
    base = np.load(a.run / f"pred_{tag}.npz")
    dead = np.load(a.run / f"pred_{tag}_dead.npz")["dead_inst"]
    f = PanNukeFold(a.fold)
    rows, ref = [], None
    for name, floor in MENU:
        merged_i = np.empty_like(base["inst"])
        merged_t = np.empty_like(base["type"])
        n_added = 0
        for i in range(len(f)):
            if floor is None:
                merged_i[i], merged_t[i] = base["inst"][i], base["type"][i]
            else:
                mi, mt, chosen = merge_dead(base["inst"][i], base["type"][i], dead[i], floor)
                merged_i[i], merged_t[i] = mi, mt
                n_added += len(chosen)
        s = evaluate(f.gt_channels, f.inst, f.type, f.tissue, merged_i, merged_t)["summary"]
        row = {"name": name, "min_area": floor, "n_added": n_added,
               "dead_pq": float(s["per_class_PQ"]["Dead"]),
               "bpq": float(s["official"]["bPQ"]), "mpq": float(s["official"]["mPQ"])}
        if ref is None:
            ref = row
        row["d_dead"] = row["dead_pq"] - ref["dead_pq"]
        row["d_bpq"] = row["bpq"] - ref["bpq"]
        row["d_mpq"] = row["mpq"] - ref["mpq"]
        rows.append(row)
    selection = select_menu(rows)
    out = a.run / "dsb_dev_eval"
    out.mkdir(exist_ok=True)
    (out / "menu.json").write_text(json.dumps({"rows": rows, "selection": selection}, indent=2))
    print(json.dumps({"selection": selection, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
