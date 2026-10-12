#!/usr/bin/env python
"""E1-T Task 3 Step 2 verification: spot-check three images' four-cell grouping by hand.

Read-only: recomputes pairing/votes for a few images with the same frozen code path and
prints the per-GT grouping so a human can eyeball it against stats.json.
Usage: python scripts/e1t_verify_spot.py --round R --img 0 100 2500 [--extra-jury ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e1t_typing_consensus import builtin_jury, pair_base_detailed  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.e1_typing import jury_votes, majority_type  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--img", type=int, nargs="+", required=True)
    ap.add_argument("--extra-jury", nargs="*", default=[])
    a = ap.parse_args(argv)
    jury = builtin_jury(a.round)
    for spec in a.extra_jury:
        path, name, arch, tf = spec.split(":")
        jury.append({"name": name, "path": path, "arch": arch, "train_fold": int(tf[5:])})
    base_path = a.round / "p0" / "split1_seed1" / "val" / "base" / "identity" / "pred.npz"
    f = PanNukeFold(2)
    with np.load(base_path) as z:
        bi, bt = z["inst"], z["type"]
    for seat in jury:
        with np.load(seat["path"]) as z:
            seat["inst"], seat["type"] = z["inst"], z["type"]
    folds = {s: seat["train_fold"] for s, seat in enumerate(jury)}
    for j in a.img:
        print(f"=== image {j} (tissue={f.tissue[j]})")
        rows = pair_base_detailed(f.inst[j], f.type[j], bi[j], bt[j])
        votes = {}
        for s, seat in enumerate(jury):
            for gid, v in jury_votes(f.inst[j], seat["inst"][j], seat["type"][j]).items():
                if v is not None:
                    votes.setdefault(gid, {})[s] = v[1]
        for r in rows:
            maj, d = majority_type(votes.get(r["gt_id"], {}), folds)
            g1 = r["status"] == "matched" and r["gt_class"] != r.get("base_class")
            cell = ("G1" if g1 else "G0") if r["status"] == "matched" else r["status"]
            mdesc = {jury[s]["name"].split("_")[0] + f"s{s}": t for s, t in votes.get(r["gt_id"], {}).items()}
            print(f"  GT{r['gt_id']} cls{r['gt_class']} {cell} base_cls={r.get('base_class')} "
                  f"maj={maj} nvotes={d['n_votes']} votes={mdesc}")


if __name__ == "__main__":
    main()
