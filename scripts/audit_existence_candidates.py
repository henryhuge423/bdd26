#!/usr/bin/env python
"""Descriptive VAL-fold audit: which signals separate TP from FP fusion additions?

    python scripts/audit_existence_candidates.py --round runs/analysis/matched_seed_20261004 \
        --out runs/analysis/existence_audit_20261005 --workers 4

For every matched-seed pair this reads ONLY the validation fold: the P0-selected
base+M/S prediction, the x2 prediction with probability table, and the pair's
frozen P2 selection (selection.json). Each fuse-eligible candidate (disjoint,
typed) gets geometry features (nucseg.postproc.existence), a distance to the
nearest base instance, type-probability features and its GT status (IoU>.5
match, type agreement, whether base already detects that GT). No thresholds are
selected here and no test fold is read; output feeds the existence-filter menu
design for the next pre-registered round.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import instance_classes
from nucseg.postproc.existence import candidate_geometry, distance_to_base, distance_to_base_map
from nucseg.postproc.scale_fusion import candidate_info, probability_rows

SEEDS = (19, 1, 2)
FEATURES = ("area", "extent", "circularity", "dist", "pmax", "margin")
_G = {}


def _one(j):
    bi, bt = _G["bi"][j], _G["bt"][j]
    xi, xt = _G["xi"][j], _G["xt"][j]
    gt, gtt = _G["gi"][j], _G["gt"][j]
    rows = []
    if not np.any(xi):
        return rows
    dist_map = distance_to_base_map(bi) if np.any(bi) else None
    _, gc = instance_classes(gt, gtt)
    ov_gx = overlap(gt, xi)
    ov_gb = overlap(gt, bi)
    base_hit = (ov_gb.iou > .5).any(axis=1)
    ids = np.unique(xi)
    ids = ids[ids > 0]
    positions = {int(i): k for k, i in enumerate(ids)}
    for c in candidate_info(bi, xi, xt):
        if not c["disjoint"] or c["class"] == 0:
            continue
        mask = xi == c["id"]
        geo = candidate_geometry(mask)
        prow = _G["prob"].get(j, {}).get(c["id"])
        p = np.sort(prow) if prow is not None else np.full(6, np.nan)
        ious = ov_gx.iou[:, positions[c["id"]]]
        best = int(np.argmax(ious)) if len(ious) else -1
        matched = bool(best >= 0 and ious[best] > .5)
        rows.append({"pair": _G["pair"], "image": j, "id": c["id"], "class": c["class"],
                     **geo, "dist": distance_to_base(mask, dist_map) if dist_map is not None else float("inf"),
                     "pmax": float(p[-1]), "margin": float(p[-1] - p[-2]),
                     "iou_best": float(ious[best]) if best >= 0 else 0.,
                     "matched": matched,
                     "typed": bool(matched and gc[best] == c["class"]),
                     "gt_class": int(gc[best]) if best >= 0 else 0,
                     "gt_already_by_base": bool(matched and base_hit[best])})
    return rows


def audit_pair(round_root: Path, pair: str, split: int, seed: int, out: Path, workers: int):
    selection = json.loads((round_root / "p2" / pair / "selection.json").read_text())
    cfg = selection["selected"]
    base_val = round_root / "p0" / pair / "val" / "base" / "ms" / "pred.npz"
    x2_val = Path(selection["provenance"]["inputs"]["x2"]["path"])
    fold = split_folds(split)[1]
    f = PanNukeFold(fold)
    with np.load(base_val) as b, np.load(x2_val) as x:
        bi, bt, xi, xt = b["inst"], b["type"], x["inst"], x["type"]
        prob = probability_rows(x)

    # exact re-implementation of the frozen P2 filter incl. the probability gate
    def passes(c, j):
        if cfg["name"] == "identity":
            return False
        if c["area"] > cfg["max_area"] or (cfg["interior_only"] and c["border"]):
            return False
        if cfg["min_prob"] > 0:
            p = prob.get(j, {}).get(c["id"])
            return p is not None and float(p[c["class"]]) >= cfg["min_prob"]
        return True

    _G.update(bi=bi, bt=bt, xi=xi, xt=xt, gi=f.inst, gt=f.type, prob=prob, pair=pair)
    with Pool(workers) as pool:
        all_rows = []
        for rows in pool.imap(_one, range(len(f)), chunksize=4):
            all_rows.extend(rows)
    _G.clear()
    for r in all_rows:
        r["added"] = passes({"id": r["id"], "area": r["area"], "border": r["border"],
                             "class": r["class"]}, r["image"])
    return all_rows


def bins_summary(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        n = len(sel)
        out.append({"bin": f"[{lo},{hi})", "n": n,
                    "matched": sum(r["matched"] for r in sel),
                    "typed": sum(r["typed"] for r in sel),
                    "dead_typed": sum(r["typed"] and r["class"] == 4 for r in sel),
                    "gt_already": sum(r["gt_already_by_base"] for r in sel)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", type=Path, default=ROOT / "runs/analysis/matched_seed_20261004")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="subset of pair names (default: all 9)")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    a.out.mkdir(parents=True)
    records = []
    todo = a.pairs or [f"split{s}_seed{e}" for s in (1, 2, 3) for e in SEEDS]
    for pair in todo:
        split, seed = int(pair[5]), int(pair[11:])
        rows = audit_pair(a.round, pair, split, seed, a.out, a.workers)
        records.extend(rows)
        added = [r for r in rows if r["added"]]
        print(f"{pair}: eligible {len(rows)}, frozen-P2 added {len(added)}, "
              f"added matched {sum(r['matched'] for r in added)}, "
              f"typed {sum(r['typed'] for r in added)}, "
              f"Dead typed {sum(r['typed'] and r['class'] == 4 for r in added)}", flush=True)
    keys = np.array([r["pair"] for r in records])
    cols = {}
    for k in FEATURES + ("class", "image", "id", "matched", "typed", "gt_class",
                         "gt_already_by_base", "added", "iou_best"):
        cols[k] = np.array([r[k] for r in records])
    np.savez_compressed(a.out / "records.npz", pair=keys, **cols)
    added = [r for r in records if r["added"]]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "round": str(a.round.resolve()),
               "n_eligible": len(records), "n_added": len(added),
               "tables": {
                   "added_by_area": bins_summary(added, "area", (0, 30, 60, 100, 150, 200, 10**9)),
                   "added_by_dist": bins_summary(added, "dist", (0, 1, 3, 6, 11, 10**9)),
                   "added_by_circularity": bins_summary(added, "circularity", (0, .5, .7, .85, 1.01)),
                   "added_by_extent": bins_summary(added, "extent", (0, .3, .5, .7, 1.01)),
                   "added_by_pmax": bins_summary(added, "pmax", (.5, .7, .8, .9, 1.01)),
                   "added_by_margin": bins_summary(added, "margin", (0, .3, .6, .9, 1.01)),
                   "added_by_border": [{"bin": str(b), **{k: v for k, v in
                        bins_summary([r for r in added if r["border"] == b], "area", (0, 1, 10**9))[0].items()
                        if k != "bin"}} for b in (True, False)],
                   "added_by_class": bins_summary(added, "class", (0, 1, 2, 3, 4, 5, 6)),
               }}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, table in summary["tables"].items():
        if name in ("added_by_border",):
            continue
        print(f"\n== {name} (frozen-P2 added only; matched/typed/dead_typed/gt_already) ==")
        for row in table:
            print(f"  {row['bin']:>14} n={row['n']:5d} matched={row['matched']:5d} "
                  f"typed={row['typed']:5d} dead_typed={row['dead_typed']:4d} "
                  f"gt_already={row['gt_already']:5d}")


if __name__ == "__main__":
    main()
