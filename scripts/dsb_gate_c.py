#!/usr/bin/env python
"""Gate C (DSB spec §6): decision-level class decoupling upper bound on dumped fold-2 maps.

    python scripts/dsb_gate_c.py --maps data/cache/maps/x1_hv30_seed19 --fold 2 \
        --out runs/analysis/dsb_gates_20261007/gate_c.json

Frozen menu: tau_dead in {0.5 (identity), 0.4, 0.35, 0.3}; every other class keeps the official
0.5. The threshold map is tau[predicted TP argmax], so lowering tau_dead can only add foreground
at pixels the type branch already calls Dead. Verdict: close_line iff some tau_dead reaches
ΔDead >= +0.005 with a bPQ tax <= 0.001 — decision decoupling alone would then make the
architecture unnecessary (spec §6 C).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from fold_guard import ensure_dev_fold
from nucseg.constants import DEAD_TYPE
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.pannuke_eval import evaluate
from nucseg.postproc.recovery import Recovery, postprocess

MENU = (0.5, 0.4, 0.35, 0.3)
GAIN = 0.005
TAX = 0.001


def gate_c_verdict(rows) -> str:
    for r in rows:
        if float(r["tau_dead"]) == 0.5:
            continue
        if float(r["d_dead"]) >= GAIN and float(r["d_bpq"]) >= -TAX:
            return "close_line"
    return "proceed"


def _decode(args):
    maps, tau_dead = args
    fg, hv, tp = maps[..., 0], maps[..., 1:3], maps[..., 3:9]
    thr = np.where(tp.argmax(-1) == DEAD_TYPE, tau_dead, 0.5).astype(np.float32)
    inst, typ = postprocess(fg, hv, tp, Recovery(), thr_map=thr)
    return inst.astype(np.int32), typ


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps", type=Path, required=True, help="dir with maps_fold{k}.npy (dump_cellvit_maps)")
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--split", type=int, default=1,
                   help="split of the run that dumped --maps (test-fold guard; x1_hv30_seed19 = 1)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    ensure_dev_fold(a.fold, a.split)
    maps = np.load(a.maps / f"maps_fold{a.fold}.npy", mmap_mode="r")
    f = PanNukeFold(a.fold)
    if len(maps) != len(f):
        raise SystemExit(f"maps/GT length mismatch: {len(maps)} vs {len(f)}")
    rows, ref = [], None
    for tau in MENU:
        with Pool(a.workers) as pool:
            decoded = pool.map(_decode, ((np.asarray(maps[i]), tau) for i in range(len(f))), chunksize=8)
        inst = np.stack([d[0] for d in decoded])
        typ = np.stack([d[1] for d in decoded])
        s = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)["summary"]
        row = {"tau_dead": tau,
               "dead_pq": float(s["per_class_PQ"]["Dead"]),
               "bpq": float(s["official"]["bPQ"]), "mpq": float(s["official"]["mPQ"])}
        if ref is None:
            ref = row
        row["d_dead"] = row["dead_pq"] - ref["dead_pq"]
        row["d_bpq"] = row["bpq"] - ref["bpq"]
        row["d_mpq"] = row["mpq"] - ref["mpq"]
        rows.append(row)
        print(json.dumps(row))
    result = {"maps": str(a.maps), "fold": a.fold, "menu": list(MENU), "gain_threshold": GAIN,
              "tax_threshold": TAX, "rows": rows, "verdict": gate_c_verdict(rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
