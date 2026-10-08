#!/usr/bin/env python
"""Gate D (DSB spec §6): Dead oracle ceiling on a validation fold.

    python scripts/dsb_oracle_ceiling.py --run runs/cellvit_uni/split1 --fold 2 \
        --out runs/analysis/dsb_gates_20261007/gate_d.json

Every GT Dead instance that (a) no prediction matches at IoU>0.5 and (b) does not touch the
image border, is inserted as an oracle prediction: once with its exact mask (IoU 1.0 ceiling)
and once shrunk by seeded pixel dropout (drop-frac f gives IoU exactly 1-f per instance).
Continue condition (frozen): ΔDead PQ (internal additions, shrunk variant) ≥ +0.010.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from fold_guard import ensure_dev_fold, run_split
from nucseg.constants import DEAD_TYPE
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import evaluate

THRESHOLD = 0.010


def gate_d_verdict(d_dead: float) -> str:
    return "pass" if float(d_dead) >= THRESHOLD else "fail"


def _border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def unmatched_internal_dead(gt_inst, gt_typ, pred) -> list[int]:
    """GT Dead ids missed by every prediction (IoU>0.5 anywhere = matched) and not touching the border."""
    gt_inst, gt_typ, pred = (np.asarray(m) for m in (gt_inst, gt_typ, pred))
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    out = []
    if len(gt_ids) == 0:
        return out
    ov = overlap(gt_inst, pred)  # rows follow sorted nonzero GT ids (relabel via np.unique)
    iou = ov.iou
    for k, gid in enumerate(gt_ids):
        mask = gt_inst == gid
        cls = int(np.bincount(gt_typ[mask]).argmax())
        if cls != DEAD_TYPE or _border(mask):
            continue
        if iou.shape[1] and (iou[k] > 0.5).any():
            continue
        out.append(int(gid))
    return out


def shrink_mask(mask, drop_frac: float, rng) -> np.ndarray:
    """Drop round(area*drop_frac) uniformly random pixels: IoU vs the original is exactly 1-f."""
    mask = np.asarray(mask, bool)
    area = int(mask.sum())
    n_drop = int(round(area * drop_frac))
    if n_drop == 0 or n_drop >= area:
        raise ValueError("drop_frac leaves an empty or unchanged mask for this area")
    ys, xs = np.nonzero(mask)
    drop = rng.choice(area, size=n_drop, replace=False)
    out = mask.copy()
    out[ys[drop], xs[drop]] = False
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True,
                   help="run dir with pred_fold{k}.npz (E2a x1_hv30_seed19 preferred, "
                        "any same-recipe split1 base run accepted as fallback)")
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--drop-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    ensure_dev_fold(a.fold, run_split(a.run))
    pred_path = a.run / f"pred_fold{a.fold}.npz"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} missing; refusing")
    base = np.load(pred_path)
    f = PanNukeFold(a.fold)

    def insert(variant):
        inst = base["inst"].astype(np.int32).copy()
        typ = base["type"].astype(np.uint8).copy()
        rng = np.random.default_rng(a.seed)
        added, ious = 0, []
        for i in range(len(f)):
            for gid in unmatched_internal_dead(f.inst[i], f.type[i], base["inst"][i]):
                mask = f.inst[i] == gid
                if variant == "shrunk":
                    mask = shrink_mask(mask, a.drop_frac, rng)
                nxt = int(inst[i].max()) + 1
                inst[i][mask] = nxt
                typ[i][mask] = DEAD_TYPE
                added += 1
        return inst, typ, added

    rows = {}
    for variant in ("base", "exact", "shrunk"):
        if variant == "base":
            inst, typ, added = base["inst"], base["type"], 0
        else:
            inst, typ, added = insert(variant)
        s = evaluate(f.gt_channels, f.inst, f.type, f.tissue, inst, typ)["summary"]
        rows[variant] = {"dead_pq": float(s["per_class_PQ"]["Dead"]),
                         "bpq": float(s["official"]["bPQ"]), "mpq": float(s["official"]["mPQ"]),
                         "strict_dead_pq": float(s["per_class_PQ_strict"]["Dead"]),
                         "n_added": added}
    d_dead = rows["shrunk"]["dead_pq"] - rows["base"]["dead_pq"]
    result = {"run": str(a.run), "fold": a.fold, "drop_frac": a.drop_frac, "seed": a.seed,
              "threshold_d_dead": THRESHOLD, "rows": rows,
              "d_dead_exact": rows["exact"]["dead_pq"] - rows["base"]["dead_pq"],
              "d_dead_shrunk": d_dead,
              "d_mpq_shrunk": rows["shrunk"]["mpq"] - rows["base"]["mpq"],
              "verdict": gate_d_verdict(d_dead)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
