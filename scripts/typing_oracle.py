#!/usr/bin/env python
"""Typing-oracle ceiling: how far official mPQ could rise if every matched instance were typed right.

Read-only diagnostic over *already-evaluated* baseline predictions (audit-type access to the test
folds; it is not a tuning entry point — any lever developed from this analysis must be tuned on
validation folds only). Instance geometry is untouched; only instance classes change, so
`RetypeEvaluator` rescores the official mPQ exactly (same image -> tissue -> class aggregation as
`nucseg.metrics.pannuke_eval.evaluate`).

Reports, per split and as a 3-split mean:
  * current  — the decoded prediction's mPQ (sanity check against the stored official eval);
  * oracle   — every IoU > 0.5 matched instance receives its GT class;
  * inf_conn — only matches whose GT class is Inflammatory or Connective are corrected.

    python scripts/typing_oracle.py [--runs runs/cellvit_uni/split{1,2,3}] [--out-json PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.pannuke_eval import instance_classes

TEST_FOLD = {1: 3, 2: 3, 3: 1}
PRED_CANDIDATES = {1: ("pred_fold3.npz", "pred_test_fold3.npz"),
                   2: ("pred_test_fold3.npz", "pred_fold3.npz"),
                   3: ("pred_test_fold1.npz", "pred_fold1.npz")}


def load_pred(run: Path, split: int) -> dict:
    for name in PRED_CANDIDATES[split]:
        if (run / name).exists():
            return dict(np.load(run / name))
    raise SystemExit(f"no prediction file found under {run} (tried {PRED_CANDIDATES[split]})")


def instance_table(pred: dict):
    """Rebuild the (img, id)-sorted instance table when the npz predates that column pair."""
    if "inst_img" in pred:
        return pred["inst_img"], pred["inst_id"]
    img, iid = [], []
    for j in range(len(pred["inst"])):
        ids = np.unique(pred["inst"][j])
        ids = ids[ids > 0]
        img.append(np.full(len(ids), j))
        iid.append(ids)
    return np.concatenate(img), np.concatenate(iid)


def score_split(run: Path, split: int) -> dict:
    pred = load_pred(run, split)
    fold = TEST_FOLD[split]
    f = PanNukeFold(fold)
    inst_img, inst_id = instance_table(pred)
    ev = RetypeEvaluator(f.gt_channels, pred["inst"], inst_img, inst_id, f.tissue)
    cur = np.concatenate([instance_classes(pred["inst"][j], pred["type"][j])[1] for j in range(len(f))])

    oracle = cur.copy()
    oracle[ev.m_row] = ev.m_cls
    inf_conn = cur.copy()
    sel = np.isin(ev.m_cls, [2, 3])
    inf_conn[ev.m_row[sel]] = ev.m_cls[sel]

    m_cur, per_cur = ev.mpq(cur)
    m_orc, per_orc = ev.mpq(oracle)
    m_ic, _ = ev.mpq(inf_conn)
    return {"split": split, "fold": fold, "current": m_cur, "oracle": m_orc, "inf_conn": m_ic,
            "per_class_current": np.atleast_1d(per_cur).tolist(),
            "per_class_oracle": np.atleast_1d(per_orc).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs=3, default=["runs/cellvit_uni/split1", "runs/cellvit_uni/split2",
                                               "runs/cellvit_uni/split3"], metavar="RUN")
    p.add_argument("--out-json", type=Path, default=None)
    a = p.parse_args()

    rows = [score_split(Path(r), s) for r, s in zip(a.runs, (1, 2, 3))]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("current", "oracle", "inf_conn")}
    for r in rows:
        print(f"split{r['split']} (fold{r['fold']}): mPQ {r['current']:.4f} -> "
              f"typing-oracle {r['oracle']:.4f} (+{r['oracle'] - r['current']:.4f}) | "
              f"Inf/Conn-only {r['inf_conn']:.4f} (+{r['inf_conn'] - r['current']:.4f})")
        per = "  ".join(f"{nm[:4]} {c:.3f}->{o:.3f}" for nm, c, o in
                        zip(CLASS_NAMES, r["per_class_current"], r["per_class_oracle"]))
        print(f"    {per}")
    print(f"\n3-split mean: current {mean['current']:.4f} | typing oracle {mean['oracle']:.4f} "
          f"(+{mean['oracle'] - mean['current']:.4f}) | Inf/Conn-only {mean['inf_conn']:.4f} "
          f"(+{mean['inf_conn'] - mean['current']:.4f})")

    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps({"rows": rows, "mean": mean,
                                          "class_names": list(CLASS_NAMES)}, indent=2))
        print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
