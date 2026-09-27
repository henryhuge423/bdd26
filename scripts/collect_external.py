#!/usr/bin/env python
"""Collect cross-domain eval reports into one table (pillar C).

    python scripts/collect_external.py --runs runs/cellvit_uni/split1 runs/cellvit_uni/split2 ... \
        --label cellvit [--json out.json]

Scans each run dir for eval_ext_{data}[_tta]/summary.json and prints mean +- std over runs
(splits) for every dataset: mPQ / bPQ / strict / AJI / F_d and per-class PQ. Absent classes
(NaN) are skipped per split, matching the official protocol.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", type=Path, required=True)
p.add_argument("--label", default="model")
p.add_argument("--json", default=None, help="optional dump of mPQ/bPQ per split")
a = p.parse_args()

by_data = defaultdict(list)  # data name -> list of summary dicts
for run in a.runs:
    for d in sorted(run.glob("eval_ext_*")):
        if not (d / "summary.json").exists():
            continue
        by_data[d.name.removeprefix("eval_ext_")].append(json.loads((d / "summary.json").read_text()))

for data, sums in sorted(by_data.items()):
    print(f"\n=== {a.label} on {data} ({len(sums)} splits) ===")

    def agg(path):
        vals = [x.get(path[0], {}).get(path[1]) if len(path) == 2 else x.get(path[0]) for x in sums]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)

    rows = [("mPQ", ("official", "mPQ")), ("bPQ", ("official", "bPQ")), ("mPQ strict", ("strict", "mPQ")),
            ("AJI", ("AJI",)), ("AJI+", ("AJI+",)), ("F_d", ("detection", "F_d"))]
    for name, path in rows:
        m, s, n = agg(path)
        print(f"  {name:10s} {m:.4f} +- {s:.4f}  (n={n})")
    classes = list(sums[0]["per_class_PQ"].keys())
    line = "  PQ/class   " + "  ".join(f"{c[:5]} {agg(('per_class_PQ', c))[0]:.3f}" for c in classes)
    print(line)
    line = "  strict     " + "  ".join(f"{c[:5]} {agg(('per_class_PQ_strict', c))[0]:.3f}" for c in classes)
    print(line)
    print("  absent-in-GT classes:", sums[0].get("_absent", "see meta.json of the dataset"))
    if a.json:
        Path(a.json).write_text(json.dumps({d: [x["official"] for x in s] for d, s in by_data.items()}))

if not by_data:
    print("no eval_ext_* reports found under", *a.runs)
