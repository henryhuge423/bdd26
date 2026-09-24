#!/usr/bin/env python
"""Stream-convert PanNuke fold zips into the compact format (see nucseg.data.prepare)."""
import argparse
import json
from pathlib import Path

from nucseg.data.prepare import convert_fold

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
p.add_argument("--out", type=Path, default=ROOT / "data/pannuke")
p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
a = p.parse_args()
for k in a.folds:
    meta = convert_fold(a.raw / f"fold_{k}.zip", a.out, k)
    print(json.dumps(meta, indent=2))
