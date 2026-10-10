#!/usr/bin/env python
"""修复特征缓存的标签列（_gt_classes 曾整体 +1 偏移，commit 6a179c6 修复前的导出携带错标签）。

    python scripts/repair_tp_labels.py --feat runs/analysis/typing_probe_20261010/feat_fold1.npz
    python scripts/repair_tp_labels.py --feat runs/analysis/typing_probe_20261010/feat_fold2.npz

确定性重导：cls（fold1）/ gt_cls（fold2）按修复后的 _gt_classes 从 GT 掩码+类型图重算；
其余列（feat/ring/prob/area/border/表列）**按构造**逐位不变——只替换这一个键，dict 其余
原样写回。重写 npz 并在 sidecar 记录 repair 条目。fold 取自 sidecar 而非文件名。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_tp_features import _gt_classes  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feat", type=Path, required=True)
    a = ap.parse_args(argv)
    side = a.feat.with_suffix(".npz.json")
    fold = json.loads(side.read_text())["fold"]
    f = PanNukeFold(fold)
    z = dict(np.load(a.feat))

    def derive(inst_img, inst_id):
        out = np.zeros(len(inst_id), np.int64)
        for j in range(len(f)):
            rows = inst_id[inst_img == j]
            if len(rows):
                out[inst_img == j] = _gt_classes(np.asarray(f.inst[j], np.int32),
                                                 np.asarray(f.type[j]), rows)
        return out

    col = "cls" if fold == 1 else "gt_cls"
    key_img, key_id = ("inst_img", "inst_id") if fold == 1 else ("gt_inst_img", "gt_inst_id")
    old = z.pop(col)
    new = derive(z[key_img], z[key_id])
    if fold == 1:
        assert np.all((new >= 1) & (new <= 5)), "labels outside 1..5 after repair"
    z[col] = new
    fixed = int((old != new).sum())

    tmp = a.feat.with_name(a.feat.stem + ".repair_tmp.npz")   # np.savez appends .npz otherwise
    np.savez_compressed(tmp, **z)
    tmp.replace(a.feat)

    meta = json.loads(side.read_text())
    meta["repair"] = {"date": "2026-10-10", "reason": "_gt_classes +1 shift (fixed 6a179c6)",
                      "column": col, "rows_changed": fixed}
    side.write_text(json.dumps(meta, indent=2))
    print(f"repaired {a.feat}: column {col}, {fixed}/{len(new)} rows changed; "
          f"other columns untouched by construction")


if __name__ == "__main__":
    main()
