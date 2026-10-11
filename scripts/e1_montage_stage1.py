#!/usr/bin/env python
"""E1-M stage-1 材料准备：把 200 个盲评窗口拼成 3×3 网格图（每格下方标注其中性 ID）。

    python scripts/e1_montage_stage1.py --packet runs/analysis/e1_packet_20261010 \
        --out runs/analysis/e1_consensus_20261010/stage1_sheets

仅供"非专家模型预审"（spec §7.2）分发用；格内为 stage1 原图（含红十字，无来源/置信度/GT）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CELL, GAP, PAD, LABEL_H = 160, 8, 8, 22


def make_sheet(cells: list[np.ndarray], ids: list[str]):
    """3×3 grid of equal-size cells with an ID label under each; returns (sheet, ids)."""
    n = len(cells)
    assert n <= 9 and len(ids) == n
    size = PAD * 2 + 3 * CELL + 2 * (GAP + LABEL_H)   # label band under EVERY row
    sheet = np.full((size, size, 3), 255, np.uint8)
    for k, (cell, iid) in enumerate(zip(cells, ids)):
        r, c = divmod(k, 3)
        y = PAD + r * (CELL + GAP + LABEL_H)
        x = PAD + c * (CELL + GAP)
        sheet[y:y + CELL, x:x + CELL] = cell
        cv2.putText(sheet, iid, (x + 4, y + CELL + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet, list(ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    ids = sorted(p.stem for p in (a.packet / "stage1").glob("IMG_*.png"))
    assert len(ids) == 200, f"expected 200 stage1 windows, got {len(ids)}"
    a.out.mkdir(parents=True)
    for s in range(0, 200, 9):
        chunk = ids[s:s + 9]
        cells = [cv2.imread(str(a.packet / "stage1" / f"{i}.png")) for i in chunk]
        sheet, _ = make_sheet(cells, chunk)
        cv2.imwrite(str(a.out / f"sheet_{s // 9:02d}.png"), sheet)
    print(f"wrote {len(range(0, 200, 9))} sheets to {a.out}")


if __name__ == "__main__":
    main()
