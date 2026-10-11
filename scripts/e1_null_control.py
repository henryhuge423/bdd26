#!/usr/bin/env python
"""E1-C 合成空对照（后补对照，2026-10-11）：GT 空白区内随机同尺寸 blob 的 jury 支持度零假设。

    python scripts/e1_null_control.py \
        --round runs/analysis/matched_seed_20261004 \
        --records runs/analysis/e1_input_check_20261010_split1_seed1/records.npz \
        --consensus runs/analysis/e1_consensus_20261010 \
        --out runs/analysis/e1_consensus_20261010 --n 2000 --seed 20261011

回答：若候选只是"随机落在 GT 未覆盖区域的同尺寸块"，jury 会给出几席支持？观测支持度
超出该零假设的部分才是"检测器真正看到了什么"的证据。blob 面积从**与 GT 零重叠候选**
的经验分布重采样；中心仅在 GT 实例膨胀 8px 之外采样（避免与 GT 边缘粘连）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.instance import overlap  # noqa: E402


def sample_blob(gt_inst: np.ndarray, area: int, rng, dilate: int = 8) -> np.ndarray:
    """Exact-`area` compact blob placed at random outside dilated GT instances.

    Construction: pick a random GT-free center, then take the `area` nearest GT-free pixels
    (within radius sqrt(area/pi)+2) — exact area, compact shape, never touches GT."""
    keep = cv2.dilate((gt_inst > 0).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))) == 0
    ys, xs = np.nonzero(keep)
    if not len(ys):
        return np.zeros_like(gt_inst, bool)
    r = np.sqrt(area / np.pi) + 2
    for _ in range(200):
        k = int(rng.integers(len(ys)))
        cy, cx = ys[k], xs[k]
        yy, xx = np.ogrid[: gt_inst.shape[0], : gt_inst.shape[1]]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        cand = keep & (d2 <= r * r)
        if cand.sum() >= area:
            flat = np.argsort(np.where(cand, d2, np.inf).ravel())[:area]
            out = np.zeros(gt_inst.size, bool)
            out[flat] = True
            return out.reshape(gt_inst.shape)
    return np.zeros_like(gt_inst, bool)      # 放不下（图内 GT 太满）：空 blob，计入失败率


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--consensus", type=Path, required=True,
                    help="dir with per_candidate.npz (jury list) from e1_consensus.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args(argv)
    if (a.out / "null_control.json").exists():
        raise SystemExit("null_control.json exists; refusing to overwrite")
    ensure_dev_fold(2, 1)

    z = np.load(a.records)
    sel = z["pair"] == "split1_seed1"
    matched = np.asarray(z["matched"])[sel]
    iou_best = np.asarray(z["iou_best"])[sel]
    area_pool = np.asarray(z["area"])[sel][~matched & (iou_best == 0)]   # 与 GT 零重叠候选的面积分布

    p = np.load(a.consensus / "per_candidate.npz")
    jury = json.loads((a.consensus / "consensus.json").read_text())["jury"]

    rng = np.random.default_rng(a.seed)
    f2 = PanNukeFold(2)
    gt_all = f2.inst
    picks = rng.integers(0, len(f2), a.n)
    areas = rng.choice(area_pool, a.n)
    blobs, imgs = [], []
    for i in range(a.n):
        blobs.append(sample_blob(np.asarray(gt_all[picks[i]]), int(areas[i]), rng))
        imgs.append(int(picks[i]))
    ok = np.array([b.any() for b in blobs])
    support = np.zeros((a.n, len(jury)), np.int8)
    for j, member in enumerate(jury):
        inst = np.load(member["path"])["inst"]
        for i in range(a.n):
            if not ok[i]:
                continue
            ov = overlap(blobs[i].astype(np.int32), inst[imgs[i]].astype(np.int32))
            if ov.np_ and float(ov.iou[0].max()) >= 0.3:
                support[i, j] = 1
        print(f"jury {member['name']}: done")
        del inst
    cnt = support.sum(1)
    payload = {
        "spec": "post-hoc control added before interpreting consensus.json (2026-10-11)",
        "n": a.n, "n_valid": int(ok.sum()),
        "area_pool": {"n": len(area_pool), "median": float(np.median(area_pool))},
        "support_hist": np.bincount(cnt[ok], minlength=len(jury) + 1).tolist(),
        "mean_support": float(cnt[ok].mean()),
        "ge3_rate": float((cnt[ok] >= 3).mean()),
        "le1_rate": float((cnt[ok] <= 1).mean()),
    }
    (a.out / "null_control.json").write_text(json.dumps(payload, indent=2))
    print(f"null: n_valid={payload['n_valid']} mean={payload['mean_support']:.2f} "
          f"ge3={payload['ge3_rate']:.3f} le1={payload['le1_rate']:.3f}")


if __name__ == "__main__":
    main()
