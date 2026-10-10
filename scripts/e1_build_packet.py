#!/usr/bin/env python
"""E1 候选盲审材料包构建器：split1_seed1 / fold2（验证折），四源 200 窗口，两阶段盲评。

    python scripts/e1_build_packet.py \
        --round runs/analysis/matched_seed_20261004 --pair split1_seed1 \
        --records runs/analysis/e1_input_check_20261010_split1_seed1/records.npz \
        --out runs/analysis/e1_packet_20261010 --seed 20261010

产物：stage1/（图像+十字，中性编号）、stage2/（同窗+候选轮廓洋红+GT 分色轮廓）、
forms/（两阶段表头）、KEY/mapping.json（中性 ID↔真实身份，只给组织者）、sample.json、
manifest.json（SHA/种子/配额）、README_reviewers.md。阶段一材料不含任何模型来源、置信度、
GT 或面积信息。fold 固定 split1 验证折 2；fold3 拒绝。tissue 统一用组织名字符串分层。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold  # noqa: E402
from audit_existence_candidates import audit_pair  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.e1_audit import (background_windows, draw_cross, overlay_outlines,  # noqa: E402
                             stratified_sample, window_slice)

GT_COLORS = [(180, 40, 40), (40, 180, 40), (40, 40, 220), (200, 120, 200), (40, 200, 220)]
CAND_COLOR = (255, 0, 255)   # 候选轮廓：洋红


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def neutral_ids(n: int, rng) -> list[str]:
    """Shuffled neutral stage ids IMG_0000..; the stage directory order carries no source info."""
    ids = [f"IMG_{i:04d}" for i in range(n)]
    order = rng.permutation(n)
    return [ids[i] for i in order]


def _centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.mean()), int(xs.mean())


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--pair", default="split1_seed1")
    ap.add_argument("--records", type=Path, default=None,
                    help="今日 replay records.npz；给定则逐列绑定，不等则退出")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    ensure_dev_fold(a.fold, 1)
    if a.fold != 2:
        raise SystemExit("E1 sampling frame is fixed to split1's validation fold 2 (spec §1)")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")

    # ---- 候选重建（与今日 replay 同一代码路径）+ 可选冻结绑定
    rows = audit_pair(a.round, a.pair, 1, 1, a.out, a.workers)
    tissue_arr = PanNukeFold(2).tissue
    for r in rows:
        r["tissue"] = str(tissue_arr[r["image"]])
    if a.records is not None:
        z = np.load(a.records)
        sel = z["pair"] == a.pair
        for col in ("image", "id", "class", "area", "border", "added", "matched"):
            mine = np.array([r[col] for r in rows])
            ref = np.asarray(z[col])[sel]
            if not np.array_equal(mine.astype(ref.dtype), ref):
                raise SystemExit(f"binding failed: column {col} diverges from frozen records")

    f2 = PanNukeFold(2)
    selection = json.loads((a.round / "p2" / a.pair / "selection.json").read_text())
    base_val = a.round / "p0" / a.pair / "val" / "base" / "ms" / "pred.npz"
    x2_val = Path(selection["provenance"]["inputs"]["x2"]["path"])
    # NpzFile 逐图索引会整组重新解压（findings 2026-10-08 E0 同款坑）：先物化一次
    base = dict(np.load(base_val))
    x2 = dict(np.load(x2_val))
    base_inst, x2_inst = base["inst"], x2["inst"]

    # ---- 四源抽样（rng 使用顺序固定：S1 → S2 → S3 → S4）
    rng = np.random.default_rng(a.seed)
    s1_rows = []
    for j in range(len(f2)):
        ids = np.unique(base_inst[j]); ids = ids[ids > 0]
        for i in ids:
            s1_rows.append({"tissue": str(f2.tissue[j]), "border": _touches_edge(base_inst[j] == i),
                            "image": int(j), "id": int(i)})
    s1 = stratified_sample(s1_rows, 50, rng)
    s2 = stratified_sample([r for r in rows if r["added"]], 50, rng)
    s3 = stratified_sample([r for r in rows if not r["added"]], 50, rng)
    s4 = background_windows(np.array([str(t) for t in f2.tissue]), 50, rng)
    assert len(s1) == 50 and len(s2) == 50 and len(s3) == 50 and len(s4) == 50

    items = []
    for r in s1:
        mask = base_inst[r["image"]] == r["id"]
        cy, cx = _centroid(mask)
        items.append({"source": "S1", "image": r["image"], "id": r["id"], "cy": cy, "cx": cx,
                      "tissue": r["tissue"], "border": r["border"], "prob": r["prob"]})
    for src, chosen in (("S2", s2), ("S3", s3)):
        for r in chosen:
            mask = x2_inst[r["image"]] == r["id"]
            area = int(mask.sum())
            assert area > 0 and abs(area - r["area"]) <= 1, \
                f"{src} mask/image mismatch img {r['image']} id {r['id']}"
            cy, cx = _centroid(mask)
            items.append({"source": src, "image": r["image"], "id": r["id"], "cy": cy, "cx": cx,
                          "tissue": r["tissue"], "border": r["border"], "prob": r["prob"],
                          "matched": bool(r["matched"]), "area": r["area"], "pmax": r["pmax"]})
    for w in s4:
        items.append({"source": "S4", "image": w["image"], "id": None, "cy": w["cy"],
                      "cx": w["cx"], "tissue": w["tissue"], "border": None, "prob": w["prob"]})

    ids = neutral_ids(len(items), rng)
    mapping = {nid: it for nid, it in zip(ids, items)}

    # ---- 材料包
    for sub in ("stage1", "stage2", "forms", "KEY"):
        (a.out / sub).mkdir(parents=True)
    for nid, it in mapping.items():
        img = np.asarray(f2.images[it["image"]])
        sy, sx, _ = window_slice(img.shape[:2], it["cy"], it["cx"])
        win = np.ascontiguousarray(img[sy, sx][..., ::-1])          # RGB -> BGR
        cy_w, cx_w = it["cy"] - sy.start, it["cx"] - sx.start
        marked = draw_cross(win, cy_w, cx_w)
        cv2.imwrite(str(a.out / "stage1" / f"{nid}.png"), marked)
        masks, colors = [], []
        if it["id"] is not None:
            src = base_inst if it["source"] == "S1" else x2_inst
            masks.append((src[it["image"]] == it["id"])[sy, sx])
            colors.append(CAND_COLOR)
        gt = np.asarray(f2.inst[it["image"]])[sy, sx]
        gt_type = np.asarray(f2.type[it["image"]])[sy, sx]
        for gid in np.unique(gt)[1:]:
            cls = int(np.round(gt_type[gt == gid].mean()))
            masks.append(gt == gid)
            colors.append(GT_COLORS[cls - 1] if 1 <= cls <= 5 else (128, 128, 128))
        cv2.imwrite(str(a.out / "stage2" / f"{nid}.png"),
                    overlay_outlines(marked, masks, colors))

    (a.out / "KEY" / "mapping.json").write_text(json.dumps(mapping, indent=2, default=str))
    (a.out / "sample.json").write_text(json.dumps(
        {"spec": "docs/superpowers/specs/2026-10-10-e1-blind-audit-design.md",
         "seed": a.seed, "pair": a.pair, "fold": a.fold, "items": mapping}, indent=2, default=str))
    (a.out / "manifest.json").write_text(json.dumps({
        "spec": "docs/superpowers/specs/2026-10-10-e1-blind-audit-design.md",
        "seed": a.seed, "counts": {"S1": 50, "S2": 50, "S3": 50, "S4": 50},
        "sha256": {str(p): _sha256(p) for p in
                   [a.records, base_val, x2_val, f2.dir / "images.npy", f2.dir / "inst.npy",
                    f2.dir / "type.npy"] if p is not None and Path(p).exists()},
    }, indent=2))
    (a.out / "forms" / "stage1_form.csv").write_text(
        "id,reviewer,is_nucleus(real/non_nucleus/uncertain),completeness(full/truncated/fragments),notes\n")
    (a.out / "forms" / "stage2_form.csv").write_text(
        "id,reviewer,relation(consistent/boundary_offset/merged/oversegmented/possible_unlabelled/"
        "not_nucleus/uncertain),notes\n")
    (a.out / "README_reviewers.md").write_text(
        "# E1 盲评说明\n\n阶段一：只看 stage1/ 图（红十字=目标位置），填 forms/stage1_form.csv。\n"
        "阶段二（阶段一答案封存后才开始）：看 stage2/（洋红=候选轮廓，彩色=GT 轮廓），填 "
        "forms/stage2_form.csv。请勿在提交阶段一之前查看 stage2 或交换意见。\n")
    print(f"wrote packet: {a.out} ({len(items)} windows)")


if __name__ == "__main__":
    main()
