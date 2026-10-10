#!/usr/bin/env python
"""T 实验（typing probe）特征导出：冻结检查点单次前向，导出每实例 tp_feat / 24px 环 / tp softmax 池化。

    python scripts/export_tp_features.py --run runs/cellvit_uni/split1 --fold 2 \
        --pred runs/cellvit_uni/split1/pred_fold2.npz \
        --out runs/analysis/typing_probe_20261010/feat_fold2.npz [--batch-size 32]
    python scripts/export_tp_features.py --run runs/cellvit_uni/split1 --fold 1 \
        --out runs/analysis/typing_probe_20261010/feat_fold1.npz

fold1 在 GT 实例掩码上池化（cls = GT 类，T2/T3 训练特征）；fold2 在预测实例掩码上池化
（prob 池化用于与 pred npz inst_prob 的绑定检查），另附 fold2 GT 实例诊断行（gt_* 键，
spec §2"GT 区域 vs 预测轮廓单列"）。fold2 的 T0 多数票类不由本脚本导出——runner 直接从
pred npz 的 type 图重算，避免两处逻辑漂移。只允许 split1 的 train/val 折（fold 1/2）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold, run_split  # noqa: E402

from nucseg.cellvit.engine import TrainConfig, build_model, run_dead_expert, run_widen  # noqa: E402
from nucseg.cellvit.model import UNI_MEAN, UNI_STD  # noqa: E402
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.typing_probe import RING_WIDTH, pool_instances, pool_rings  # noqa: E402


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _forward(model, imgs_u8: torch.Tensor):
    """imgs_u8 (B,H,W,3) uint8 cuda -> tp softmax (B,H,W,6), tp_feat (B,H,W,64), both float32 NHWC."""
    mean = torch.tensor(UNI_MEAN, device=imgs_u8.device)
    std = torch.tensor(UNI_STD, device=imgs_u8.device)
    x = ((imgs_u8.float() / 255.0 - mean) / std).permute(0, 3, 1, 2).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model(x, return_features=True)
    tp = torch.softmax(p["tp"].float(), -1).cpu().numpy()
    feat = p["tp_feat"].permute(0, 2, 3, 1).float().cpu().numpy()
    return tp, feat


def _gt_classes(inst: np.ndarray, typ: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Majority GT class per raw instance id (background votes zeroed; ties -> lowest class)."""
    votes = np.zeros((int(ids.max()) + 1, 6), np.int64)
    np.add.at(votes, (inst.ravel(), typ.ravel().astype(np.int64)), 1)
    votes[:, 0] = 0
    cls = votes[ids].argmax(1) + 1
    cls[votes[ids].sum(1) == 0] = 0
    return cls


def _border(inst: np.ndarray, ids: np.ndarray) -> np.ndarray:
    edge = np.zeros(inst.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    hit = np.zeros(int(ids.max()) + 1, bool)
    hit[inst[edge]] = True
    return hit[ids]


def _ids_of(inst_map: np.ndarray) -> np.ndarray:
    ids = np.unique(inst_map)
    return ids[ids > 0].astype(np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True,
                    help="split1 的 train/val 折（1/2）；fold3 由 fold_guard 拒绝")
    ap.add_argument("--pred", type=Path, default=None, help="fold2: pred npz with inst + tables")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="final.pth")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    ensure_dev_fold(a.fold, run_split(a.run))
    if a.fold == 2 and a.pred is None:
        raise SystemExit("--pred is required for --fold 2 (binding check against inst_prob)")

    f = PanNukeFold(a.fold)
    cfg = TrainConfig(split=1, out_dir=str(a.run), dead_expert=run_dead_expert(a.run),
                      widen=run_widen(a.run))
    model = build_model(cfg, pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu", weights_only=False)["model"])

    pred = tables = None
    if a.fold == 2:
        pred = dict(np.load(a.pred))
        tables = (pred["inst_img"].astype(np.int64), pred["inst_id"].astype(np.int64))

    rows = {k: [] for k in ("inst_img", "inst_id", "feat", "ring", "prob", "area", "border", "cls")}
    gt_rows = {k: [] for k in ("inst_img", "inst_id", "feat", "ring", "cls")}
    n_empty_ring = 0
    t_pool = 0.0
    with torch.no_grad():
        for s in range(0, len(f), a.batch_size):
            imgs = torch.from_numpy(np.asarray(f.images[s:s + a.batch_size])).cuda()
            tp, feat = _forward(model, imgs)
            for b in range(len(tp)):
                t0 = time.perf_counter()
                j = s + b
                gt_map = np.asarray(f.inst[j], np.int32)
                gt_ids = _ids_of(gt_map)
                # --- 预测/GT 主行 ---
                if a.fold == 2:
                    inst_map = pred["inst"][j].astype(np.int32)
                    ids = tables[1][tables[0] == j]
                    cls = None
                else:
                    inst_map = gt_map
                    ids = gt_ids
                    cls = _gt_classes(gt_map, np.asarray(f.type[j]), gt_ids)
                if len(ids):
                    fe = pool_instances(feat[b], inst_map, ids)
                    ri = pool_rings(feat[b], inst_map, ids, RING_WIDTH)
                    pr = pool_instances(tp[b], inst_map, ids)
                    ar = np.bincount(inst_map.ravel(), minlength=int(ids.max()) + 1)[ids]
                    bo = _border(inst_map, ids)
                    n_empty_ring += int((ri == 0).all(1).sum())
                else:
                    fe = np.zeros((0, feat.shape[-1]), np.float32)
                    ri = pr = fe.copy()
                    ar = np.zeros(0, np.int64); bo = np.zeros(0, bool)
                rows["inst_img"].append(np.full(len(ids), j)); rows["inst_id"].append(ids)
                rows["feat"].append(fe); rows["ring"].append(ri); rows["prob"].append(pr)
                rows["area"].append(ar); rows["border"].append(bo)
                if cls is not None:
                    rows["cls"].append(cls)
                # --- fold2 GT 诊断行 ---
                if a.fold == 2 and len(gt_ids):
                    gfe = pool_instances(feat[b], gt_map, gt_ids)
                    gri = pool_rings(feat[b], gt_map, gt_ids, RING_WIDTH)
                    n_empty_ring += int((gri == 0).all(1).sum())
                    gt_rows["inst_img"].append(np.full(len(gt_ids), j))
                    gt_rows["inst_id"].append(gt_ids)
                    gt_rows["feat"].append(gfe); gt_rows["ring"].append(gri)
                    gt_rows["cls"].append(_gt_classes(gt_map, np.asarray(f.type[j]), gt_ids))
                t_pool += time.perf_counter() - t0
            print(f"\r{s + len(tp)}/{len(f)}", end="", flush=True)
    print()

    payload = {k: np.concatenate(v) for k, v in rows.items()}
    payload["tissue"] = f.tissue[payload["inst_img"]]
    binding = None
    if a.fold == 2:
        assert np.array_equal(payload["inst_img"], tables[0]), "row order drifted from pred table"
        assert np.array_equal(payload["inst_id"], tables[1]), "row order drifted from pred table"
        d = float(np.abs(payload["prob"] - pred["inst_prob"]).max())
        if not d < 1e-4:
            raise SystemExit(f"binding check failed: pooled tp softmax vs inst_prob max|d|={d}")
        binding = d
        for k, v in gt_rows.items():
            payload[f"gt_{k}"] = np.concatenate(v)
        payload["gt_tissue"] = f.tissue[payload["gt_inst_img"]]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **payload)
    sidecar = {
        "run": str(a.run), "ckpt": a.ckpt, "fold": a.fold, "ring_width": RING_WIDTH,
        "batch_size": a.batch_size, "n_rows": int(len(payload["inst_id"])),
        "n_gt_rows": int(len(payload.get("gt_inst_id", []))),
        "n_empty_ring": int(n_empty_ring), "pool_ms_per_img": t_pool * 1000 / len(f),
        "binding_max_abs_diff": binding,
        "sha256": {str(p): _sha256(p) for p in
                   [a.run / a.ckpt, a.pred, f.dir / "images.npy", f.dir / "inst.npy",
                    f.dir / "type.npy"] if p is not None and Path(p).exists()},
        "spec": "docs/superpowers/specs/2026-10-10-typing-probe-t-design.md",
    }
    a.out.with_suffix(".npz.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {a.out} rows={sidecar['n_rows']} gt_rows={sidecar['n_gt_rows']} "
          f"empty_rings={n_empty_ring} pool={sidecar['pool_ms_per_img']:.2f} ms/img "
          f"binding={binding}")


if __name__ == "__main__":
    main()
