"""Runner tests: mini-world end-to-end, allowlist guard, overwrite guard, binding rejections."""
import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path

from scripts.merged_class_diag import ALLOWED_PREDS, analyze_split, main


def _rect(a, i, y, x, h, w):
    a[y:y + h, x:x + w] = i


@dataclass
class _Fold:
    gt_channels: np.ndarray
    tissue: np.ndarray


def _mini_world():
    # 2 图 × 5 类 GT 通道；pred 含跨类错分（Inf<->Conn）、漏检 Dead、拆半 Conn、1 个 untyped(cls=0)
    H = W = 32
    gt = np.zeros((2, H, W, 5), np.int32)
    _rect(gt[0, :, :, 0], 1, 2, 2, 6, 6)      # img0 Neo
    _rect(gt[0, :, :, 1], 1, 12, 4, 5, 5)     # img0 Inf
    _rect(gt[0, :, :, 2], 1, 4, 20, 6, 6)     # img0 Conn
    _rect(gt[0, :, :, 3], 1, 20, 20, 4, 4)    # img0 Dead（pred 漏检）
    _rect(gt[0, :, :, 4], 1, 22, 2, 5, 5)     # img0 Epi
    _rect(gt[1, :, :, 1], 1, 3, 3, 7, 7)      # img1 Inf
    _rect(gt[1, :, :, 2], 1, 14, 14, 8, 8)    # img1 Conn（pred 拆两半，各 IoU=.5 不匹配）
    inst = np.zeros((2, H, W), np.int32); typ = np.zeros((2, H, W), np.uint8)
    _rect(inst[0], 1, 2, 2, 6, 6); typ[0][inst[0] == 1] = 1   # Neo 对
    _rect(inst[0], 2, 12, 4, 5, 5); typ[0][inst[0] == 2] = 2  # Inf 对
    _rect(inst[0], 3, 4, 20, 6, 6); typ[0][inst[0] == 3] = 2  # Conn 说成 Inf（merge(2,3) 修复）
    _rect(inst[0], 4, 22, 2, 5, 5); typ[0][inst[0] == 4] = 5  # Epi 对
    _rect(inst[1], 1, 3, 3, 7, 7); typ[1][inst[1] == 1] = 3   # Inf 说成 Conn（merge(2,3) 修复）
    _rect(inst[1], 2, 14, 14, 4, 8); _rect(inst[1], 3, 18, 14, 4, 8)
    typ[1][np.isin(inst[1], (2, 3))] = 2                     # 拆半的 Conn
    _rect(inst[1], 4, 24, 2, 4, 4)                           # spurious，全 0 type -> cls=0
    pred = {"inst": inst, "type": typ}
    fold = _Fold(gt, np.array(["Breast", "Lung"]))
    # frozen=None 时 analyze_split 跳过冻结比对，只做内部不变量（Dead 列逐位、bPQ 同值）。
    return pred, fold


def test_analyze_split_mini():
    pred, fold = _mini_world()
    out = analyze_split(pred, fold, split=1, frozen=None)
    assert set(out) >= {"m0", "merges", "oracle_under_merge", "invariants"}
    assert out["invariants"]["dead_pq_unchanged"]
    assert len(out["merges"]) == 10 and "(2,3)" in out["merges"]
    # 装配层不破坏恒等性：M0 == RetypeEvaluator 直算
    from nucseg.metrics.fast_retype import RetypeEvaluator
    from nucseg.metrics.pannuke_eval import instance_classes
    ii, iid = [], []
    for j in range(2):
        ids = np.unique(pred["inst"][j]); ids = ids[ids > 0]
        ii.append(np.full(len(ids), j)); iid.append(ids)
    ev = RetypeEvaluator(fold.gt_channels, pred["inst"], np.concatenate(ii),
                         np.concatenate(iid), fold.tissue, workers=1)
    cls = np.concatenate([instance_classes(pred["inst"][j], pred["type"][j])[1] for j in range(2)])
    assert out["m0"]["mpq"] == ev.mpq(cls)[0]


def test_bindings_reject_mismatch():
    pred, fold = _mini_world()
    out = analyze_split(pred, fold, split=1, frozen=None)
    names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    wrong = {"official": {"mPQ": out["m0"]["mpq"] + 0.01, "bPQ": 0.5},
             "per_class_PQ": {k: v + 0.01 for k, v in zip(names, out["m0"]["per_class"])}}
    with pytest.raises(SystemExit):
        analyze_split(pred, fold, split=1, frozen=wrong)
    right = {"official": {"mPQ": out["m0"]["mpq"], "bPQ": out["m0"]["bpq_recomputed"]},
             "per_class_PQ": dict(zip(names, out["m0"]["per_class"]))}
    analyze_split(pred, fold, split=1, frozen=right)  # 不抛


def test_allowlist_guard(tmp_path):
    bogus = tmp_path / "pred_evil.npz"
    bogus.write_bytes(b"")
    with pytest.raises(SystemExit):
        main(["--pred", f"1={bogus}", "--out", str(tmp_path / "o")])


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "already"; out.mkdir(); (out / "results.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--out", str(out)])
