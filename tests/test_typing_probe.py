"""T 实验（typing probe）配套测试：fast_retype strict、bootstrap、pooling/环/头、runner 纯逻辑。

Spec: docs/superpowers/specs/2026-10-10-typing-probe-t-design.md（用户 2026-10-10 批准，值冻结）。
"""
import numpy as np
import pytest

from nucseg.constants import NUM_CLASSES, TISSUES
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.pannuke_eval import evaluate


def _toy_fold(n=4):
    """2 组织 × n 图；每图 2 个 GT 核（类 1 与类 3），预测同几何、类 1 保持、类 3 错标为类 5。"""
    gt_ch = np.zeros((n, 8, 8, NUM_CLASSES), np.uint16)
    gt_inst = np.zeros((n, 8, 8), np.int32)
    pred_inst = np.zeros((n, 8, 8), np.int32)
    tissue = np.array([TISSUES[0]] * (n // 2) + [TISSUES[1]] * (n - n // 2))  # 组织名，非索引
    for i in range(n):
        gt_ch[i, 1:4, 1:4, 0] = 1; gt_ch[i, 5:7, 5:7, 2] = 1
        gt_inst[i, 1:4, 1:4] = 1; gt_inst[i, 5:7, 5:7] = 2
        pred_inst[i] = gt_inst[i]
    gt_type = np.where(gt_inst == 1, 1, np.where(gt_inst == 2, 3, 0)).astype(np.uint8)
    pred_type = np.where(pred_inst == 1, 1, np.where(pred_inst == 2, 5, 0)).astype(np.uint8)
    # 预测类 5 在图中无 GT → strict 罚 0
    return gt_ch, gt_inst, gt_type, pred_inst, pred_type, tissue


def _tables(pred_inst):
    img = np.concatenate([np.full(len(np.unique(p[p > 0])), j) for j, p in enumerate(pred_inst)])
    iid = np.concatenate([np.unique(p[p > 0]) for p in pred_inst])
    return img, iid


def test_mpq_strict_matches_canonical():
    gt_ch, gt_inst, gt_type, pred_inst, pred_type, tissue = _toy_fold()
    canon = evaluate(gt_ch, gt_inst, gt_type, tissue, pred_inst, pred_type, workers=1)
    img, iid = _tables(pred_inst)
    ev = RetypeEvaluator(gt_ch, pred_inst, img, iid, tissue, workers=1)
    cls = np.where(iid == 1, 1, 5)
    m_fast, per_fast = ev.mpq(cls)
    assert abs(m_fast - canon["summary"]["official"]["mPQ"]) < 1e-9   # 既有官方行为不回归
    s_fast, per_s_fast = ev.mpq_strict(cls)
    assert abs(s_fast - canon["summary"]["strict"]["mPQ"]) < 1e-9
    assert s_fast < m_fast                                          # 缺类 FP 确实被罚
