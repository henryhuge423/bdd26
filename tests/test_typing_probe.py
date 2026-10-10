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


def test_bootstrap_delta_identical_and_stratified():
    gt_ch, gt_inst, gt_type, pred_inst, pred_type, tissue = _toy_fold()
    img, iid = _tables(pred_inst)
    ev = RetypeEvaluator(gt_ch, pred_inst, img, iid, tissue, workers=1)
    rng = np.random.default_rng(0)
    cls = rng.integers(1, 6, len(iid))
    r = ev.mpq_bootstrap_delta(cls, cls, n_boot=64, seed=1)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    r2 = ev.mpq_bootstrap_delta(cls, cls, n_boot=64, seed=1)
    assert r2 == r                                                   # 种子确定
    # 点估计 == mpq(a) - mpq(b)
    assert abs(r["delta"] - (ev.mpq(cls)[0] - ev.mpq(cls)[0])) < 1e-12
    # 不同分型的 CI 应有宽度（对本玩具数据两种随机分型差的分布非退化）
    cls_b = rng.integers(1, 6, len(iid))
    rb = ev.mpq_bootstrap_delta(cls, cls_b, n_boot=256, seed=2)
    assert rb["hi"] - rb["lo"] > 0


def test_pool_and_ring():
    from nucseg.typing_probe import pool_instances, pool_rings
    f = np.zeros((8, 8, 2), np.float32); f[1:3, 1:3] = (1., 2.); f[5:7, 5:7] = (3., 4.)
    inst = np.zeros((8, 8), np.int32); inst[1:3, 1:3] = 1; inst[5:7, 5:7] = 2
    pooled = pool_instances(f, inst, np.array([1, 2]))
    assert np.allclose(pooled, [[1., 2.], [3., 4.]], atol=1e-6)
    # 环必须不含任何实例像素：用指示特征验证（膨胀1px−全部实例）
    f_id = (inst > 0).astype(np.float32)[..., None]
    r_id = pool_rings(f_id, inst, np.array([1, 2]), width=1)
    assert r_id.sum() == 0
    # 非零背景特征确实进环
    f_bg = np.ones((8, 8, 1), np.float32); f_bg[inst > 0] = 0
    r_bg = pool_rings(f_bg, inst, np.array([1, 2]), width=1)
    assert (r_bg > 0).all()
    # 满图实例 → 空环 = 0 向量
    inst_full = np.ones((8, 8), np.int32)
    assert pool_rings(np.ones((8, 8, 2), np.float32), inst_full, np.array([1]), width=3)[0].sum() == 0


def test_t1_rule_drops_background_channel():
    from nucseg.typing_probe import t1_classes
    p = np.array([[.6, .3, .1, 0, 0, 0], [.05, .02, .03, 0, 0, .9]], np.float32)
    assert list(t1_classes(p)) == [1, 5]          # 第1行背景最大但被丢弃→类1；第2行→类5
    assert t1_classes(p).min() >= 1


def test_linear_head_separable_and_seeded():
    import numpy as np
    from nucseg.typing_probe import train_linear_head, apply_head, sklearn_logreg
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-1, .3, (64, 4)), rng.normal(1, .3, (64, 4))]).astype(np.float32)
    y = np.array([0] * 64 + [1] * 64)
    # epochs=300（128 样本单批 ×300 步）足以越过随机初始化;margin;spec 的 20-epoch 配方是
    # 63k 样本（~320 步、强信号）的部署值，不由本测试背书
    W1, b1 = train_linear_head(X, y, seed=0, epochs=300)
    assert (apply_head(W1, b1, X) == np.where(y == 0, 1, 2)).all()   # 可分数据全对，输出1..5域
    W2, _ = train_linear_head(X, y, seed=1, epochs=300)
    assert not np.allclose(W1, W2)                                    # 种子确实改变初始化/洗牌
    assert (sklearn_logreg(X, y) == np.where(y == 0, 1, 2)).all()
