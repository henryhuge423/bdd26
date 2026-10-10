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


def test_export_cli_guards(tmp_path, monkeypatch):
    import sys
    run = tmp_path / "run"; run.mkdir()
    (run / "config.json").write_text('{"split": 1}')
    pred = tmp_path / "p.npz"
    np.savez(pred, inst=np.zeros((1, 8, 8), np.int32))
    from scripts import export_tp_features as ex
    # split1 的测试折 fold3 拒绝
    with pytest.raises(SystemExit, match="TEST fold"):
        ex.main(["--run", str(run), "--fold", "3", "--pred", str(pred),
                 "--out", str(tmp_path / "o.npz")])
    # 输出已存在拒绝覆盖
    out = tmp_path / "o.npz"
    out.write_bytes(b"x")
    with pytest.raises(SystemExit, match="exists"):
        ex.main(["--run", str(run), "--fold", "1", "--pred", str(pred), "--out", str(out)])


def test_t0_classes_alignment_guard():
    from scripts.run_typing_probe import t0_classes
    # relabel 枚举与 inst_id 表不同序时必须报错（Review Focus #1）
    inst = np.zeros((1, 8, 8), np.int32); inst[0, 1:3, 1:3] = 7      # 非常规 id
    typ = np.where(inst > 0, 3, 0).astype(np.uint8)
    with pytest.raises(AssertionError):
        t0_classes(inst, typ, np.array([0]), np.array([3]))           # 表里写 id=3，图里是 7
    # 一致的表返回多数票类
    cls = t0_classes(inst, typ, np.array([0]), np.array([7]))
    assert list(cls) == [3]


def test_evaluate_decision_rules():
    from scripts.run_typing_probe import evaluate_decision
    ok = {"delta": .004, "lo": .001, "hi": .008, "strict_delta": .0005, "dead_delta": .0,
          "beats_t1": True, "seeds_same_direction": True}
    assert evaluate_decision(ok)["pass_gate"] is True
    assert evaluate_decision({**ok, "delta": .002})["pass_gate"] is False       # ΔmPQ<.003
    assert evaluate_decision({**ok, "lo": -.001})["pass_gate"] is False         # CI 跨零
    assert evaluate_decision({**ok, "strict_delta": -.001})["pass_gate"] is False
    assert evaluate_decision({**ok, "dead_delta": -.003})["pass_gate"] is False  # Dead 掉>.002
    assert evaluate_decision({**ok, "beats_t1": False})["pass_gate"] is False
    assert evaluate_decision({**ok, "seeds_same_direction": False})["pass_gate"] is False


def test_forward_layout_contract():
    """_forward 必须返回 NHWC 且 tp 在类轴上归一化（曾漏 permute：softmax 作用在 W 轴）。"""
    import torch
    from scripts.export_tp_features import _forward

    class Stub(torch.nn.Module):
        def forward(self, x, return_features=False):
            b, _, h, w = x.shape
            out = {"tp": torch.arange(6 * h * w, dtype=torch.float32).reshape(1, 6, h, w) % 7 + .1,
                   "tp_feat": torch.ones(1, 64, h, w)}
            return out

    tp, feat = _forward(Stub(), torch.zeros(1, 8, 8, 3, dtype=torch.uint8))
    assert tp.shape == (1, 8, 8, 6) and feat.shape == (1, 8, 8, 64)
    assert np.allclose(tp.sum(-1), 1.0, atol=1e-5)     # 类轴归一化（permute 正确）


def test_export_finalize_and_gt_classes_empty():
    """fold2 不产 cls 列（空列表不能 np.concatenate 崩）；空实例图的 _gt_classes 须返回空。"""
    from scripts.export_tp_features import _finalize_rows, _gt_classes
    rows = {"inst_img": [np.array([0, 0])], "cls": []}      # fold2: cls 永不追加
    payload = _finalize_rows(rows)
    assert payload["cls"].shape == (0,)
    assert payload["inst_img"].tolist() == [0, 0]
    assert _gt_classes(np.zeros((4, 4), np.int32),
                       np.zeros((4, 4), np.uint8), np.zeros(0, np.int64)).shape == (0,)


def test_export_empty_rows_widths():
    """零实例图的空行必须按列宽生成（prob=6 维，feat/ring=64 维）——曾把 64 维 feat 拷给 prob。"""
    from scripts.export_tp_features import _empty_rows
    fe, ri, pr = _empty_rows(n_feat=64, n_tp=6)
    assert fe.shape == (0, 64) and ri.shape == (0, 64) and pr.shape == (0, 6)
    assert fe.dtype == np.float32 and pr.dtype == np.float32


def test_gt_classes_values():
    """GT 类别必须是 1..5 本身（6 列投票 argmax 索引即类别；曾错加 +1 整体偏移）。"""
    from scripts.export_tp_features import _gt_classes
    inst = np.zeros((8, 8), np.int32); inst[1:3, 1:3] = 7; inst[5:7, 5:7] = 9
    typ = np.zeros((8, 8), np.uint8); typ[1:3, 1:3] = 3; typ[5:7, 5:7] = 5
    cls = _gt_classes(inst, typ, np.array([7, 9]))
    assert list(cls) == [3, 5]
    # 背景污染票不影响多数票
    typ2 = typ.copy(); typ2[0, 0] = 4                      # 背景像素的杂散票
    assert list(_gt_classes(inst, typ2, np.array([7]))) == [3]


def test_write_retyped_types_per_image():
    """重分型 type 图必须逐图建立 id→class 映射（id 每图从 1 重编号；全局 LUT 会串图）。"""
    from scripts.run_typing_probe import write_retyped_types
    inst = np.zeros((2, 6, 6), np.int32)
    inst[0, 0:2, 0:2] = 1; inst[0, 4:6, 4:6] = 2      # 图0：id1→类1，id2→类5
    inst[1, 0:2, 0:2] = 1; inst[1, 4:6, 4:6] = 3      # 图1：id1→类3，id3→类2
    inst_img = np.array([0, 0, 1, 1]); inst_id = np.array([1, 2, 1, 3])
    cls = np.array([1, 5, 3, 2])
    typ = write_retyped_types(inst, inst_img, inst_id, cls)
    from nucseg.metrics.pannuke_eval import instance_classes
    _, c0 = instance_classes(inst[0], typ[0]); _, c1 = instance_classes(inst[1], typ[1])
    assert list(c0) == [1, 5] and list(c1) == [3, 2]   # 按 id 升序：图1 的 id1→3、id3→2
