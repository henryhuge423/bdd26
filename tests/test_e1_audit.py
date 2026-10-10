"""E1 候选盲审配套测试：分层抽样、窗口/绘图原语、材料包守卫、冻结分析。

Spec: docs/superpowers/specs/2026-10-10-e1-blind-audit-design.md（用户 2026-10-10 批准）。
"""
import numpy as np
import pytest

from nucseg.e1_audit import background_windows, stratified_sample, window_slice


def test_stratified_sample_counts_probs_merge():
    rows = ([{"tissue": 0, "border": True, "k": i} for i in range(100)] +
            [{"tissue": 0, "border": False, "k": i} for i in range(100, 300)] +
            [{"tissue": 1, "border": False, "k": i} for i in range(300, 320)])
    rng = np.random.default_rng(20261010)
    sel = stratified_sample(rows, 50, rng)
    assert len(sel) == 50 and len({r["k"] for r in sel}) == 50          # 无放回
    # tissue1 组 N=20 配额=50*20/320=3.125≥3 不合并；概率=组内入样率
    t1 = [r for r in sel if r["tissue"] == 1]
    assert abs(len(t1) - 50 * 20 / 320) <= 1
    assert all(r["prob"] == pytest.approx(len(t1) / 20) for r in t1)
    # 确定性
    assert [r["k"] for r in stratified_sample(rows, 50, np.random.default_rng(20261010))] \
        == [r["k"] for r in sel]
    # 合并规则：tissue1 border=True 只有 2 个（配额<3）并入同 tissue 另一 border 组
    rows2 = ([{"tissue": 0, "border": False, "k": i} for i in range(28)] +
             [{"tissue": 1, "border": True, "k": 100 + i} for i in range(2)] +
             [{"tissue": 1, "border": False, "k": 200 + i} for i in range(30)])
    sel2 = stratified_sample(rows2, 20, np.random.default_rng(0))
    t1b = [r for r in sel2 if r["tissue"] == 1]
    # 合并后 tissue1 的两个组是一个单元 N=32；组内概率=len(抽中 tissue1)/32
    # （该断言本身证明合并发生：未合并时 tissue1 的两组会各有 <3 或 0 的独立概率）
    assert all(r["prob"] == pytest.approx(len(t1b) / 32) for r in t1b)
    assert len(t1b) > 0


def test_background_windows_probs_and_clamp():
    tissue = np.zeros(10, int); tissue[5:] = 1
    rng = np.random.default_rng(1)
    w = background_windows(tissue, 10, rng)
    assert len(w) == 10
    C = (256 - 160 + 1) ** 2
    # 分层配额：每组织 5 图，π = (5/5)·(1/C) —— 每组织恰好 5 张图全选
    assert all(x["prob"] == pytest.approx(1.0 / C) for x in w)
    assert all(80 <= x["cy"] <= 176 and 80 <= x["cx"] <= 176 for x in w)   # 有效中心范围
    assert len({x["image"] for x in w}) == 10                              # 无放回（图级）


def test_window_slice_constant_size():
    for cy, cx in [(0, 0), (255, 255), (80, 176), (0, 255)]:
        sy, sx, off = window_slice((256, 256), cy, cx, size=160)
        assert sy.stop - sy.start == 160 and sx.stop - sx.start == 160    # 尺寸恒定
        assert off == (sy.start, sx.start)                                 # 偏移=窗口左上角
    assert window_slice((256, 256), 0, 0)[2] == (0, 0)
    assert window_slice((256, 256), 255, 255)[2] == (96, 96)


def test_packet_cli_guards_and_neutral_ids(tmp_path):
    import sys
    from scripts.e1_build_packet import main, neutral_ids
    ids = neutral_ids(200, np.random.default_rng(20261010))
    assert len(set(ids)) == 200 and all(i.startswith("IMG_") for i in ids)
    out = tmp_path / "pkt"; out.mkdir()
    # 非法折 / 输出已存在都要拒绝（不起 GPU、不读数据）
    with pytest.raises(SystemExit, match="TEST fold"):
        main(["--out", str(out), "--fold", "3", "--round", str(tmp_path), "--pair", "split1_seed1"])
    with pytest.raises(SystemExit, match="exists"):
        main(["--out", str(out), "--fold", "2", "--round", str(tmp_path), "--pair", "split1_seed1"])


def test_ht_and_kappa():
    from scripts.e1_analyze import ht_estimate, kappa
    labels = [1, 1, 0, 1]
    probs = [.5, .25, .25, 1.0]
    # HT：Σ w·y / Σ w，w = 1/π
    assert ht_estimate(labels, probs) == pytest.approx(
        (2 * 1 + 4 * 1 + 4 * 0 + 1 * 1) / (2 + 4 + 4 + 1))
    assert kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    # uncertain 行（-1）从 kappa 分母剔除
    assert kappa([1, -1, 0, 0], [1, 1, 0, 0]) == 1.0
