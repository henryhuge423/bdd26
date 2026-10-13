"""E4 visibility probe pure-logic tests: windows, crop grouping, miss, DiD, bootstrap."""
import numpy as np
import pytest

from nucseg.visibility_probe import (bin_area, bin_area_native, crop_group, did,
                                     max_iou_table, miss_table, original_full,
                                     rate_gap_bootstrap, sample_windows, shift_pad)


def test_miss_table_keys_by_original_ids():
    # 非连续 id {3,7}：miss/max-iou 表按原始 id 键（runner 的身份对应依赖）
    gt = np.zeros((56, 56), np.int32); _disc(gt, 3, 20, 20, 6); _disc(gt, 7, 40, 40, 6)
    pred = np.zeros((56, 56), np.int32); _disc(pred, 5, 20, 20, 6)
    m = miss_table(gt, pred)
    assert sorted(m) == [3, 7] and m[3] is False and m[7] is True
    u = max_iou_table(gt, pred)
    assert u[3] == pytest.approx(1.0) and u[7] == 0.0


def _disc(a, i, cy, cx, r):
    yy, xx = np.ogrid[: a.shape[0], : a.shape[1]]
    a[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i


def test_window_sampling_deterministic_and_range():
    w1 = sample_windows(2523, 192, 20261013)
    w2 = sample_windows(2523, 192, 20261013)
    assert np.array_equal(w1, w2)                       # 逐位确定
    assert w1.shape == (2523, 2) and w1.dtype == np.int32
    assert w1.min() >= 0 and w1.max() <= 64             # 192: {0..64}
    w3 = sample_windows(100, 160, 20261013)
    assert w3.min() >= 0 and w3.max() <= 96             # 160: {0..96}


def test_shift_pad_identity_and_edge():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    s = shift_pad(img, 2)
    assert s.shape == img.shape
    assert s[2:, 2:] == pytest.approx(img[:2, :2])       # 内容移 +y+x
    assert s[0, 0] == s[0, 1] == s[1, 0] == img[0, 0]    # 顶/左 edge 复制到角
    assert np.all(s[2:, 0] == img[:2, 0]) and np.all(s[2:, 1] == img[:2, 0])
    assert np.array_equal(shift_pad(img, 0), img)


def test_crop_visibility_exact():
    # 256 帧内圆盘 id=1 半径 10 圆心 (128,128)；合法窗偏移 ≤64：(64,64,192) 完整包含
    inst = np.zeros((256, 256), np.int32); _disc(inst, 1, 128, 128, 10)
    g = crop_group(inst, (64, 64, 192))
    a_full = int((inst == 1).sum())
    assert g[1]["group"] == "full" and g[1]["area"] == a_full and not g[1]["touches_crop"]
    assert np.array_equal(g[1]["visible"], inst[64:256, 64:256] == 1)
    # 圆心 (60,60)（窗心外侧）r=10：窗 (64,64,192) 的上/左边切过圆盘 → truncated、触两边、
    # 可见 = 圆盘 ∩ 窗内区域，裁窗坐标手工构造核对
    inst3 = np.zeros((256, 256), np.int32); _disc(inst3, 3, 60, 60, 10)
    g3 = crop_group(inst3, (64, 64, 192))
    visible = np.zeros((192, 192), bool)
    visible[(inst3[64:256, 64:256] == 3)] = True
    assert g3[3]["group"] == "truncated" and g3[3]["touches_crop"]
    assert g3[3]["area"] == int(visible.sum()) and g3[3]["area"] > 0
    assert np.array_equal(g3[3]["visible"], visible)
    # 完全落窗外
    inst2 = np.zeros((256, 256), np.int32); _disc(inst2, 2, 10, 10, 3)
    assert crop_group(inst2, (64, 64, 192))[2]["group"] == "outside"


def test_grouping_partition():
    inst = np.zeros((256, 256), np.int32)
    _disc(inst, 1, 128, 128, 8)   # 窗内完整
    _disc(inst, 2, 60, 60, 8)     # 窗 (64,64) 上/左边截断
    _disc(inst, 3, 5, 5, 5)       # 贴原图外圈（0 行/列）且完全窗外
    g = crop_group(inst, (64, 64, 192))
    of = original_full(inst)
    inside = [i for i, r in g.items() if r["group"] != "outside"]
    assert set(inside) == {1, 2}
    assert of == {1: True, 2: True, 3: False}
    # 原完整 ∩ 窗内 = truncated ∪ full（互斥完备）
    assert {g[i]["group"] for i in inside if of[i]} == {"truncated", "full"}


def test_miss_threshold():
    gt = np.zeros((56, 56), np.int32); _disc(gt, 1, 28, 28, 6)   # 离散圆盘 ≈113 px
    # 整核包含时 IoU = |gt|/|pred|；取离 .1 两侧都有裕度的半径（离散像素数避免刀锋值）
    pred = np.zeros((56, 56), np.int32); _disc(pred, 5, 28, 28, 20)   # ≈1257 px, IoU≈.090 < .1
    assert miss_table(gt, pred)[1] is True
    pred = np.zeros((56, 56), np.int32); _disc(pred, 5, 28, 28, 15)   # ≈709 px, IoU≈.159 ≥ .1
    assert miss_table(gt, pred)[1] is False
    # 无预测实例 → miss
    assert miss_table(gt, np.zeros((56, 56), np.int32))[1] is True


def test_area_bins():
    assert bin_area(29) == "<30" and bin_area(30) == "30-100" and bin_area(100) == ">=100"
    assert bin_area_native(99) == "<100" and bin_area_native(100) == "100-400"
    assert bin_area_native(400) == ">=400"


def test_did_construction():
    # 手工 4 组概率：截断 .5→.8，完整 .2→.4 ⇒ DiD = .3−.2 = 0.1
    d = did(np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1]),   # Ck 截断 miss 率 .8
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),   # C0 截断 miss 率 .5
            np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),   # Ck 完整 miss 率 .4
            np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))   # C0 完整 miss 率 .2
    assert d == pytest.approx((0.8 - 0.5) - (0.4 - 0.2), abs=1e-12)
    assert d == pytest.approx(0.1, abs=1e-12)


def test_rate_gap_and_bootstrap_stratified():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 2, 40)
    tissue = np.where(img == 0, "Breast", "Lung")
    miss_a = rng.random(40) < 0.5
    miss_b = miss_a.copy()                      # 同核同值 → gap 0，CI 收缩
    r = rate_gap_bootstrap(miss_a, miss_b, img, tissue, n_boot=200, seed=1)
    assert r["gap"] == 0.0 and abs(r["lo"]) < 1e-12 and abs(r["hi"]) < 1e-12
    # 全 1 vs 全 0 → gap 1，CI 覆盖 1
    r2 = rate_gap_bootstrap(np.ones(40), np.zeros(40), img, tissue, n_boot=200, seed=1)
    assert r2["gap"] == 1.0 and r2["lo"] <= 1.0 <= r2["hi"]


def test_bootstrap_stratum_blocks_aligned():
    """M 评审 2026-10-14 的教训（draws 必须镜像点估计）：组织交错且层内值恒定——
    Breast 唯一核 miss 0、Lung 三核 miss 1，b 侧全 0。合并率 gap = 3/4 − 0 = 0.75；
    层内值恒定 ⇒ 任何重抽的 multiset 的合并率不变 → 每次 draw 恰 0.75，CI=[.75,.75]。
    若实现把分层错当加权（或切块错位），lo/hi 将偏离点估计。"""
    tissue = np.array(["Lung", "Breast", "Lung", "Lung"])
    r = rate_gap_bootstrap(np.array([1, 0, 1, 1]), np.array([0, 0, 0, 0]),
                           np.arange(4), tissue, n_boot=100, seed=5)
    assert r["gap"] == pytest.approx(0.75, abs=1e-12)
    assert r["lo"] == pytest.approx(0.75, abs=1e-12) and r["hi"] == pytest.approx(0.75, abs=1e-12)
