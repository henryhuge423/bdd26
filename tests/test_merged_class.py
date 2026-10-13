"""M merged-class diagnostic tests: identity binding, merge vs naive official eval, invariants."""
import numpy as np
import pytest

from nucseg.constants import NUM_CLASSES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.merged_class import (bootstrap_delta, class_map,
                                         mpq, mpq_strict, per_image_class_pq)
from nucseg.metrics.pannuke_eval import instance_classes
from nucseg.text.retype import paint_types

N = 24  # fold1 前 24 图，覆盖多组织


@pytest.fixture(scope="module")
def world():
    f = PanNukeFold(1)
    if not (f.dir / "inst.npy").exists():
        pytest.skip("PanNuke fold1 not prepared")
    rng = np.random.default_rng(0)
    inst, typ = [], []
    for i in range(N):
        lab = np.asarray(f.inst[i]).copy()
        t = np.asarray(f.type[i]).copy()
        # 随机把 ~1/6 实例整体改类 + ~1/10 抹成 untyped，制造跨类错分与 cls=0
        ids = np.unique(lab); ids = ids[ids > 0]
        for g in ids:
            m = lab == g
            if rng.random() < 1 / 6:
                t[m] = rng.integers(1, NUM_CLASSES + 1)
            elif rng.random() < 0.1:
                t[m] = 0
        inst.append(lab); typ.append(t)
    inst, typ = np.stack(inst), np.stack(typ)
    gt = np.asarray(f.gt_channels[:N])
    tissue = np.asarray(f.tissue[:N])
    rows = [(j, i) for j in range(N) for i in np.unique(inst[j]) if i]
    ii = np.array([r[0] for r in rows]); iid = np.array([r[1] for r in rows])
    ev = RetypeEvaluator(gt, inst, ii, iid, tissue, workers=4)
    cls = np.concatenate([instance_classes(inst[j], typ[j])[1] for j in range(N)])
    return f, ev, inst, typ, ii, iid, cls, gt, tissue


def _merged_gt_channels(gt, cmap):
    """官方语义的合并 GT：(n,H,W,C)。合并通道用奇偶 id 分离两原通道
    （通道内 id 1..n 会撞号；像素集不相交，overlap() 反正会 relabel）。"""
    c = int(cmap.max())
    out = np.zeros(gt.shape[:3] + (c,), np.int64)
    pairs: dict[int, list[int]] = {}
    for k in range(1, NUM_CLASSES + 1):
        pairs.setdefault(int(cmap[k]) - 1, []).append(k)
    for kk, ks in pairs.items():
        if len(ks) == 1:
            out[..., kk] = gt[..., ks[0] - 1]
        else:
            a, b = ks  # 任意多对（本实验固定单对合并）
            out[..., kk] = 2 * gt[..., a - 1].astype(np.int64) \
                + np.where(gt[..., b - 1] > 0, 2 * gt[..., b - 1] + 1, 0).astype(np.int64)
    return out


def _naive_pq(tp, fp, fn, siou):
    d = tp + 0.5 * fp + 0.5 * fn
    return np.nan if d == 0 else (tp / d) * (siou / (tp + 1e-6))


def _naive_merged_eval(gt, inst, typ, ii, iid, cls, cmap, tissue):
    """从官方原语（overlap/pq_from_overlap/instance_classes）逐类重算的合并空间
    official/strict per-class PQ —— 独立于 fast_retype 与 merged_class 两条快速路径。"""
    from nucseg.metrics.instance import overlap, pq_from_overlap
    from nucseg.text.retype import paint_types

    C = int(cmap.max())
    painted = paint_types(inst, ii, iid, cmap[cls])
    n = len(tissue)
    pq_tab = np.full((n, C), np.nan)
    strict_tab = pq_tab.copy()
    for j in range(n):
        p_lab, p_cls = instance_classes(inst[j], painted[j])
        mgt = _merged_gt_channels(gt[j:j + 1], cmap)[0]
        for k in range(C):
            pred = np.where(np.isin(p_lab, np.nonzero(p_cls == k + 1)[0] + 1), p_lab, 0)
            ov = overlap(mgt[..., k], pred)
            sc = pq_from_overlap(ov)
            if ov.nt == 0:  # GT 缺类：official NaN；strict 仅在有预测实例时记 0
                strict_tab[j, k] = 0.0 if ov.np_ > 0 else np.nan
                continue
            pq_tab[j, k] = strict_tab[j, k] = _naive_pq(sc.tp, sc.fp, sc.fn, sc.sum_iou)

    def tissue_avg(v):
        per = [np.nanmean(v[tissue == t]) for t in dict.fromkeys(tissue)]
        return float(np.nanmean(per))

    with np.errstate(all="ignore"):
        img, img_s = np.nanmean(pq_tab, 1), np.nanmean(strict_tab, 1)
    return tissue_avg(img), np.nanmean(pq_tab, 0), tissue_avg(img_s), np.nanmean(strict_tab, 0)


def test_class_map():
    assert list(class_map(None)) == [0, 1, 2, 3, 4, 5]
    assert list(class_map((2, 3))) == [0, 1, 2, 2, 3, 4]
    assert list(class_map((1, 5))) == [0, 1, 2, 3, 4, 1]
    for a in range(1, 6):
        for b in range(a + 1, 6):
            m = class_map((a, b))
            assert m[0] == 0 and sorted(set(m[1:])) == [1, 2, 3, 4] and m[a] == m[b]


def test_identity_map_bitexact(world):
    _, ev, _, _, _, _, cls, _, _ = world
    ident = class_map(None)
    m_m, per_m = mpq(ev, cls, ident)
    m_r, per_r = ev.mpq(cls)
    assert m_m == m_r and np.array_equal(per_m, per_r, equal_nan=True)
    s_m, sper_m = mpq_strict(ev, cls, ident)
    s_r, sper_r = ev.mpq_strict(cls)
    assert s_m == s_r and np.array_equal(sper_m, sper_r, equal_nan=True)


def test_merge_matches_naive_official_eval(world):
    f, ev, inst, typ, ii, iid, cls, gt, tissue = world
    for pair in [(2, 3), (1, 5), (4, 5)]:
        cmap = class_map(pair)
        m, per = mpq(ev, cls, cmap)
        s, sper = mpq_strict(ev, cls, cmap)
        nm, nper, ns, nsper = _naive_merged_eval(gt, inst, typ, ii, iid, cls, cmap, tissue)
        assert m == pytest.approx(nm, abs=1e-10)
        assert s == pytest.approx(ns, abs=1e-10)
        assert per == pytest.approx(nper, abs=1e-10, nan_ok=True)
        assert sper == pytest.approx(nsper, abs=1e-10, nan_ok=True)


def test_nan_and_strict_conventions(world):
    _, ev, _, _, _, _, cls, gt, tissue = world
    cmap = class_map((2, 3))
    pq = per_image_class_pq(ev, cls, cmap)
    # 合并类 2 的 GT 计数 = 原 2+3 通道计数和；GT 缺 → NaN（official），strict 只在"有预测"时记 0
    n_gt_m = ev.n_gt[:, 1] + ev.n_gt[:, 2]
    assert np.isnan(pq[n_gt_m == 0, 1]).all()
    strict = mpq_strict(ev, cls, cmap)
    assert strict[0] <= mpq(ev, cls, cmap)[0]


def test_untyped_excluded(world):
    _, ev, _, _, _, _, cls, gt, tissue = world
    cmap = class_map((2, 3))
    pq0 = per_image_class_pq(ev, cls, cmap)
    cls0 = cls.copy(); cls0[:] = 0  # 全部 untyped：任何合并类无 TP 且无预测计入
    pq1 = per_image_class_pq(ev, cls0, cmap)
    assert pq1.shape == (ev.n, 4)
    assert np.isnan(pq1).all() or (pq1[~np.isnan(pq1)] == 0).all()
    # 非合并类（Neo/Dead/Epi）列与恒等空间同名列逐位一致
    ident = per_image_class_pq(ev, cls0, class_map(None))
    assert np.array_equal(pq1[:, 0], ident[:, 0], equal_nan=True)   # Neo
    assert np.array_equal(pq1[:, 2], ident[:, 3], equal_nan=True)   # Dead（cmap[4]=3）
    assert np.array_equal(pq1[:, 3], ident[:, 4], equal_nan=True)   # Epi
    # 合并类列：原 Inf 或 Conn 任一有 GT 即 0（否则 NaN）
    n_23 = ev.n_gt[:, 1] + ev.n_gt[:, 2]
    assert np.array_equal(np.isnan(pq1[:, 1]), n_23 == 0)


def test_bootstrap_delta_same_map_is_zero(world):
    _, ev, _, _, _, _, cls, _, _ = world
    cmap = class_map((2, 3))
    d = bootstrap_delta(ev, cls, cmap, cmap, n_boot=50, seed=1)
    assert d["delta"] == 0.0 and abs(d["lo"]) < 1e-12 and abs(d["hi"]) < 1e-12


def test_bootstrap_delta_point_estimate_consistent(world):
    _, ev, _, _, _, _, cls, _, _ = world
    d = bootstrap_delta(ev, cls, class_map(None), class_map((2, 3)), n_boot=50, seed=2)
    m0 = mpq(ev, cls, class_map(None))[0]
    m23 = mpq(ev, cls, class_map((2, 3)))[0]
    assert d["delta"] == pytest.approx(m23 - m0, abs=1e-12)
    assert d["lo"] <= d["delta"] <= d["hi"]


def test_bootstrap_delta_two_assignments(world):
    # 同一合并空间下两个类分配的差（oracle-under-merge 干预差）：delta == mPQ(b)-mPQ(a)
    _, ev, _, _, _, _, cls, _, _ = world
    cmap = class_map((2, 3))
    oracle = cls.copy(); oracle[ev.m_row] = ev.m_cls
    d = bootstrap_delta(ev, cls, cmap, cmap, n_boot=50, seed=3, cls_b=oracle)
    assert d["delta"] == pytest.approx(mpq(ev, oracle, cmap)[0] - mpq(ev, cls, cmap)[0],
                                       abs=1e-12)
    assert d["lo"] <= d["delta"] <= d["hi"]


def test_bootstrap_delta_strict_mode(world):
    # strict=True：点估计 == strict mPQ 差，且与本世界的 official 差不同
    _, ev, _, _, _, _, cls, _, _ = world
    ident, cmap = class_map(None), class_map((2, 3))
    d = bootstrap_delta(ev, cls, ident, cmap, n_boot=50, seed=4, strict=True)
    assert d["delta"] == pytest.approx(
        mpq_strict(ev, cls, cmap)[0] - mpq_strict(ev, cls, ident)[0], abs=1e-12)
    d_off = bootstrap_delta(ev, cls, ident, cmap, n_boot=50, seed=4)
    assert abs(d["delta"] - d_off["delta"]) > 1e-6


class _EvStub:
    """最小 evaluator 桩：bootstrap_delta 只读 n / tissue（经 per_image_class_pq 桩）。"""

    n = 4
    tissue = np.array(["Lung", "Breast", "Lung", "Lung"])  # 交错且层大小不等（1 vs 3）
    m_img = np.array([0, 1, 2, 3])
    m_cls = np.ones(4, np.int64)
    m_row = np.arange(4)
    m_iou = np.ones(4)
    n_gt = np.ones((4, 5), np.int64)
    inst_img = np.arange(4)


def test_bootstrap_delta_stratum_blocks_aligned():
    # C1 回归（评审 2026-10-14）：重抽样必须按层内抽取的 draw 求层均值，再用原层位置
    # 切重排后的向量（组织交错时混层，CI 系统性失准）。
    # 构造：image1（层 A，唯一成员）delta 0；images 0/2/3（层 B）delta -1。
    # 点估计 = (0 + (-1)) / 2 = -0.5；正确 bootstrap 每次 draw 都恰为 -0.5 → CI=[-.5,-.5]。
    ev = _EvStub()

    import nucseg.metrics.merged_class as mc
    orig = mc.per_image_class_pq

    def fake_pq(ev_, cls_, cmap_):
        # pb 侧（cmap_b）：image1 PQ=1（delta 0），其余 0（delta -1）；pa 侧恒 1
        v = np.where(np.arange(ev_.n) == 1, 1.0, 0.0)
        if cmap_ is IDENT_A:
            v = np.ones(ev_.n)
        return np.repeat(v[:, None], cmap_.max(), 1)

    IDENT_A = np.array([0, 1, 2, 3, 4, 5])
    IDENT_B = np.array([0, 9, 2, 3, 4, 5])  # 任意非恒等映射，仅作侧别标记
    mc.per_image_class_pq = fake_pq
    try:
        d = mc.bootstrap_delta(ev, np.ones(4), IDENT_A, IDENT_B, n_boot=100, seed=5)
    finally:
        mc.per_image_class_pq = orig
    assert d["delta"] == pytest.approx(-0.5, abs=1e-12)
    assert d["lo"] == pytest.approx(-0.5, abs=1e-12) and d["hi"] == pytest.approx(-0.5, abs=1e-12)
