import numpy as np
import pytest

from nucseg.candidate_quality.features import (
    FEATURE_NAMES,
    TRAIN_PAIRS,
    design_matrix,
    select_training_rows,
)


def _rec(pairs):
    n = len(pairs)
    return {"pair": np.array(pairs), "area": np.full(n, 120.0),
            "extent": np.full(n, .7), "circularity": np.full(n, .8), "dist": np.full(n, 5.0),
            "pmax": np.full(n, .95), "margin": np.full(n, .2), "border": np.zeros(n, bool),
            "class": np.zeros(n, np.int64), "image": np.arange(n), "id": np.ones(n, np.int64),
            "matched": np.array([True, False] * ((n + 1) // 2))[:n], "typed": np.ones(n, bool),
            "gt_class": np.zeros(n, np.int64), "gt_already_by_base": np.zeros(n, bool),
            "added": np.zeros(n, bool), "iou_best": np.zeros(n)}


def test_select_training_rows_filters_and_keeps_order():
    rec = _rec(["split2_seed19", "split1_seed1", "split3_seed19", "split2_seed1"])
    sub = select_training_rows(rec)
    assert list(sub["pair"]) == ["split2_seed19", "split2_seed1"]
    assert len(sub["area"]) == 2


def test_design_matrix_shape_and_columns():
    rec = select_training_rows(_rec(list(TRAIN_PAIRS)))
    X, y = design_matrix(rec)
    assert X.shape == (3, 12) and y.dtype == bool and list(y) == [True, False, True]
    assert FEATURE_NAMES[0] == "log_area" and np.allclose(X[:, 0], np.log1p(120.0))
    assert (X[:, 6] == 0).all() and (X[:, 7] == 1).all()  # border, class0 one-hot


def test_design_matrix_rejects_nonfinite():
    rec = select_training_rows(_rec(list(TRAIN_PAIRS)))
    rec["pmax"] = rec["pmax"].copy(); rec["pmax"][1] = np.nan
    with pytest.raises(ValueError, match="pmax"):
        design_matrix(rec)


def test_design_matrix_rejects_class_out_of_range():
    rec = select_training_rows(_rec(list(TRAIN_PAIRS)))
    rec["class"] = rec["class"].copy(); rec["class"][0] = 5
    with pytest.raises(ValueError, match="class"):
        design_matrix(rec)


# ---------------- support (Task 2) ----------------
from pathlib import Path

from nucseg.candidate_quality.support import (
    effective_denominator,
    jury_paths_fold1,
    support_matrix,
)


def _save_npz(path, inst):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez_compressed(fh, inst=inst[None], type=(inst > 0).astype(np.uint8)[None])


@pytest.fixture()
def support_world(tmp_path):
    root = tmp_path / "round"
    # 源 x2 预测：split2_seed19 的 inst 里有候选 1（4x4 @ (2,2)），
    # split2_seed1 的 inst 里有候选 2（4x4 @ (9,9)）
    s19 = np.zeros((16, 16), np.int32); s19[2:6, 2:6] = 1
    s1 = np.zeros((16, 16), np.int32); s1[9:13, 9:13] = 2
    _save_npz(root / "p0/split2_seed19/val/x2/identity/pred.npz", s19)
    _save_npz(root / "p0/split2_seed1/val/x2/identity/pred.npz", s1)
    _save_npz(root / "p0/split2_seed2/val/x2/identity/pred.npz", np.zeros((16, 16), np.int32))
    # jury 三席：s19-x2（与候选 1 完全重合）、s1-x2（与候选 2 完全重合）、
    # s19-base（与候选 1 的 IoU = 8/16 = 0.5）
    a = np.zeros((16, 16), np.int32); a[2:6, 2:6] = 9
    b = np.zeros((16, 16), np.int32); b[9:13, 9:13] = 9
    c = np.zeros((16, 16), np.int32); c[2:6, 2:4] = 9
    _save_npz(tmp_path / "seatA.npz", a); _save_npz(tmp_path / "seatB.npz", b)
    _save_npz(tmp_path / "seatC.npz", c)
    jury = [{"name": "s19_x2", "path": str(tmp_path / "seatA.npz"), "source_pair": "split2_seed19"},
            {"name": "s1_x2", "path": str(tmp_path / "seatB.npz"), "source_pair": "split2_seed1"},
            {"name": "s19_base", "path": str(tmp_path / "seatC.npz"), "source_pair": None}]
    rec = {"pair": np.array(["split2_seed19", "split2_seed1"]),
           "image": np.array([0, 0]), "id": np.array([1, 2])}
    return root, rec, jury


def test_support_matrix_source_exclusion_and_thr(support_world):
    root, rec, jury = support_world
    mat, excl = support_matrix(rec, root, jury, thr=0.3)
    assert mat.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert excl.tolist() == [[True, False, False], [False, True, False]]
    assert effective_denominator(excl).tolist() == [2, 2]


def test_jury_paths_fold1_enumeration(tmp_path):
    jury = jury_paths_fold1(tmp_path, None)
    assert [j["name"] for j in jury] == ["cellvit_s2_e19_x1", "cellvit_s2_e19_x2",
                                         "cellvit_s2_e1_x1", "cellvit_s2_e1_x2",
                                         "cellvit_s2_e2_x1", "cellvit_s2_e2_x2"]
    assert jury[1]["source_pair"] == "split2_seed19" and jury[0]["source_pair"] is None
    assert all(j["path"].startswith(str(tmp_path)) for j in jury)
    with_hn = jury_paths_fold1(tmp_path, tmp_path / "hn.npz")
    assert len(with_hn) == 7 and with_hn[-1]["name"] == "hovernet_s2"


# ---------------- model (Task 3) ----------------
from nucseg.candidate_quality.model import fit_logistic, predict, reliability, torch_head_directions


def test_fit_predict_separable():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.r_[rng.normal(0, .5, n // 2), rng.normal(2, .5, n // 2)],
                         rng.normal(0, .5, n)])
    y = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    groups = np.arange(n)  # 每实例一图（分组 CV 退化为普通 K 折）
    fit = fit_logistic(X, y, groups)
    assert fit["C"] in (0.1, 1.0, 10.0) and np.isfinite(fit["cv_auc"]) and fit["cv_auc"] > 0.9
    p = predict(fit, X)
    assert p.shape == (n,) and (p >= 0).all() and (p <= 1).all()
    assert ((p > .5) == y).mean() > 0.9


def test_reliability_bins_sum():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    bins = reliability(rng.uniform(0, 1, 500) < p, p, bins=10)
    assert sum(b["n"] for b in bins) == 500
    assert all(b["mean_p"] == b["mean_p"] for b in bins)  # finite


def test_torch_head_directions_agree():
    rng = np.random.default_rng(2)
    n = 150
    X = np.column_stack([np.r_[rng.normal(0, .5, n // 2), rng.normal(2, .5, n // 2)]])
    y = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    Xa = np.linspace(-1, 3, 40)[:, None]
    outs = torch_head_directions(X, y, Xa, seeds=(0, 1))
    assert len(outs) == 2 and all(o.shape == (40,) for o in outs)
    from scipy.stats import spearmanr
    assert spearmanr(outs[0], outs[1]).statistic > 0.8


# ---------------- menu (Task 4) ----------------
from nucseg.candidate_quality.menu import a1_allowed, build_menu, selection_rows


def _apply_rec():
    return {"pair": np.array(["split1_seed1"] * 5), "image": np.array([0, 0, 1, 1, 2]),
            "id": np.array([1, 2, 3, 4, 5]), "area": np.array([20, 50, 150, 250, 150], float),
            "pmax": np.array([.95, .85, .95, .95, .95]), "border": np.array([False, False, False, False, True]),
            "matched": np.array([True, False, True, False, True])}


def test_a1_allowed_semantics():
    rec = _apply_rec()
    cfg = {"max_area": 200, "min_prob": 0.9, "interior_only": True}
    allowed, n = a1_allowed(rec, cfg, ef_floor=30)
    assert allowed == {(1, 3)} and n == 1          # 唯一同时过 cap/prob/floor/interior 的行
    allowed2, n2 = a1_allowed(rec, {"max_area": 200, "min_prob": 0.9, "interior_only": False})
    assert allowed2 == {(1, 3), (2, 5)} and n2 == 2  # interior0 时第 5 行也过


def test_build_menu_rows_and_closed_threshold():
    rec = _apply_rec()
    p_low = np.full(5, 0.3)   # 恰在阈值上：闭区间应保留全部
    p_sup = np.array([.1, .2, .3, .4, .5])
    rows = build_menu(rec, p_low, p_sup)
    assert [r["name"] for r in rows[:2]] == ["identity", "a1_incumbent"]
    by_name = {r["name"]: r for r in rows}
    assert len(rows) == 18
    assert set(by_name) == {"identity", "a1_incumbent"} \
        | {f"a2_f{f}_c{c}" for f in (0, 30) for c in (100, 200, 400)} \
        | {f"a3_p{t}" for t in (.3, .4, .5, .6, .7)} \
        | {f"a4_p{t}" for t in (.3, .4, .5, .6, .7)}
    assert by_name["a3_p0.3"]["allowed"] == {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)}
    assert by_name["a4_p0.4"]["allowed"] == {(1, 4), (2, 5)}
    assert by_name["a2_f30_c100"]["allowed"] == {(0, 2)}  # 仅 area 50 同时落在 [30,100]
    assert by_name["a2_f0_c400"]["allowed"] == {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)}


def test_selection_rows_reuses_select_fusion_semantics():
    def row(name, arm, mpq, bpq, sd, im, ad):
        return {"name": name, "arm": arm, "mPQ": mpq, "bPQ": bpq, "strict_dead": sd,
                "interior_matched": im, "added": ad, "Dead_PQ": 0.0, "unmatched": 0}
    rows = [row("identity", "A0", .4835, .6516, .1345, 499, 0),
            row("a2_f0_c100", "A2", .4840, .6510, .1346, 500, 900),
            row("a2_f30_c200", "A2", .4830, .6490, .1330, 498, 700),   # bPQ/strict gate 失败
            row("a3_p0.5", "A3", .4850, .6520, .1350, 505, 600),
            row("a4_p0.5", "A4", .4820, .6500, .1330, 497, 400)]       # strict gate 失败
    sel = selection_rows(rows)
    assert sel["A2"]["name"] == "a2_f0_c100"
    assert sel["A3"]["name"] == "a3_p0.5"
    assert sel["A4"]["name"] == "identity"   # A4 全部被 gate 拒 → 回落 identity
