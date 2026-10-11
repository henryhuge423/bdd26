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
