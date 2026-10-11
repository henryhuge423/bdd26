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
