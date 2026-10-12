import numpy as np
import pytest

from nucseg.e1_typing import (
    clustered_rate_bootstrap,
    jury_votes,
    majority_type,
    pair_base,
)


def _rect(inst, i, y, x, h, w):
    inst[y:y + h, x:x + w] = i


def test_pair_base_ambiguous_excluded():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6)
    base = np.zeros((16, 16), np.int32); _rect(base, 5, 2, 2, 6, 3); _rect(base, 7, 2, 5, 6, 3)
    out = pair_base(gt, base)
    assert out[1] is None  # IoU 并列最大 → 歧义排除


def test_pair_base_plain():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6); _rect(gt, 2, 9, 9, 4, 4)
    base = np.zeros((16, 16), np.int32); _rect(base, 5, 2, 2, 6, 6); _rect(base, 6, 10, 10, 2, 2)
    out = pair_base(gt, base)
    assert out[1] == (5, 1.0) and out[2] is None  # GT2 vs base6 IoU=4/16<0.5


def test_jury_vote_threshold_and_best_instance():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6)
    m = np.zeros((16, 16), np.int32); _rect(m, 3, 2, 2, 6, 6)
    mt = np.zeros((16, 16), np.uint8); mt[m == 3] = 2
    assert jury_votes(gt, m, mt)[1] == (3, 2)
    assert jury_votes(gt, m, mt, thr=0.95)[1] == (3, 2)
    far = np.zeros((16, 16), np.int32); _rect(far, 4, 10, 10, 4, 4)
    ft = np.zeros((16, 16), np.uint8); ft[far == 4] = 1
    assert jury_votes(gt, far, ft)[1] is None  # IoU=0 < 0.3


def test_majority_clauses():
    folds = {0: 1, 1: 1, 2: 3, 3: 3, 4: 3}
    maj, d = majority_type({0: 2, 1: 2, 2: 2, 3: 1}, folds)
    assert maj == 2 and d["n_votes"] == 4 and d["both_folds_ok"]
    assert majority_type({0: 2, 1: 2, 3: 1}, folds)[0] is None        # 两个训练折缺一
    assert majority_type({0: 2, 1: 2, 2: 1, 3: 1}, folds)[0] is None  # 平票
    assert majority_type({0: 2, 1: 2}, folds)[0] is None              # 总票<3


def test_bootstrap_clusters_by_image():
    ind = np.array([1.0, 0.0, 1.0, 1.0])
    img = np.array([0, 0, 1, 1])
    lo, hi = clustered_rate_bootstrap(ind, img, n_boot=500, seed=7)
    assert 0.0 <= lo <= hi <= 1.0
    lo2, hi2 = clustered_rate_bootstrap(np.ones(4), img, n_boot=500, seed=7)
    assert lo2 == hi2 == 1.0


def test_jury_vote_majority_semantics():
    """canonical instance_classes semantics: pixel-majority class, 0 when untyped."""
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6)
    m = np.zeros((16, 16), np.int32); _rect(m, 3, 2, 2, 6, 6)
    mt = np.zeros((16, 16), np.uint8)
    mt[2:8, 2:5] = 2; mt[2:8, 5:8] = 1          # 18 像素 type2 vs 18 像素 type1 → 平票取 argmax=2? 见断言
    assert jury_votes(gt, m, mt)[1][1] in (1, 2)
    mt2 = np.zeros((16, 16), np.uint8); mt2[2:8, 2:5] = 2; mt2[2:8, 5:8] = 1
    mt2[2:8, 2:4] = 0                            # type2 被清零 12px → 剩 6px，type1 18px 占多数
    assert jury_votes(gt, m, mt2)[1][1] == 1
    mt3 = np.zeros((16, 16), np.uint8)           # 全 0 → class 0
    assert jury_votes(gt, m, mt3)[1] == (3, 0)
