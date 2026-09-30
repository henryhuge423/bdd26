"""Tests for the KongNet instance decode (nucseg.kongnet.decode)."""
import numpy as np
import pytest

from nucseg.kongnet.decode import (ellipse3_kernel, decode_instances, open_close,
                                   per_class_masks, resolve_conflicts)


def two_objects_probs() -> np.ndarray:
    """(64, 64, 18) probs: class 0 (neoplastic) has two blobs, class 3 (dead) one;
    seg channel filled, contour channel a 1-px ring, everything else ~0."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 0.05, (64, 64, 18)).astype(np.float32)
    yy, xx = np.mgrid[:64, :64]
    for (cy, cx, r) in [(20, 20, 8), (44, 44, 6)]:
        ball = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        ring = ball & ~((yy - cy) ** 2 + (xx - cx) ** 2 <= (r - 1) ** 2)
        probs[ball, 0] = 0.9  # head-1 seg channel (ch 0)
        probs[ring, 1] = 0.9  # head-1 contour channel (ch 1)
    ball = (yy - 32) ** 2 + (xx - 10) ** 2 <= 25
    ring = ball & ~((yy - 32) ** 2 + (xx - 10) ** 2 <= 16)
    probs[ball, 9] = 0.8  # dead head (head 4) seg channel (ch 9)
    probs[ring, 10] = 0.8  # dead contour (ch 10)
    return probs


def test_kernel_matches_cv2_ellipse3():
    k = ellipse3_kernel()
    assert k.shape == (3, 3) and k.sum() == 7 and k[1, 1]


def test_open_close_removes_specks_and_fills_holes():
    m = np.zeros((32, 32), bool)
    m[5:15, 5:15] = True
    m[8:10, 8:10] = False  # 2x2 hole (closing with a 3x3 kernel fills it)
    m[0, 30] = True  # 1-px speck
    out = open_close(m, ellipse3_kernel())
    assert out[8:10, 8:10].all()  # hole closed
    assert not out[0, 30]  # speck opened away


def test_per_class_masks_cut_by_contour():
    probs = two_objects_probs()
    masks = per_class_masks(probs, [0, 3, 6, 9, 12], [1, 4, 7, 10, 13], 0.5, 0.3)
    assert masks[0].sum() > 0 and masks[3].sum() > 0
    # contour ring removed from the filled mask: interior survives, ring gone
    yy, xx = np.mgrid[:64, :64]
    ring = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 64) & \
           ~((yy - 20) ** 2 + (xx - 20) ** 2 <= 49)
    assert not (masks[0] & ring).any()


def test_conflict_resolution_goes_to_argmax():
    masks = np.zeros((2, 4, 4), bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 1:3, 1:3] = True  # full overlap
    seg_probs = np.zeros((2, 4, 4), np.float32)
    seg_probs[1, 1:3, 1:3] = 0.9  # class 1 wins
    out = resolve_conflicts(masks, seg_probs)
    assert not out[0][1:3, 1:3].any() and out[1][1:3, 1:3].all()


def test_decode_ids_and_types():
    probs = two_objects_probs()
    inst, typ = decode_instances(probs, [0, 3, 6, 9, 12], [1, 4, 7, 10, 13])
    assert inst.max() == 3 and set(np.unique(typ[inst > 0])) == {1, 4}
    # per-instance type is homogeneous
    for i in range(1, 4):
        assert len(set(typ[inst == i].tolist())) == 1


def test_decode_empty():
    probs = np.zeros((16, 16, 18), np.float32)
    inst, typ = decode_instances(probs, [0, 3, 6, 9, 12], [1, 4, 7, 10, 13])
    assert inst.max() == 0 and typ.max() == 0


@pytest.mark.parametrize("seed", [0, 1])
def test_decode_is_deterministic(seed):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0, 1, (24, 24, 18)).astype(np.float32)
    a = decode_instances(probs, [0, 3, 6, 9, 12], [1, 4, 7, 10, 13])
    b = decode_instances(probs, [0, 3, 6, 9, 12], [1, 4, 7, 10, 13])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
