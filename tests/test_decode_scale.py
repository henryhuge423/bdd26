"""u-scaled decode constants (docs/findings.md 2026-09-29 pre-registered x2 re-decode): u=1 must stay
bit-exact vs the official decode; u>1 scales remove_small_objects min_size by area (10*u^2) and the
Sobel ksize by length (odd(21*u))."""

import cv2
import numpy as np

from nucseg.hovernet.data import hv_targets
from nucseg.hovernet.engine import NR_TYPES, _post
from nucseg.hovernet.official import post_process
from nucseg.postproc.recovery import decode_pred_map, scaled_ksize, scaled_min_size


def _synthetic_pred_map(seed=0):
    """Plausible network outputs (crisp fg so blob areas are exact) with a 6x6 = 36 px nucleus that
    survives the official min_size=10 but not the u=2 min_size=10*2^2=40."""
    rng = np.random.default_rng(seed)
    inst = np.zeros((256, 256), np.int32)
    cv2.circle(inst, (70, 60), 12, 1, -1)      # ~452 px
    inst[200:206, 150:156] = 2                 # 36 px
    fg = (inst > 0).astype(np.float32) + rng.normal(0, 0.01, (256, 256)).astype(np.float32)
    hv = hv_targets(inst) + rng.normal(0, 0.02, (256, 256, 2)).astype(np.float32)
    tp = np.full((256, 256, 6), 0.2 / 6, np.float32)
    tp[inst > 0, 2] = 0.9
    tp = tp / tp.sum(-1, keepdims=True)
    return np.concatenate([tp.argmax(-1)[..., None].astype(np.float32), fg[..., None], hv], -1)


def test_scaled_constants():
    assert scaled_min_size(1) == 10 and scaled_min_size(2) == 40 and scaled_min_size(3) == 90
    assert scaled_ksize(1) == 21 and scaled_ksize(2) == 41 and scaled_ksize(3) == 63


def test_u1_bit_exact_vs_official():
    m = _synthetic_pred_map()
    ref = post_process(m, nr_types=NR_TYPES)
    out = decode_pred_map(m, nr_types=NR_TYPES, u=1.0)
    assert np.array_equal(out[0], ref[0]) and out[1].keys() == ref[1].keys()
    assert np.array_equal(_post((m, 1.0))[0], ref[0])   # _post accepts both (map,) and (map, u)


def test_u2_drops_small_nucleus():
    m = _synthetic_pred_map()
    inst1, _ = decode_pred_map(m, nr_types=NR_TYPES, u=1.0)
    inst2, _ = decode_pred_map(m, nr_types=NR_TYPES, u=2.0)
    assert (np.unique(inst1) > 0).sum() == 2
    assert (np.unique(inst2) > 0).sum() == 1
    assert inst2[200:206, 150:156].max() == 0           # 36 px nucleus: removed at u=2
    assert (inst2 > 0).sum() > 400                      # big nucleus kept


def test_rescaled_sobel_kernels_reproduce_official_at_21():
    import cv2
    from nucseg.postproc.recovery import _rescaled_sobel_kernels
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 64))
    kx, ky = _rescaled_sobel_kernels(21)                # 21 -> 21 resample is the identity
    r1 = cv2.sepFilter2D(a, cv2.CV_64F, kx, ky)
    r2 = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=21)
    assert np.allclose(r1 / np.abs(r1).max(), r2 / np.abs(r2).max(), atol=1e-8)


def test_module_global_restored():
    m = _synthetic_pred_map(1)
    decode_pred_map(m, nr_types=NR_TYPES, u=2.0)
    ref = post_process(m, nr_types=NR_TYPES)
    assert np.array_equal(_post((m,))[0], ref[0])       # official decode back in place afterwards
