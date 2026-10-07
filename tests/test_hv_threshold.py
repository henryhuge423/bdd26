"""HV cutoff controls must change support, not the official geometry or NP labels."""

import numpy as np
import pytest

from nucseg.hovernet.data import HV_PAD, hv_targets
from nucseg.hovernet.official import gen_instance_hv_map


def _labels():
    inst = np.zeros((64, 64), np.int32)
    inst[20:24, 20:24] = 7          # 16 pixels, sparse id
    inst[30:35, 20:26] = 42         # exactly 30 pixels
    inst[:6, :6] = 100             # corner/border object
    inst[50:56, 50:56] = 100        # disconnected duplicate id, remapped upstream
    return inst


def test_cutoff_controls_tiny_support_without_mutating_labels():
    inst = _labels()
    before = inst.copy()
    low = hv_targets(inst, min_size=8)
    old = hv_targets(inst, min_size=30)
    high = hv_targets(inst, min_size=120)
    assert np.any(low[inst == 7])
    assert not np.any(old[inst == 7])
    assert np.any(old[inst == 42])  # inclusive threshold
    assert not np.any(high)
    np.testing.assert_array_equal(low[inst == 42], old[inst == 42])
    np.testing.assert_array_equal(inst, before)
    assert low.dtype == np.float32 and low.shape == (64, 64, 2)


def test_scale_and_cutoff_are_independent_controls():
    inst = _labels()
    up = np.repeat(np.repeat(inst, 2, axis=0), 2, axis=1)
    native = hv_targets(inst, min_size=30)
    original_x2 = hv_targets(up, min_size=30)
    physical_x2 = hv_targets(up, min_size=120)
    assert not np.any(native[inst == 7])
    assert np.any(original_x2[up == 7])
    assert not np.any(physical_x2[up == 7])
    assert np.any(physical_x2[up == 42])


@pytest.mark.parametrize("seed", range(4))
def test_default_targets_are_bit_exact_to_pinned_official(seed):
    inst = _labels()
    rng = np.random.default_rng(seed)
    for i in range(12):
        row, col = rng.integers(0, 56, size=2)
        h, w = rng.integers(2, 9, size=2)
        inst[row:row+h, col:col+w] = 200 + 13*i
    padded = np.pad(inst, HV_PAD)
    expected = gen_instance_hv_map(padded, padded.shape)[HV_PAD:-HV_PAD, HV_PAD:-HV_PAD]
    np.testing.assert_array_equal(hv_targets(inst), expected)
    np.testing.assert_array_equal(hv_targets(inst, min_size=30), expected)


@pytest.mark.parametrize("cutoff", [1, 8, 30, 120])
def test_empty_labels_return_zero_targets(cutoff):
    out = hv_targets(np.zeros((16, 20), np.int32), min_size=cutoff)
    assert out.shape == (16, 20, 2) and not out.any()


@pytest.mark.parametrize("cutoff", [0, -1, 7.5, True, "30", None])
def test_invalid_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError, match="positive integer"):
        hv_targets(_labels(), min_size=cutoff)
