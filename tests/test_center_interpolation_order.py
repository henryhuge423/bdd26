"""Behavior checks for the fixed interpolation-order counterfactual."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1]/'scripts/diagnose_center_interpolation_order.py'
spec = importlib.util.spec_from_file_location('interpolation_order', path)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.mark.parametrize('value', [-8, 0, 8])
def test_constant_grid_has_identical_probability(value):
    coarse = np.full((128, 128), value, dtype=np.float32)
    actual = d.interpolate_scores(coarse)
    expected = float(torch.tensor(float(value)).sigmoid())
    assert np.allclose(actual, expected, rtol=0, atol=1e-7)


def test_isolated_peak_crosses_threshold_under_the_predicted_order():
    background, peak = .01, .93
    coarse = np.full((128, 128), np.log(background/(1-background)), dtype=np.float32)
    coarse[64, 64] = np.log(peak/(1-peak))
    tensor = torch.from_numpy(coarse)[None, None]
    fine = F.interpolate(tensor, scale_factor=2, mode='bilinear', align_corners=False)[0, 0].numpy()
    original, alternative = d.original_scores(fine), d.interpolate_scores(coarse)
    assert original.max() < .5 < alternative.max()
    assert alternative.max() == pytest.approx((9*peak+7*background)/16, abs=1e-7)
    # Preserve the actual four-point plateau and original unique matching rule.
    _, points = d.d.trace_peaks(alternative)
    assert points == [(128, 128), (128, 129), (129, 128), (129, 129)]
    score = d.d.score_points(points, [(4, (128, 128), True, 20)])
    assert score['matched'] == 1 and score['candidates'] == 4


def test_alternative_interpolation_preserves_inputs_and_edge_clamping():
    coarse = np.full((128, 128), -4, dtype=np.float32)
    coarse[0, 0] = 2
    before = coarse.copy()
    actual = d.interpolate_scores(coarse)
    assert np.array_equal(coarse, before)
    assert actual.shape == (256, 256)
    assert actual[0, 0] == float(torch.tensor(2.).sigmoid())
    assert np.isfinite(actual).all() and 0 <= actual.min() <= actual.max() <= 1


def test_point_provenance_keeps_assignment_and_native_mask_membership():
    inst = np.zeros((256, 256), dtype=np.uint16)
    inst[30:40, 30:40] = 7
    points = [(35, 35), (60, 60)]
    rows = d.point_records(points, [(4, (35, 35), True, 100)], inst,
                           np.full((256, 256), .7), 178, 'alternative')
    assert rows[0]['assigned_gt'] == 0 and rows[0]['mask_instance_id'] == 7
    assert rows[1]['assigned_gt'] is None and rows[1]['mask_instance_id'] == 0
