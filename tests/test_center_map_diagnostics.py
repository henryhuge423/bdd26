"""Behavior checks for diagnostic geometry and passive model instrumentation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1]/'scripts/diagnose_proposal_center_maps.py'
spec = importlib.util.spec_from_file_location('center_diagnostics', path)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.mark.parametrize('y,x',[(0,0),(0,255),(255,0),(255,255),(41,83),(42,84)])
def test_contributors_reconstruct_actual_half_pixel_interpolation(y,x):
    coarse = torch.arange(128*128,dtype=torch.float32).reshape(1,1,128,128)
    actual = F.interpolate(coarse,size=(256,256),mode='bilinear',align_corners=False)[0,0,y,x]
    cells = d.interpolation_contributors(y,x)
    assert sum(cells.values()) == pytest.approx(1)
    assert sum(float(coarse[0,0,cy,cx])*weight for (cy,cx),weight in cells.items()) == pytest.approx(float(actual))


@pytest.mark.parametrize('dy,dx',[(0,1),(0,-1),(1,0),(-1,0)])
def test_translation_does_not_wrap_or_change_retained_rgb(dy,dx):
    image = np.zeros((7,7,3),dtype=np.uint8)
    image[3,3] = [42,66,89]
    shifted = d.shift_image(image,dy,dx)
    assert shifted[3+dy,3+dx].tolist() == [42,66,89]
    assert np.count_nonzero(np.all(shifted == 255,axis=-1)) == 7
    assert np.count_nonzero(np.all(shifted == [42,66,89],axis=-1)) == 1


def test_decoder_trace_preserves_equal_plateaus_threshold_and_cap():
    prob = np.zeros((256,256),dtype=float)
    prob[1::3,1::3] = .5
    prob[100:102,100:102] = .9
    trace,points = d.trace_peaks(prob)
    assert len(points) == 512
    assert len(trace['threshold_ids']) > 512
    assert points[:4] == [(100,100),(100,101),(101,100),(101,101)]
    assert len(trace['local_ids']) >= len(trace['threshold_ids'])


def test_object_localization_is_separate_from_threshold_failure():
    inst = np.zeros((256,256),dtype=np.uint16)
    inst[30:41,30:41] = 7
    gt = [(4,(35,35),True,121)]
    fine = np.full((256,256),-9,dtype=np.float32)
    fine[35,39] = 2
    coarse = np.full((128,128),-9,dtype=np.float32)
    trace,points = d.trace_peaks(torch.from_numpy(fine).sigmoid().numpy())
    rows,pairs = d.inspect_objects(coarse,fine,inst,gt,points,trace,3,'model','development')
    assert pairs == [] and len(rows) == 1
    assert rows[0]['in_mask_displaced_points'] == 1
    assert rows[0]['max_logit_r3'] < 0
    assert rows[0]['mask_max_logit'] == 2
    assert rows[0]['mask_max_distance_px'] == 4


def test_capture_is_passive_and_reads_the_actual_coarse_tensor():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out = torch.nn.Conv2d(2,1,1)
        def forward(self,feats):
            return F.interpolate(self.out(feats[0]),scale_factor=2,mode='bilinear',align_corners=False)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()
        def forward(self,x):
            return self.head([x])
    model = Model().eval()
    x = torch.randn(2,2,5,5)
    expected = model(x)
    capture = d.Capture(model)
    actual = model(x)
    assert torch.equal(expected,actual)
    assert torch.equal(F.interpolate(capture.coarse,scale_factor=2,mode='bilinear',align_corners=False),actual)
    assert torch.equal(capture.features[0],x)
    capture.close()
    assert not model.head._forward_pre_hooks and not model.head.out._forward_hooks


@pytest.mark.parametrize('with_objects', [False, True])
def test_png_delivery_works_without_matplotlib_and_preserves_inputs(tmp_path, monkeypatch, with_objects):
    from PIL import Image
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    image = np.full((256, 256, 3), 120, dtype=np.uint8)
    inst = np.zeros((256, 256), dtype=np.uint16)
    coarse = np.full((128, 128), -20, dtype=np.float32)
    fine = np.full((256, 256), -20, dtype=np.float32)
    coarse[64:, 64:] = 20
    fine[128:, 128:] = 20
    gt = [(4, (35, 35), True, 25), (0, (80, 80), True, 25)] if with_objects else []
    points = [(35, 35), (120, 120)] if with_objects else []
    before = [a.copy() for a in (image, inst, coarse, fine)]
    target = tmp_path/'diagnostic.png'
    d.render_panel(image, inst, gt, points, coarse, fine, target, 'PNG acceptance: fixed synthetic input')
    with Image.open(target) as png:
        png.load()
        assert png.format == 'PNG' and png.mode == 'RGB'
        assert png.size == (2724, 960)
        # Both grid panels retain a shared black/white 0-to-1 scale.
        for origin in (704, 1380):
            assert png.getpixel((origin+30, 204)) == (0, 0, 0)
            assert png.getpixel((origin+600, 774)) == (255, 255, 255)
        if with_objects:
            colors = set(png.getdata())
            assert {(68, 204, 68), (255, 191, 0), (255, 68, 68)} <= colors
    for original, saved in zip((image, inst, coarse, fine), before):
        assert np.array_equal(original, saved)
