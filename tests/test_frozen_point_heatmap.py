import importlib.util
from pathlib import Path

import numpy as np
import torch


def api():
    candidates = [Path(__file__).with_name("probe_frozen_point_heatmap.py"),
                  Path(__file__).resolve().parents[1] / "scripts" / "probe_frozen_point_heatmap.py"]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("p1probe", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m

    class Missing:
        pass

    return Missing()


def test_gt_objects_centroid_tie_row_major_majority_and_interior():
    m = api()
    inst = np.zeros((6, 6), dtype=np.uint16)
    inst[1:3, 1:3] = 7          # interior square
    inst[0, 4:6] = 9            # touches top ring
    typ = np.zeros((6, 6), dtype=np.uint8)
    typ[1, 1] = 1
    typ[1, 2] = typ[2, 1] = typ[2, 2] = 4
    typ[0, 4] = 2
    typ[0, 5] = 3
    out = m.gt_objects(inst, typ)
    assert set(out) == {7, 9}
    cls7, pt7, int7 = out[7]
    assert cls7 == 4 and int7 and pt7 == (1, 1)  # centroid-equidistant tie, row-major first
    cls9, pt9, int9 = out[9]
    assert cls9 == 2 and not int9  # touches outer ring


def test_gaussian_target_exact_one_and_edge_clamp():
    m = api()
    t = m.gaussian_target([(5, 5)], 16, 16)
    assert t[5, 5] == 1.0
    assert t[5, 8] < 1.0 and t[0, 0] == 0.0
    t2 = m.gaussian_target([(0, 0)], 16, 16)
    assert t2[0, 0] == 1.0 and t2[3, 3] < 1.0
    overlap = m.gaussian_target([(5, 5), (5, 6)], 16, 16)
    assert overlap[5, 5] == 1.0 and overlap[5, 6] == 1.0


def test_focal_center_loss_confident_is_near_zero_and_grad_finite():
    m = api()
    logits = torch.full((1, 8, 8), -20.0, requires_grad=True)
    target = torch.zeros(1, 8, 8)
    target[0, 4, 4] = 1.0
    with torch.no_grad():
        logits[0, 4, 4] = 20.0
    loss = m.focal_center_loss(logits, target, 1)
    assert float(loss) < 1e-3
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())
    bad = m.focal_center_loss(-logits.detach().requires_grad_(True), target, 1)
    assert float(bad) > float(loss)


def test_decode_peaks_order_plateau_and_cap():
    m = api()
    p = np.zeros((8, 8))
    p[2, 3] = 0.9
    p[6, 6] = 0.7
    pts = m.decode_peaks(p, threshold=0.5, cap=512)
    assert pts == [(2, 3), (6, 6)]
    p2 = np.zeros((4, 4))
    p2[1, 1] = p2[1, 2] = 0.8  # equal plateau, row-major stable
    assert m.decode_peaks(p2, threshold=0.5)[0] == (1, 1)
    p3 = np.zeros((4, 4)) + 0.4  # below threshold
    assert m.decode_peaks(p3, threshold=0.5) == []
    p4 = np.zeros((16, 16))
    for k, (yy, xx) in enumerate([(1, 1), (1, 5), (1, 9), (5, 1), (5, 5), (9, 9)]):
        p4[yy, xx] = 0.9 - k * 0.01
    assert len(m.decode_peaks(p4, threshold=0.5, cap=5)) == 5
    assert len(m.decode_peaks(p4, threshold=0.5, cap=512)) == 6


def test_decode_nms_suppresses_radius_and_is_deterministic():
    m = api()
    p = np.zeros((8, 8))
    p[2, 2] = 0.9
    p[2, 3] = 0.8  # within 3x3 of first -> suppressed
    p[6, 6] = 0.7
    pts = m.decode_nms(p, k=10)
    assert pts == [(2, 2), (6, 6)]
    assert m.decode_nms(p, k=10) == pts
    p2 = np.full((4, 4), 0.2)
    assert len(m.decode_nms(p2, k=3)) == 3


def test_hits_radius_boundary_inclusive():
    m = api()
    gt = {1: (1, (5, 5), True)}
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 5] = 1
    hits, idead, inside, n_all, n_idead = m.hits_for_points([(5, 8)], gt, mask)
    assert (hits, idead, inside, n_all, n_idead) == (1, 0, 0, 1, 0)  # distance 3 counts; class 1 is not Dead
    hits, *_ = m.hits_for_points([(5, 9)], gt, mask)
    assert hits == 0  # distance 4 misses
    hits, _, inside, *_ = m.hits_for_points([(5, 5)], gt, mask)
    assert hits == 1 and inside == 1


def test_idead_excludes_border_dead():
    m = api()
    gt = {1: (4, (3, 3), False), 2: (4, (8, 8), True)}
    mask = np.zeros((16, 16), dtype=np.uint8)
    hits, idead, _, n_all, n_idead = m.hits_for_points([(3, 3)], gt, mask)
    assert (hits, idead, n_all, n_idead) == (1, 0, 2, 1)  # border Dead hit does not count as internal
    hits, idead, _, _, _ = m.hits_for_points([(8, 8)], gt, mask)
    assert (hits, idead) == (1, 1)


def test_bootstrap_ci_on_constant_delta():
    m = api()
    rows = [{"p_idead_hits": 1, "p_idead_n": 2, "b_idead_hits": 0, "b_idead_n": 2} for _ in range(10)]
    tissues = ["A"] * 5 + ["B"] * 5
    lo, hi = m.bootstrap_delta(rows, tissues, "idead", draws=50)
    assert lo == hi == 0.5


def test_point_head_forward_shapes_and_deterministic_init():
    m = api()
    torch.manual_seed(19)
    head1 = m.PointHead([32, 32, 32, 32])
    torch.manual_seed(19)
    head2 = m.PointHead([32, 32, 32, 32])
    for a, b in zip(head1.state_dict().values(), head2.state_dict().values()):
        assert torch.equal(a, b)
    feats = [torch.randn(2, 32, 18, 18), torch.randn(2, 32, 18, 18),
             torch.randn(2, 32, 18, 18), torch.randn(2, 32, 18, 18)]
    out = head1(feats)
    assert out.shape == (2, 1, 256, 256)
    out.sum().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head1.parameters())
    feats16 = [torch.randn(1, 32, 16, 16) for _ in range(4)]
    assert head1(feats16).shape == (1, 1, 256, 256)


def test_np_components_centroid_rule():
    m = api()
    prob = np.zeros((32, 32), dtype=np.float32)
    prob[10:13, 10:13] = 0.9
    prob[20:22, 25:28] = 0.8
    pts = m.np_components(prob)
    assert sorted(pts) == [(11, 11), (20, 26)]
    assert len(m.np_components(np.zeros((8, 8), dtype=np.float32))) == 0
