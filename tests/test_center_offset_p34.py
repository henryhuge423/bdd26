"""Behavior checks for the frozen-detector center-offset readout (P34)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1]/'scripts/train_center_offset_p34.py'
spec = importlib.util.spec_from_file_location('center_offset_p34', path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from proposal_p3_utils import gt_objects, score_points


class FixedRng:
    """Deterministic stand-in exposing the calls made by the original augmentation."""
    def __init__(self, rotations, flip):
        self.rotations, self.flip = rotations, flip

    def integers(self, high):
        return self.rotations

    def random(self):
        return 0. if self.flip else 1.

    def uniform(self, low, high):
        return 1.1


def scene():
    inst = np.zeros((256, 256), dtype=np.int32)
    typ = np.zeros((256, 256), dtype=np.uint8)
    inst[20:31, 40:55] = 3      # 11 x 15 rectangle
    inst[100:104, 200:203] = 9  # small 4 x 3 object
    inst[200:230, 10:20] = 12
    typ[inst == 3], typ[inst == 9], typ[inst == 12] = 4, 1, 2
    return inst, typ, gt_objects(inst, typ)


def test_center_fields_point_every_mask_pixel_to_its_own_center():
    inst, _, gt = scene()
    centers, weights = p.center_fields(inst, gt)
    assert (centers[inst == 0] == -1).all() and (weights[inst == 0] == 0).all()
    for oid, (_, (y, x), _, area) in zip((3, 9, 12), gt):
        assert set(centers[inst == oid].tolist()) == {y * 256 + x}
        assert weights[inst == oid].sum() == pytest.approx(1)
        assert area == (inst == oid).sum()
    assert weights.sum() == pytest.approx(3)


def test_center_fields_reject_inconsistent_annotation():
    inst, _, gt = scene()
    with pytest.raises(ValueError):
        p.center_fields(inst, gt[:2])
    moved = [(gt[0][0], (0, 0), gt[0][2], gt[0][3])] + gt[1:]
    with pytest.raises(ValueError):
        p.center_fields(inst, moved)


@pytest.mark.parametrize('rotations', [0, 1, 2, 3])
@pytest.mark.parametrize('flip', [False, True])
def test_augmented_targets_follow_the_moved_centers(rotations, flip):
    inst, _, gt = scene()
    image = np.random.default_rng(3).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    x, target, weight, count = p.training_sample(image, inst, gt, FixedRng(rotations, flip))
    assert count == 3 and x.shape == (3, 256, 256) and target.shape == (2, 256, 256)
    # Independent reference: move one-hot center images and the instance map with torch directly.
    def move(array):
        t = torch.rot90(torch.from_numpy(array), rotations, (-2, -1))
        return t.flip(-1) if flip else t
    moved_inst = move(inst.copy())
    assert torch.equal(weight > 0, moved_inst > 0)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
    for oid, (_, (y, x0), _, area) in zip((3, 9, 12), gt):
        onehot = np.zeros((256, 256), dtype=np.int32)
        onehot[y, x0] = 1
        cy, cx = [int(v) for v in torch.nonzero(move(onehot))[0]]
        mask = moved_inst == oid
        assert int(mask.sum()) == area
        assert torch.equal((yy + target[0])[mask], torch.full((area,), float(cy)))
        assert torch.equal((xx + target[1])[mask], torch.full((area,), float(cx)))
        assert weight[mask].sum().item() == pytest.approx(1)
    assert (target[:, moved_inst == 0] == 0).all()


def test_photometric_change_leaves_targets_unchanged_and_cached_fields_agree():
    inst, _, gt = scene()
    image = np.full((256, 256, 3), 90, dtype=np.uint8)
    plain = p.training_sample(image, inst, gt, None)
    cached = p.training_sample(image, inst, gt, None, p.center_fields(inst, gt))
    assert torch.equal(plain[1], cached[1]) and torch.equal(plain[2], cached[2])
    moved = p.training_sample(image, inst, gt, FixedRng(0, False))
    assert torch.equal(plain[1], moved[1]) and not torch.equal(plain[0], moved[0])


def test_loss_is_object_balanced_and_zero_for_exact_offsets():
    inst, _, gt = scene()
    _, target, weight, count = p.training_sample(np.zeros((256, 256, 3), np.uint8), inst, gt, None)
    target, weight = target[None], weight[None]
    assert p.offset_loss(target.clone(), target, weight, count).item() == 0
    wrong = target.clone()
    wrong[0, 0][torch.from_numpy(inst == 9)] += 2      # 12-pixel object, error 2 px
    small = p.offset_loss(wrong, target, weight, count).item()
    wrong = target.clone()
    wrong[0, 1][torch.from_numpy(inst == 12)] -= 2     # 300-pixel object, same error
    large = p.offset_loss(wrong, target, weight, count).item()
    assert small == pytest.approx(2 / 3) and large == pytest.approx(2 / 3)
    background = target.clone()
    background[0, :, 0, 0] = 50
    assert p.offset_loss(background, target, weight, count).item() == 0


def test_head_starts_at_zero_offset_and_has_the_frozen_size():
    torch.manual_seed(0)
    head = p.OffsetHead()
    out = head(torch.randn(2, 64, 128, 128))
    assert out.shape == (2, 2, 256, 256) and torch.count_nonzero(out) == 0
    assert sum(q.numel() for q in head.parameters()) == 64*64*9 + 64 + 64*64*9 + 64 + 2*64 + 2


def test_refinement_reads_the_peak_pixel_and_respects_bounds():
    offsets = np.zeros((2, 256, 256), dtype=np.float32)
    points = [(10, 10), (50, 60), (250, 250), (100, 100)]
    assert p.refine_points(points, offsets) == [(10., 10.), (50., 60.), (250., 250.), (100., 100.)]
    offsets[:, 50, 60] = (-1.5, 2.25)
    offsets[:, 250, 250] = (9, 9)
    offsets[:, 100, 100] = (30, -40)
    offsets[:, 51, 60] = (7, 7)                      # neighbouring pixel must not be read
    refined = p.refine_points(points, offsets)
    assert refined[1] == (48.5, 62.25) and refined[2] == (255., 255.)
    assert np.hypot(refined[3][0] - 100, refined[3][1] - 100) == pytest.approx(p.MAX_SHIFT)
    assert refined[3][0] > 100 and refined[3][1] < 100
    offsets[0, 10, 10] = np.nan
    with pytest.raises(FloatingPointError):
        p.refine_points(points, offsets)


def test_merge_keeps_the_higher_ranked_point_within_radius_only():
    points = [(10., 10.), (11., 11.), (10., 12.), (10., 12.01), (40., 40.), (11.5, 11.)]
    kept, owner = p.merge_points(points)
    assert kept == [0, 3, 4] and owner == [0, 0, 0, 3, 4, 0]
    assert p.merge_points([]) == ([], [])


def test_displaced_peak_becomes_a_match_and_duplicate_is_removed():
    inst, _, gt = scene()
    y, x = gt[0][1]
    prob = np.zeros((256, 256))
    prob[y, x + 5], prob[y + 2, x - 4] = .9, .8      # two displaced peaks in one nucleus
    cy, cx = gt[2][1]
    prob[cy, cx] = .7
    yy, xx = np.mgrid[:256, :256]
    centers, _ = p.center_fields(inst, gt)
    offsets = np.where(centers >= 0, np.stack([centers // 256 - yy, centers % 256 - xx]), 0).astype(np.float32)
    decoded = p.decode_all(prob, offsets)
    scores, objects, points = p.inspect_image(gt, inst, prob, offsets, decoded, 'm', 'development', 5)
    assert scores['original']['matched'] == 1 and scores['original']['candidates'] == 3
    assert scores['refined']['matched'] == 2 and scores['refined']['candidates'] == 3
    assert scores['merged']['matched'] == 2 and scores['merged']['candidates'] == 2
    assert scores['merged']['dead_hit'] == 1 and scores['original']['dead_hit'] == 0
    first = objects[0]
    assert not first['matched_original'] and first['matched_merged'] and first['points_inside_mask'] == 2
    assert first['nearest_inside_point_before_px'] == pytest.approx(np.hypot(2, 4))
    assert first['nearest_inside_point_after_px'] == 0 and first['mean_vote_error_px'] == 0
    assert objects[1]['points_inside_mask'] == 0 and objects[1]['nearest_inside_point_after_px'] is None
    assert [r['kept'] for r in points] == [True, False, True]
    assert points[1]['merged_into'] == 0 and points[1]['gt_merged'] is None and points[0]['gt_merged'] == 0


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.nn.Conv2d(64, 1, 1)


class ToyDetector(torch.nn.Module):
    """Frozen stand-in: bright blobs give peaks; coordinate channels make offsets learnable."""
    def __init__(self):
        super().__init__()
        self.head = ToyHead()
        with torch.no_grad():
            self.head.out.weight.zero_()
            self.head.out.weight[0, 0] = 40
            self.head.out.bias.fill_(-20)

    def forward(self, x):
        gray = F.avg_pool2d(x.mean(1, keepdim=True), 2)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 128), torch.linspace(0, 1, 128), indexing='ij')
        grid = torch.stack([yy, xx])[None].expand(len(x), -1, -1, -1).to(x)
        smooth = [F.avg_pool2d(gray, k, 1, k // 2) for k in (3, 5, 9)]
        wide = F.pad(smooth[2], (2, 2, 2, 2), mode='replicate')
        slope = torch.cat([wide[:, :, 4:, 2:-2] - wide[:, :, :-4, 2:-2], wide[:, :, 2:-2, 4:] - wide[:, :, 2:-2, :-4]], 1)
        features = torch.cat([gray, grid, *smooth, 20 * slope, -20 * slope,
                              torch.zeros(len(x), 54, 128, 128).to(x)], 1)
        return F.interpolate(self.head.out(features), size=(256, 256), mode='bilinear', align_corners=False)[:, 0]


class ToyFold:
    def __init__(self, count=12):
        rng = np.random.default_rng(11)
        self.images = np.full((count, 256, 256, 3), 30, dtype=np.uint8)
        self.inst = np.zeros((count, 256, 256), dtype=np.int32)
        typ = np.zeros((count, 256, 256), dtype=np.uint8)
        yy, xx = np.mgrid[:256, :256]
        for i in range(count - 1):                       # last image stays empty
            for k in range(4):
                cy, cx = 40 + 55 * (k // 2) + rng.integers(0, 9), 50 + 110 * (k % 2) + rng.integers(0, 9)
                blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= (6 + k) ** 2
                self.inst[i][blob] = k + 1
                typ[i][blob] = 4 if k == 0 else 1
                self.images[i][blob] = 230
        self.tissue = np.array(['Colon', 'Lung'] * (count // 2))
        self.gt = [gt_objects(a, b) for a, b in zip(self.inst, typ)]


def test_feature_tap_is_passive_and_fingerprint_detects_changes():
    model = ToyDetector().eval()
    x = torch.rand(2, 3, 256, 256)
    expected = model(x)
    before = p.state_fingerprint(model)
    tap = p.FeatureTap(model)
    logits, features = p.forward_frozen(model, tap, x, torch.device('cpu'))
    assert torch.equal(expected, logits) and features.shape == (2, 64, 128, 128)
    tap.close()
    assert not model.head.out._forward_pre_hooks and p.state_fingerprint(model) == before
    with torch.no_grad():
        model.head.out.bias.add_(1e-3)
    assert p.state_fingerprint(model) != before


def test_complete_run_on_a_frozen_toy_detector(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setattr(p, 'BATCH', 2)          # several optimizer steps per toy epoch
    monkeypatch.setattr(p, 'HEAD_LR', 2e-3)
    fold, model, device = ToyFold(), ToyDetector(), torch.device('cpu')
    fit, audit = list(range(8)), list(range(8, 12))
    selection = dict(development_images=[8, 11], training_images=[0])
    tap = p.FeatureTap(model)
    historical = {}
    for role, ids in (('development', audit), ('training', fit)):
        rows, _, _, _ = p.evaluate_role(model, tap, p.OffsetHead(), fold, ids, device, 'toy', role)
        historical[role] = {r['image']: r for r in rows['original']}
    tap.close()
    out = tmp_path/'toy'
    results, summary = p.run_model('toy', 19, model, fold, fit, audit, selection, out, device, historical, epochs=6)
    assert summary['frozen_detector_unchanged'] and summary['zero_start_identity'] and summary['reload_equal']
    assert summary['last_epoch_loss'] < summary['first_epoch_loss'] and summary['epochs'] == 6
    assert summary['witness']['first_output_weight_change'] > 0
    for role in ('development', 'training'):
        assert summary[role]['original_count_replay_equal']
        assert [r['image'] for r in results[role]['original']] == [r['image'] for r in results[role]['merged']]
        for name in p.DECODERS:
            saved = json.loads((out/f'{role}_{name}.json').read_text())
            assert saved == results[role][name]
            assert summary[role]['counts'][name]['total'] == sum(len(fold.gt[i]) for i in (audit if role == 'development' else fit))
    assert sorted(q.name for q in (out/'figures').iterdir()) == ['toy_development_0008.png', 'toy_development_0011.png', 'toy_training_0000.png']
    from PIL import Image
    with Image.open(out/'figures/toy_development_0008.png') as png:
        png.load()
        assert png.size == (3*768 + 2*36 + 2*28, 150 + 768 + 110) and png.mode == 'RGB'
    assert (out/'offset_head.pth').stat().st_size > 0 and len((out/'loss.jsonl').read_text().splitlines()) == 6
    header = (out/'development_objects.csv').read_text().splitlines()[0].split(',')
    assert {'matched_original', 'matched_merged', 'center_vote_error_px', 'pooled_vote_error_px'} <= set(header)
    assert summary['development']['offset_error']['all']['center_vote_error_px']['n'] == 12
    # A changed stored control must stop the run instead of being reported.
    historical['development'][8] = dict(historical['development'][8], matched=99)
    with pytest.raises(AssertionError):
        p.run_model('toy', 19, model, fold, fit, audit, selection, tmp_path/'changed', device, historical, epochs=2)


def test_first_epoch_time_guard_stops_before_further_training(tmp_path):
    fold, model, device = ToyFold(), ToyDetector(), torch.device('cpu')
    with pytest.raises(TimeoutError):
        p.run_model('toy', 1, model, fold, list(range(8)), list(range(8, 12)),
                    dict(development_images=[], training_images=[]), tmp_path/'slow', device, None,
                    epochs=3, epoch_limit=0)


def test_paired_decisions_use_the_original_rule_with_original_decoder_as_control():
    def rows(dead_hit, matched, candidates):
        return [dict(image=i, tissue='Colon' if i < 6 else 'Lung', matched=matched, total=20, candidates=candidates,
                     dead_hit=dead_hit, dead_total=10, small_hit=0, small_total=0) for i in range(12)]
    results = {('M', seed): dict(development=dict(original=rows(5, 14, 20), refined=rows(7, 16, 20),
                                                  merged=rows(7, 16, 19))) for seed in (19, 1)}
    decisions = p.paired_decisions(results)
    assert set(decisions) == {'M_refined', 'M_merged'}
    assert decisions['M_merged']['pass'] and decisions['M_merged']['details']['19']['delta'] == pytest.approx(.2)
    results[('M', 1)]['development']['merged'] = rows(5, 14, 20)
    assert not p.paired_decisions(results)['M_merged']['pass']


def test_configuration_is_the_single_frozen_recipe():
    config = p.run_config()
    assert config['epochs'] == 20 and config['max_shift_px'] == 16 and config['merge_radius_px'] == 2
    assert config['detector'] == 'frozen' and config['head_lr'] == 3e-4
    assert score_points([(1.2, 3.9)], [(4, (3, 5), True, 30)])['matched'] == 1
