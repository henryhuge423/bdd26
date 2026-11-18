"""Behavior checks for the frozen-detector center-offset readout (P34)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1]/'scripts/train_center_offset_p34.py'
spec = importlib.util.spec_from_file_location('center_offset_p34', path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from proposal_p3_utils import decode_peaks, gt_objects, score_points


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
    inst[20:31, 40:55] = 3      # 11 x 15 rectangle, center (25, 47)
    inst[100:104, 200:203] = 9  # small 4 x 3 object, center (101, 201)
    inst[200:230, 10:20] = 12   # center (214, 14)
    typ[inst == 3], typ[inst == 9], typ[inst == 12] = 4, 1, 2
    return inst, typ, gt_objects(inst, typ)


def count_rows(dead_hit, matched, candidates):
    return [dict(image=i, tissue='Colon' if i < 6 else 'Lung', matched=matched, total=20, candidates=candidates,
                 dead_hit=dead_hit, dead_total=10, small_hit=0, small_total=0) for i in range(12)]


def test_shift_moves_content_without_wrapping():
    a = np.arange(25).reshape(5, 5)
    s = p.shift_array(a, 1, -2, -7)
    assert s[1:, :3].tolist() == a[:4, 2:].tolist()
    assert (s[0] == -7).all() and (s[:, 3:] == -7).all()
    assert np.array_equal(p.shift_array(a, 0, 0, 9), a) and (p.shift_array(a, 5, 0, 9) == 9).all()
    rgb = np.zeros((7, 7, 3), dtype=np.uint8)
    rgb[3, 3] = (42, 66, 89)
    moved = p.shift_array(rgb, -2, 1, 255)
    assert moved[1, 4].tolist() == [42, 66, 89] and (moved[5:] == 255).all() and (moved[:, 0] == 255).all()
    assert rgb[3, 3].tolist() == [42, 66, 89]


def test_instance_fields_follow_the_annotation_rule():
    inst, _, gt = scene()
    assert [g[1] for g in gt] == [(25, 47), (101, 201), (214, 14)]
    centers, weights, count = p.instance_fields(inst, gt)
    assert count == 3 and (centers[inst == 0] == -1).all() and (weights[inst == 0] == 0).all()
    for oid, (_, (y, x), _, area) in zip((3, 9, 12), gt):
        assert set(centers[inst == oid].tolist()) == {y * 256 + x}
        assert weights[inst == oid].sum() == pytest.approx(1) and area == (inst == oid).sum()
    free = p.instance_fields(inst)
    assert np.array_equal(free[0], centers) and np.array_equal(free[1], weights) and free[2] == 3
    assert p.instance_fields(np.zeros((256, 256), dtype=np.uint16), [])[2] == 0


def test_instance_fields_reject_inconsistent_annotation():
    inst, _, gt = scene()
    with pytest.raises(ValueError):
        p.instance_fields(inst, gt[:2])
    with pytest.raises(ValueError):
        p.instance_fields(inst, gt + [gt[0]])
    with pytest.raises(ValueError):
        p.instance_fields(inst, [(gt[0][0], (0, 0), gt[0][2], gt[0][3])] + gt[1:])
    with pytest.raises(ValueError):
        p.instance_fields(inst, [(gt[0][0], gt[0][1], gt[0][2], gt[0][3] + 1)] + gt[1:])


def test_translated_annotation_recomputes_cut_nuclei_and_drops_departed_ones():
    inst, _, _ = scene()
    centers, weights, count = p.instance_fields(p.shift_array(inst, 0, -45, 0))
    assert count == 2 and weights.sum() == pytest.approx(2)
    assert set(centers[20:31, 0:10].ravel().tolist()) == {25 * 256 + 4}       # visible part of the cut rectangle
    assert set(centers[100:104, 155:158].ravel().tolist()) == {101 * 256 + 156}
    assert (centers[200:230] == -1).all()


@pytest.mark.parametrize('rotations', [0, 1, 2, 3])
@pytest.mark.parametrize('flip', [False, True])
def test_augmented_targets_follow_the_moved_centers(rotations, flip):
    inst, _, gt = scene()
    image = np.random.default_rng(3).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    x, target, weight, count = p.training_sample(image, p.instance_fields(inst, gt), FixedRng(rotations, flip))
    assert count == 3 and x.shape == (3, 256, 256) and target.shape == (2, 256, 256)

    def move(array):   # independent reference built directly with torch
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


def test_color_change_leaves_targets_unchanged():
    inst, _, gt = scene()
    image = np.full((256, 256, 3), 90, dtype=np.uint8)
    fields = p.instance_fields(inst, gt)
    plain = p.training_sample(image, fields, None)
    moved = p.training_sample(image, fields, FixedRng(0, False))
    assert torch.equal(plain[1], moved[1]) and torch.equal(plain[2], moved[2])
    assert not torch.equal(plain[0], moved[0])


def test_translation_draws_are_reproducible_bounded_and_varied():
    draws = [p.translation_draw(19, epoch, index) for epoch in range(6) for index in range(40)]
    assert all(-p.TRANSLATION <= v <= p.TRANSLATION for d in draws for v in d)
    assert len(set(draws)) > 100 and draws == [p.translation_draw(19, e, i) for e in range(6) for i in range(40)]
    assert draws != [p.translation_draw(1, e, i) for e in range(6) for i in range(40)]


def test_readouts_share_every_draw_except_the_translation():
    inst, _, gt = scene()
    image = np.random.default_rng(5).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    index = next(i for i in range(50) if p.translation_draw(19, 2, i) != (0, 0))
    dy, dx = p.translation_draw(19, 2, index)
    plain = p.readout_sample(image, inst, gt, 19, 2, index, False)
    moved = p.readout_sample(image, inst, gt, 19, 2, index, True)
    by_hand = p.readout_sample(p.shift_array(image, dy, dx, 255), p.shift_array(inst, dy, dx, 0), None,
                               19, 2, index, False)
    for a, b in zip(moved[:3], by_hand[:3]):
        assert torch.equal(a, b)
    assert moved[3] == by_hand[3] and not torch.equal(plain[1], moved[1])
    again = p.readout_sample(image, inst, gt, 19, 2, index, False)
    assert all(torch.equal(a, b) for a, b in zip(plain[:3], again[:3]))


def test_shifted_view_keeps_the_original_interior_status():
    inst = np.zeros((1, 256, 256), dtype=np.int32)
    typ = np.zeros((1, 256, 256), dtype=np.uint8)
    inst[0, 0:10, 0:10] = 1         # cut by the original border
    inst[0, 100:111, 100:111] = 2   # interior before and after
    inst[0, 240:251, 50:61] = 3     # interior before, on the border after the move
    typ[inst > 0] = 4
    fold = SimpleNamespace(images=np.full((1, 256, 256, 3), 100, dtype=np.uint8), inst=inst,
                           gt=[gt_objects(inst[0], typ[0])], tissue=np.array(['Colon']))
    view = p.ShiftedView(fold, typ, [0], *p.DIAGNOSTIC_SHIFT)
    assert [g[2] for g in fold.gt[0]] == [False, True, True]
    assert [g[2] for g in view.gt[0]] == [False, True, False]
    assert view.gt[0][1][1] == (fold.gt[0][1][1][0] + 5, fold.gt[0][1][1][1] + 3) and view.gt[0][1][3] == 121
    assert (view.images[0][:5] == 255).all() and (view.images[0][:, :3] == 255).all()
    assert (view.inst[0][105:116, 103:114] == 2).all()
    assert p.instance_fields(view.inst[0], view.gt[0])[2] == 3


def test_loss_is_object_balanced_and_zero_for_exact_offsets():
    inst, _, gt = scene()
    _, target, weight, count = p.training_sample(np.zeros((256, 256, 3), np.uint8), p.instance_fields(inst, gt))
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
    prob = np.zeros((256, 256))
    prob[25, 52], prob[27, 43] = .9, .8             # two displaced peaks inside the first nucleus
    prob[214, 14] = .7                               # a peak on the third nucleus center
    points = decode_peaks(prob)
    assert points == [(25, 52), (27, 43), (214, 14)]
    yy, xx = np.mgrid[:256, :256]
    centers, _, _ = p.instance_fields(inst, gt)
    exact = np.where(centers >= 0, np.stack([centers // 256 - yy, centers % 256 - xx]), 0).astype(np.float32)
    decoded = {'same_augmentation': p.decode_readout(points, exact),
               'with_translation': p.decode_readout(points, np.zeros_like(exact))}
    scores, objects, rows = p.inspect_image(gt, inst, prob, points, decoded, 'm', 'development', 5)
    assert scores['original']['matched'] == 1 and scores['original']['candidates'] == 3
    assert scores['original']['dead_hit'] == 0 and scores['original']['dead_total'] == 1
    assert scores[('same_augmentation', 'refined')]['matched'] == 2
    assert scores[('same_augmentation', 'refined')]['candidates'] == 3
    merged = scores[('same_augmentation', 'merged')]
    assert merged['matched'] == 2 and merged['candidates'] == 2 and merged['dead_hit'] == 1
    assert scores[('with_translation', 'refined')] == scores['original'] == scores[('with_translation', 'merged')]
    first = objects[0]
    assert not first['matched_original'] and first['matched_merged_same_augmentation']
    assert not first['matched_merged_with_translation'] and first['points_inside_mask'] == 2
    assert first['nearest_inside_point_px'] == pytest.approx(np.hypot(2, 4))
    assert first['nearest_inside_point_refined_px_same_augmentation'] == 0
    assert first['mean_vote_error_px_same_augmentation'] == 0 and first['mean_vote_error_px_with_translation'] > 3
    assert first['center_vote_error_px_with_translation'] == 0
    assert objects[1]['points_inside_mask'] == 0 and objects[1]['nearest_inside_point_px'] is None
    assert [r['kept_same_augmentation'] for r in rows] == [True, False, True]
    assert rows[1]['merged_into_same_augmentation'] == 0 and rows[1]['gt_merged_same_augmentation'] is None
    assert rows[0]['gt_merged_same_augmentation'] == 0 and rows[0]['gt_original'] is None and rows[2]['gt_original'] == 2
    assert rows[0]['refined_y_same_augmentation'] == 25 and rows[0]['offset_x_same_augmentation'] == -5
    assert [r['kept_with_translation'] for r in rows] == [True, True, True]


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.nn.Conv2d(64, 1, 1)


class ToyDetector(torch.nn.Module):
    """Frozen stand-in: bright blobs give peaks; smoothed-intensity slopes make offsets learnable."""
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
        self.types = np.zeros((count, 256, 256), dtype=np.uint8)
        yy, xx = np.mgrid[:256, :256]
        for i in range(count - 1):                       # the last image stays empty
            for k in range(4):
                cy, cx = 40 + 55 * (k // 2) + rng.integers(0, 9), 50 + 110 * (k % 2) + rng.integers(0, 9)
                blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= (6 + k) ** 2
                self.inst[i][blob] = k + 1
                self.types[i][blob] = 4 if k == 0 else 1
                self.images[i][blob] = 230
        self.tissue = np.array(['Colon', 'Lung'] * (count // 2))
        self.gt = [gt_objects(a, b) for a, b in zip(self.inst, self.types)]


def toy_setup():
    fold, model = ToyFold(), ToyDetector()
    selection = dict(development_images=[8, 11], training_images=[0, 1], training_figures=[0])
    return fold, model, torch.device('cpu'), list(range(8)), list(range(8, 12)), selection


def toy_stored_counts(fold, model, device, audit, selection):
    tap = p.FeatureTap(model)
    blank = {name: p.OffsetHead() for name in p.READOUTS}
    stored = {}
    for role, ids in (('development', audit), ('training', selection['training_images'])):
        rows, _, _, info = p.evaluate_role(model, tap, blank, fold, ids, device, 'toy', role)
        assert info['comparison'] is None and info['forward_seconds'] > 0
        stored[role] = {r['image']: r for r in rows['original']}
    tap.close()
    return stored


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


def test_stored_count_differences_are_reported_per_image():
    fold, model, device, _, audit, selection = toy_setup()
    stored = toy_stored_counts(fold, model, device, audit, selection)['development']
    tap = p.FeatureTap(model)
    heads = {name: p.OffsetHead() for name in p.READOUTS}
    _, objects, points, info = p.evaluate_role(model, tap, heads, fold, audit, device, 'toy', 'development', stored,
                                               point_images={8})
    assert info['comparison']['images_with_different_counts'] == 0 and info['comparison']['images'] == 4
    assert info['comparison']['stored_counts'] == info['comparison']['recomputed_counts']
    assert {r['image'] for r in points} == {8} and len(objects) == 12
    stored[8] = dict(stored[8], matched=stored[8]['matched'] + 1)
    info = p.evaluate_role(model, tap, heads, fold, audit, device, 'toy', 'development', stored)[3]
    tap.close()
    assert info['comparison']['images_with_different_counts'] == 1
    assert info['comparison']['stored_counts']['matched'] == info['comparison']['recomputed_counts']['matched'] + 1


def test_complete_run_on_a_frozen_toy_detector(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setattr(p, 'BATCH', 2)          # several optimizer steps per toy epoch
    monkeypatch.setattr(p, 'HEAD_LR', 2e-3)
    fold, model, device, fit, audit, selection = toy_setup()
    stored = toy_stored_counts(fold, model, device, audit, selection)
    out = tmp_path/'results'/'toy'
    results, summary = p.run_detector('toy', 19, model, fold, fold.types, fit, audit, selection, out, device,
                                      stored, epochs=8)
    assert summary['frozen_detector_unchanged'] and summary['zero_start_identity'] and summary['reload_equal']
    assert summary['annotation_rule_verified_images'] == 8 and summary['head_parameters'] == 73986
    for name in p.READOUTS:
        trained = summary['training'][name]
        assert trained['loss_decreased'] and trained['epochs'] == 8
        assert trained['witness']['first_output_weight_change'] > 0
        assert len((out/f'loss_{name}.jsonl').read_text().splitlines()) == 8
        assert (out/f'offset_head_{name}.pth').stat().st_size > 0
    heads = [summary['training'][name]['head_sha256'] for name in p.READOUTS]
    assert heads[0] != heads[1]
    expected_images = dict(development=4, training=2, training_shifted=2)
    for role in p.ROLES:
        described = summary['roles'][role]
        assert described['images'] == expected_images[role]
        assert [r['image'] for r in results[role]['original']] == \
            [r['image'] for r in results[role]['with_translation']['merged']]
        assert json.loads((out/f'{role}_original.json').read_text()) == results[role]['original']
        zero = summary['preflight'][role]
        for name in p.READOUTS:
            for decoder in p.NEW_DECODERS:
                assert json.loads((out/f'{role}_{name}_{decoder}.json').read_text()) == results[role][name][decoder]
                assert described['readouts'][name][decoder]['counts']['total'] == described['original']['counts']['total']
            assert zero['readouts'][name]['refined']['counts'] == zero['original']['counts']
        assert zero['original']['counts'] == described['original']['counts']
        header = (out/f'{role}_objects.csv').read_text().splitlines()[0].split(',')
        assert {'matched_original', 'matched_merged_with_translation', 'center_vote_error_px_same_augmentation',
                'pooled_vote_error_px_with_translation'} <= set(header)
    assert summary['roles']['development']['original']['counts']['total'] == 12
    assert summary['roles']['development']['stored_original_comparison']['images_with_different_counts'] == 0
    assert summary['roles']['training']['stored_original_comparison']['images'] == 2
    assert summary['roles']['training_shifted']['stored_original_comparison'] is None
    assert summary['roles']['development']['readouts']['with_translation']['offset_error']['all'][
        'center_vote_error_px']['n'] == 12
    names = ['toy_development_0008.png', 'toy_development_0011.png', 'toy_training_0000.png',
             'toy_training_shifted_0000.png']
    assert sorted(q.name for q in (out/'figures').iterdir()) == names
    assert sorted(q.name for q in (out/'preflight/figures').iterdir()) == names
    assert sorted(q.name for q in (out/'maps').iterdir()) == [n.replace('.png', '.npz') for n in names]
    from PIL import Image
    with Image.open(out/'figures/toy_development_0008.png') as png:
        png.load()
        assert png.size == (4*640 + 3*36 + 2*28, 150 + 640 + 110) and png.mode == 'RGB'
    saved = np.load(out/'maps/toy_development_0008.npz')
    assert saved['fine_logits'].shape == (256, 256) and saved['offsets_with_translation'].shape == (2, 256, 256)
    with pytest.raises(FileExistsError):
        p.run_detector('toy', 19, model, fold, fold.types, fit, audit, selection, out, device, stored, epochs=2)


def test_first_epoch_time_guard_stops_before_further_training(tmp_path):
    fold, model, device, fit, audit, selection = toy_setup()
    with pytest.raises(TimeoutError):
        p.run_detector('toy', 1, model, fold, fold.types, fit, audit, selection, tmp_path/'slow', device, None,
                       epochs=3, epoch_limit=0)
    assert not (tmp_path/'slow/offset_head_same_augmentation.pth').exists()
    assert (tmp_path/'slow/preflight/development_original.json').exists()


def test_paired_decisions_use_the_original_rule_with_the_original_decoder_as_control():
    results = {('M', seed): {'original': count_rows(5, 14, 20),
                             **{name: dict(refined=count_rows(7, 16, 20), merged=count_rows(7, 16, 19))
                                for name in p.READOUTS}} for seed in (19, 1)}
    decisions = p.paired_decisions(results)
    assert set(decisions) == {f'M:{name}:{decoder}' for name in p.READOUTS for decoder in p.NEW_DECODERS}
    chosen = decisions['M:with_translation:merged']
    assert chosen['pass'] and chosen['details']['19']['delta'] == pytest.approx(.2)
    assert chosen['details']['1']['trainable']['precision'] == pytest.approx(16 / 19)
    results[('M', 1)]['with_translation']['merged'] = count_rows(5, 14, 20)
    decisions = p.paired_decisions(results)
    assert not decisions['M:with_translation:merged']['pass'] and decisions['M:same_augmentation:merged']['pass']


def write_detector(root, run, epochs):
    folder = root/p.model_key(run)
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps(dict(experiment='P34', model=p.model_key(run), epochs=epochs,
                                                        exit_code=0)))
    (folder/'summary.json').write_text(json.dumps(dict(model=p.model_key(run))))
    (folder/'development_original.json').write_text(json.dumps(count_rows(5, 14, 20)))
    for name in p.READOUTS:
        for decoder in p.NEW_DECODERS:
            (folder/f'development_{name}_{decoder}.json').write_text(json.dumps(count_rows(7, 16, 19)))


def test_summary_combines_completed_detectors_and_rejects_incomplete_ones(tmp_path, capsys):
    runs = [dict(experiment='P22', arm='M', seed=19), dict(experiment='P22', arm='M', seed=1)]
    for run in runs:
        write_detector(tmp_path/'results', run, p.EPOCHS)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(runs=runs)))
    p.main(['--manifest', str(manifest), '--out', str(tmp_path/'results'), '--summarize'])
    assert '"M:with_translation:merged": true' in capsys.readouterr().out
    combined = json.loads((tmp_path/'results/summary.json').read_text())
    assert [d['model'] for d in combined['detectors']] == ['P22_M_s19', 'P22_M_s1']
    assert len(combined['decisions']) == 4 and all(v['pass'] for v in combined['decisions'].values())
    with pytest.raises(FileExistsError):
        p.summarize(tmp_path/'results', runs)
    write_detector(tmp_path/'short', runs[0], p.EPOCHS)
    write_detector(tmp_path/'short', runs[1], 3)
    with pytest.raises(ValueError):
        p.summarize(tmp_path/'short', runs)
    with pytest.raises(FileNotFoundError):
        p.summarize(tmp_path/'missing', runs)
    with pytest.raises(SystemExit):
        p.main(['--manifest', str(manifest), '--out', str(tmp_path/'results')])


def test_selection_and_center_geometry_depend_on_annotation_only():
    fold, _, _, fit, audit, _ = toy_setup()
    selection = p.frozen_selection(fold, fit, audit)
    assert selection == dict(development_images=[8, 9, 10, 11], training_images=list(range(8)),
                             training_figures=[0, 1, 2, 3])
    close = SimpleNamespace(gt=[[(1, (10, 10), True, 30), (1, (10, 12), True, 30), (1, (100, 100), True, 30)],
                                [(1, (5, 5), True, 30)], []])
    geometry = p.training_center_geometry(close, [0, 1, 2])
    assert geometry['training_objects_with_neighbor'] == 3 and geometry['nearest_center_within_merge_radius'] == 2
    assert geometry['nearest_center_within_3px'] == 2 and geometry['nearest_center_within_6px'] == 2


def test_configuration_is_the_single_frozen_recipe():
    config = p.run_config('cpu')
    assert config['epochs'] == 12 and config['max_shift_px'] == 16 and config['merge_radius_px'] == 2
    assert config['translation_px'] == 8 and config['readouts'] == ['same_augmentation', 'with_translation']
    assert config['diagnostic_shift_px'] == [5, 3] and config['head_lr'] == 3e-4 and config['batch'] == 16
    assert config['detector'] == 'frozen' and config['precision'] == 'cpu_float32'
    assert p.run_config('cuda')['precision'] == 'cuda_bfloat16_autocast'
    assert p.model_key(dict(experiment='P22', arm='M', seed=1)) == 'P22_M_s1'
    assert score_points([(1.2, 3.9)], [(4, (3, 5), True, 30)])['matched'] == 1
