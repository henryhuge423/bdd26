"""Behavior checks for alignment-resolved records, threshold selection and consensus decoding (P36)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('alignment_consensus_p36', scripts/'evaluate_alignment_consensus_p36.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from proposal_p3_utils import gt_objects, decode_peaks
from train_center_offset_p34 import shift_array, sha

CPU = torch.device('cpu')


class Equivariant(torch.nn.Module):
    """Score depends only on local image content: dark pixels score high."""
    def forward(self, x):
        return 12 - 40 * x.mean(1)


class GridBound(torch.nn.Module):
    """Same content rule, but the response exists only on rows 0-3 of every 16-pixel cell."""
    def forward(self, x):
        rows = (torch.arange(x.shape[-2]) % 16 < 4).float().view(1, -1, 1)
        return (12 - 40 * x.mean(1)) * rows - 12 * (1 - rows)


class Displaced(torch.nn.Module):
    """Peak two rows below and one column right of the dark pixel at every alignment."""
    def forward(self, x):
        return torch.roll(12 - 40 * x.mean(1), (2, 1), (-2, -1))


class Jittering(torch.nn.Module):
    """Peak column moves with the position of the image in the batch, that is with the alignment."""
    def forward(self, x):
        y = 12 - 40 * x.mean(1)
        return torch.stack([torch.roll(m, i % 3 - 1, -1) for i, m in enumerate(y)])


class AlignmentSensitive(torch.nn.Module):
    """Full response at the trained alignment; a score of 0.38 at the dark pixel at every other alignment."""
    def forward(self, x):
        y = 12 - 40 * x.mean(1)
        weak = torch.where(y > 0, torch.full_like(y, -.5), y)
        index = torch.arange(len(y)).view(-1, 1, 1)
        return torch.where(index == p.ORIGINAL, y, weak)


def scene(centers=((40, 60), (100, 130), (200, 90))):
    """White image with one dark pixel per nucleus center; 5 x 5 annotated squares of class Dead."""
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    inst = np.zeros((256, 256), dtype=np.int32)
    typ = np.zeros((256, 256), dtype=np.uint8)
    for number, (y, x) in enumerate(centers, start=1):
        image[y, x] = 0
        inst[y - 2:y + 3, x - 2:x + 3] = number
        typ[inst == number] = 4
    return image, gt_objects(inst, typ)


BOUND = ((34, 60), (99, 130), (200, 90))   # rows 34 and 99 respond at the trained alignment, row 200 does not


def toy_fold(n=6):
    a, gta = scene()
    b, gtb = scene(BOUND)
    return SimpleNamespace(images=[a, b] * (n // 2), gt=[gta, gtb] * (n // 2),
                           tissue=np.asarray(['Colon', 'Lung'] * (n // 2)))


def test_mirror_shift_moves_content_and_fills_the_strip_with_the_reflection():
    a = np.arange(36).reshape(6, 6)
    m = p.mirror_shift(a, 2, -1)
    assert np.array_equal(m[2:, :5], a[:4, 1:])                      # same movement as the white-fill shift
    assert np.array_equal(m[2:, :5], shift_array(a, 2, -1, -1)[2:, :5])
    assert np.array_equal(m[1, :5], a[1, 1:]) and np.array_equal(m[0, :5], a[2, 1:])   # rows mirrored about row 0
    assert np.array_equal(m[2:, 5], a[:4, 4])                        # column mirrored about the last column
    assert np.array_equal(p.mirror_shift(a, 0, 0), a) and p.mirror_shift(a, 0, 0) is not a
    rgb = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert np.array_equal(p.mirror_shift(rgb, -3, 4)[:13, 4:], rgb[3:, :12]) and rgb.dtype == np.uint8


def test_alignment_maps_return_to_original_coordinates_for_both_fills():
    image, _ = scene()
    plain = Equivariant()(torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255)[0].numpy()
    for fill in p.FILLS:
        logits = p.aligned_maps(Equivariant(), image, CPU, fill)
        assert logits.shape == (16, 256, 256) and np.array_equal(logits[p.ORIGINAL], plain)
        for k in range(16):
            assert np.allclose(logits[k][p.VALID[k]], plain[p.VALID[k]]) and not logits[k][~p.VALID[k]].any()
    assert p.VALID[p.ORIGINAL].all() and p.VALID.sum(0).min() == 4 and len(p.SHIFTED) == 15


def test_recorded_peaks_reproduce_the_original_decoder_and_keep_lower_scores():
    image, gt = scene()
    logits = p.aligned_maps(Equivariant(), image, CPU, 'white')
    logits[:, 150, 150] = -1.                                        # score 0.269: recorded, not retained
    logits[:, 20, 20] = -3.                                          # score 0.047: below the record threshold
    peaks = p.record_peaks(logits)
    assert p.retained(peaks[p.ORIGINAL]) == decode_peaks(p.probability(logits[p.ORIGINAL])) == [g[1] for g in gt]
    assert (150, 150) in [(y, x) for y, x, _ in peaks[p.ORIGINAL]] and (20, 20) not in [(y, x) for y, x, _ in peaks[3]]
    assert p.retained(peaks[p.ORIGINAL], .2)[-1] == (150, 150) and len(p.retained(peaks[p.ORIGINAL], .3)) == 3
    assert all(p.VALID[k, y, x] for k in range(16) for y, x, _ in peaks[k])


def test_consensus_groups_count_alignments_once_and_average_positions():
    points = [[] for _ in range(16)]
    points[0] = [(50, 50, .9), (52, 50, .6)]                         # two points of one alignment near each other
    points[1] = [(51, 51, .8)]
    points[2] = [(50, 53, .7), (120, 120, .95)]                      # (50, 53) is 3 px from the first point
    points[3] = [(51, 50, .4)]                                       # below 0.5: ignored
    points[p.ORIGINAL] = [(50, 50, .99)]                             # the trained alignment does not vote
    groups = p.consensus_groups(points, 2.)
    assert [(g['votes'], g['points']) for g in groups] == [(1, 1), (2, 3), (1, 1)]
    assert groups[0]['position'] == (120., 120.) and groups[1]['position'] == pytest.approx((51., 50 + 1 / 3))
    assert groups[2]['position'] == (50., 53.) and all(g['usable'] == 15 for g in groups)
    wide = p.consensus_groups(points, 3.)
    assert [(g['votes'], g['points']) for g in wide] == [(1, 1), (3, 4)]
    assert p.consensus_groups([[] for _ in range(16)], 3.) == []


def test_consensus_votes_are_relative_to_the_alignments_valid_at_the_position():
    interior = dict(position=(100., 100.), votes=7, points=7, usable=15)
    corner = dict(position=(2., 2.), votes=2, points=2, usable=3)    # near a corner only three shifted alignments see it
    edge = dict(position=(2., 100.), votes=3, points=3, usable=7)
    groups = [interior, corner, edge]
    assert p.consensus_points(groups, 7) == [(100., 100.), (2., 2.)]
    assert p.consensus_points(groups, 8) == [(2., 2.)] and p.consensus_points(groups, 11) == []
    assert p.consensus_points(groups, 6) == [(100., 100.), (2., 2.), (2., 100.)]
    seed = p.consensus_groups([[(2, 2, .9)] if k == p.SHIFTED[-1] else [] for k in range(16)], 2.)
    assert seed[0]['usable'] == 3 and p.consensus_groups([[(2, 100, .9)] if k == 15 else [] for k in range(16)], 2.)[0]['usable'] == 7


def test_equivariant_detector_gives_the_original_points_at_every_rule():
    fold = toy_fold(2)
    peaks, _ = p.collect(Equivariant(), fold, [0, 1], CPU)
    table = p.grid(peaks, fold, [0, 1])
    assert table['total'].tolist() == [3, 3] and table['dead_total'].tolist() == [3, 3]
    assert (table['thresholds'] == 3).all() and (table['consensus'] == 3).all()
    assert (table['shifted_thresholds'] == 45).all()                 # three nuclei at fifteen alignments
    assert p.select_threshold(table)['threshold'] == .5 and p.select_threshold(table)['f1'] == 1
    rule = p.select_rule(table)
    assert (rule['fill'], rule['radius'], rule['votes'], rule['f1']) == ('white', 2., 15, 1.)
    single, objects = p.describe(peaks, fold, [0, 1], rule, 'toy', 'training')
    assert (single == 3).all() and all(o['matched_consensus'] and o['shifted_matched_white'] == 15 for o in objects)
    assert all(o['mean_dy_mirror'] == 0 and o['spread2_mirror'] == 0 and o['neighbors12'] == 0 for o in objects)


def test_alignment_bound_misses_are_recovered_by_a_low_vote_rule_only():
    image, gt = scene(BOUND)
    fold = SimpleNamespace(images=[image], gt=[gt], tissue=np.asarray(['Lung']))
    peaks, _ = p.collect(GridBound(), fold, [0], CPU)
    table = p.grid(peaks, fold, [0])
    half = p.THRESHOLDS.index(.5)
    assert table['thresholds'][0, half].tolist() == [2, 2, 2]        # the trained alignment misses the third nucleus
    white = table['consensus'][0, 0, 0]                              # white fill, radius 2
    assert white[:3, 0].tolist() == [3, 3, 3] and white[3, 0] == 1 and (white[4:, 0] == 0).all()
    rule = p.select_rule(table)
    assert (rule['fill'], rule['radius'], rule['votes'], rule['matched'], rule['points']) == ('white', 2., 3, 3, 3)
    _, objects = p.describe(peaks, fold, [0], rule, 'toy', 'development')
    assert [o['matched_original'] for o in objects] == [True, True, False]
    assert [o['shifted_matched_white'] for o in objects] == [3, 3, 4] and all(o['matched_consensus'] for o in objects)
    # Bit k of the mask = matched at alignment k: the first nucleus at the four alignments without a row shift.
    assert objects[0]['phases_white'] == sum(1 << k for k, (dy, _) in enumerate(p.PHASES) if dy == 0)
    assert objects[2]['phases_white'] == sum(1 << k for k, (dy, _) in enumerate(p.PHASES) if dy == -8)


def test_nearest_offsets_report_the_closest_point_within_the_window():
    centers = [(10, 10), (50, 50), (90, 90)]
    points = [(12, 9), (10, 14), (57, 50)]
    out = p.nearest_offsets(points, centers)
    assert out[0].tolist() == [2., -1.] and np.isnan(out[1]).all() and np.isnan(out[2]).all()
    assert p.nearest_offsets(points, centers, 8.)[1].tolist() == [7., 0.]
    assert p.nearest_offsets([], centers).shape == (3, 2) and p.nearest_offsets(points, []).shape == (0, 2)


def test_stable_displacement_and_alignment_jitter_are_separated():
    image, gt = scene()
    fold = SimpleNamespace(images=[image], gt=[gt], tissue=np.asarray(['Colon']))
    rule = dict(fill='white', radius=3., votes=8)
    for model, stable, jitter in ((Displaced(), 5., 0.), (Jittering(), 0., None)):
        peaks, _ = p.collect(model, fold, [0], CPU)
        _, objects = p.describe(peaks, fold, [0], rule, 'toy', 'development')
        summary = p.object_summary(objects)
        assert summary['central_objects'] == 3 and summary['white']['isolated_objects_with_two_or_more_points'] == 3
        if jitter is None:
            columns = np.asarray([i % 3 - 1 for i in p.SHIFTED], float)
            assert summary['white']['alignment_jitter_px2'] == pytest.approx(columns.var(ddof=1))
            # Unbiased estimate of the squared mean displacement: squared sample mean minus variance / count.
            expected = columns.mean() ** 2 - columns.var(ddof=1) / len(columns)
            assert summary['white']['stable_displacement_px2'] == pytest.approx(expected)
            assert objects[0]['mean_dx_white'] == pytest.approx(columns.mean()) and objects[0]['mean_dy_white'] == 0
        else:
            assert summary['white']['alignment_jitter_px2'] == 0 and summary['white']['stable_displacement_px2'] == stable
            assert (objects[0]['original_dy'], objects[0]['original_dx']) == (2., 1.)
        assert summary['border_bands']['32-128']['objects'] == 3 and summary['white']['histogram'][15] == 3


def test_border_bands_and_alignment_histogram_use_nuclei_valid_at_every_alignment():
    def row(border, valid, hit, original):
        return dict(border=border, valid_phases=valid, matched_original=original, shifted_matched_white=hit,
                    shifted_matched_mirror=15, neighbors12=1, spread2_white=None, spread2_mirror=None)
    summary = p.object_summary([row(5, 12, 9, True), row(10, 16, 6, True), row(20, 16, 12, False), row(40, 16, 15, True),
                                row(100, 16, 0, False)])
    assert summary['valid_at_every_alignment'] == 4 and summary['central_objects'] == 2
    assert summary['border_bands']['0-16'] == dict(objects=1, original=1., white=.4, mirror=1.)
    assert summary['border_bands']['16-24']['white'] == pytest.approx(.8) and summary['border_bands']['24-32']['objects'] == 0
    assert summary['white']['histogram'][0] == 1 and summary['white']['histogram'][15] == 1
    assert summary['white']['alignment_jitter_px2'] is None and summary['mirror']['histogram'][15] == 2


def test_rule_selection_prefers_f1_then_white_fill_smaller_radius_and_more_votes():
    table = dict(total=np.asarray([10]), consensus=np.zeros((1, 2, 3, 15, 3), np.int32))
    table['consensus'][0, :, :, :, 1] = 10
    table['consensus'][0, :, :, :, 0] = 6
    table['consensus'][0, 1, 2, 4] = (9, 10, 0)                      # mirror, radius 4, votes 5: best F1
    assert {k: p.select_rule(table)[k] for k in ('fill', 'radius', 'votes')} == dict(fill='mirror', radius=4., votes=5)
    table['consensus'][0, 1, 2, 4] = (6, 10, 0)
    assert {k: p.select_rule(table)[k] for k in ('fill', 'radius', 'votes')} == dict(fill='white', radius=2., votes=15)
    table['consensus'][0, 0, 1, 7] = (8, 12, 0)                      # F1 16/22 beats 12/20
    best = p.select_rule(table)
    assert (best['fill'], best['radius'], best['votes']) == ('white', 3., 8) and best['f1'] == pytest.approx(16 / 22)
    assert p.rule_index(best) == (0, 1, 7)


def test_threshold_selection_uses_f1_at_the_shifted_alignments_and_prefers_values_near_one_half():
    table = dict(total=np.asarray([10]), shifted_thresholds=np.zeros((1, 2, 9, 3), np.int32))
    table['shifted_thresholds'][0, :, :, 0] = 75                     # per fill: 75 of 150 nucleus views matched
    table['shifted_thresholds'][0, :, :, 1] = 100
    best = p.select_threshold(table)
    assert best['threshold'] == .5 and best['nuclei'] == 300 and best['f1'] == pytest.approx(300 / 500)
    table['shifted_thresholds'][0, :, 2] = (100, 130, 0)             # threshold 0.3: F1 400/560
    best = p.select_threshold(table)
    assert best['threshold'] == .3 and (best['matched'], best['points']) == (200, 260)
    assert best['f1'] == pytest.approx(400 / 560)
    table['shifted_thresholds'][0, :, 3] = (100, 130, 0)             # the same F1 at 0.4, which is closer to 0.5
    assert p.select_threshold(table)['threshold'] == .4


def test_threshold_selection_sees_weak_shifted_responses_that_the_consensus_at_one_half_cannot_use():
    image, gt = scene()
    fold = SimpleNamespace(images=[image], gt=[gt], tissue=np.asarray(['Colon']))
    peaks, _ = p.collect(AlignmentSensitive(), fold, [0], CPU)
    table = p.grid(peaks, fold, [0])
    assert (table['thresholds'][0, :, 0] == 3).all()                 # trained alignment: found at every threshold
    assert table['shifted_thresholds'][0, :, :3, 0].tolist() == [[45, 45, 45]] * 2
    assert not table['shifted_thresholds'][0, :, 3:].any() and not table['consensus'].any()
    cut, rule = p.select_threshold(table), p.select_rule(table)
    assert cut['threshold'] == .3 and cut['f1'] == 1
    assert (rule['fill'], rule['radius'], rule['votes'], rule['f1']) == ('white', 2., 15, 0.)


def test_complete_run_selects_on_training_images_before_development_and_writes_every_record(tmp_path):
    fold = toy_fold(6)
    historical = dict(development={4: dict(matched=3, total=3, candidates=3, dead_hit=3, dead_total=3),
                                   5: dict(matched=3, total=3, candidates=3, dead_hit=3, dead_total=3)})
    seen = []

    class Watched(GridBound):
        def forward(self, x):
            seen.append((tmp_path/'toy/selected_rule.json').exists())
            return super().forward(x)

    results, summary = p.run_detector('toy', Watched(), fold, [0, 1, 2, 3], [4, 5], [5], tmp_path/'toy', CPU, historical)
    assert seen == [False] * 8 + [True] * 4                          # two fills per image; training first
    saved_rule = json.loads((tmp_path/'toy/selected_rule.json').read_text())
    rule = saved_rule['consensus']
    assert (rule['fill'], rule['radius'], rule['votes'], saved_rule['development_images_read']) == ('white', 2., 3, 0)
    assert saved_rule['threshold']['threshold'] == .5 == summary['threshold']['threshold'] and summary['rule'] == rule
    dev = summary['roles']['development']
    # Image 4 has no nucleus on a responding row at the trained alignment; image 5 has two.
    assert dev['decoders']['original']['counts'] == dict(matched=2, total=6, candidates=2, dead_hit=2, dead_total=6)
    assert dev['decoders']['consensus']['counts'] == dict(matched=6, total=6, candidates=6, dead_hit=6, dead_total=6)
    assert dev['stored_original_comparison']['images_with_different_counts'] == 2
    assert dev['thresholds'][4] == dict(threshold=.5, counts=[2, 2, 2]) and dev['nuclei'] == 6 and dev['interior_dead'] == 6
    assert np.asarray(dev['consensus_grid']).shape == (2, 3, 15, 3) and np.asarray(dev['single_phases']).shape == (2, 16, 3)
    assert np.asarray(dev['shifted_thresholds']).shape == (2, 9, 3)
    assert dev['decoders']['selected_threshold'] == dev['decoders']['original']
    assert summary['roles']['training']['stored_original_comparison'] is None and summary['frozen_detector_unchanged']
    assert [r['image'] for r in results['development']['consensus']] == [4, 5]
    for name in ('development_original.json', 'development_consensus.json', 'development_selected_threshold.json',
                 'development_objects.csv',
                 'development_counts.npz', 'development_points.npz', 'training_objects.csv', 'training_counts.npz',
                 'summary.json', 'figures/toy_development_0005.png', 'figures/toy_training_0000.png'):
        assert (tmp_path/'toy'/name).is_file(), name
    from PIL import Image
    assert Image.open(tmp_path/'toy/figures/toy_development_0005.png').size[0] > 2600
    saved = np.load(tmp_path/'toy/development_points.npz')
    assert set(saved['image'].tolist()) == {4, 5} and set(saved['fill'].tolist()) == {0, 1} and saved['score'].min() >= .1
    counts = np.load(tmp_path/'toy/development_counts.npz')
    assert counts['single'].shape == (2, 2, 16, 3) and counts['consensus'].shape == (2, 2, 3, 15, 3)
    assert counts['shifted_thresholds'].shape == (2, 2, 9, 3)
    with pytest.raises(FileExistsError):
        p.run_detector('toy', GridBound(), fold, [0, 1], [4, 5], [5], tmp_path/'toy', CPU)


def test_nonfinite_detector_output_stops_the_evaluation():
    class Broken(torch.nn.Module):
        def forward(self, x):
            return x.mean(1) * float('nan')
    with pytest.raises(FloatingPointError):
        p.aligned_maps(Broken(), scene()[0], CPU, 'mirror')


def test_training_sample_is_fixed_by_image_number_alone():
    fit = list(range(0, 3000, 3))
    sample = p.sample_training(fit)
    assert len(sample) == 240 == len(set(sample)) and sample == sorted(sample) and set(sample) <= set(fit)
    assert p.sample_training(fit[::-1]) == sample and p.sample_training(np.asarray(fit)) == sample
    assert p.sample_training(fit + [3001]) != sample or 3001 not in p.sample_training(fit + [3001])


def count_rows(dead_hit, matched, candidates):
    return [dict(image=i, tissue='Colon' if i < 6 else 'Lung', matched=matched, total=20, candidates=candidates,
                 dead_hit=dead_hit, dead_total=10) for i in range(12)]


def test_paired_decisions_compare_decoders_within_a_detector_and_recipes_against_their_control(tmp_path, capsys):
    base, more = count_rows(5, 14, 18), count_rows(7, 15, 18)
    control = dict(original=base, selected_threshold=base, consensus=base)
    treated = dict(original=more, selected_threshold=more, consensus=base)
    decoder = dict(original=base, selected_threshold=base, consensus=more)
    decisions = p.paired_decisions({('P22', 'C', 19): control, ('P22', 'C', 1): control, ('P22', 'M', 19): treated,
                                    ('P22', 'M', 1): treated, ('P15', 'J', 19): decoder, ('P15', 'J', 1): decoder,
                                    ('P15', 'C', 19): control})
    assert decisions['P22:M:original_vs_C:original']['pass'] and decisions['P22:M:selected_threshold_vs_C:original']['pass']
    assert not decisions['P22:M:consensus_vs_C:original']['pass'] and not decisions['P22:M:consensus']['pass']
    assert decisions['P22:M:original_vs_C:original']['details']['1']['delta'] == pytest.approx(.2)
    assert decisions['P15:J:consensus']['pass'] and not decisions['P15:J:selected_threshold']['pass']
    assert decisions['P15:J:consensus_vs_C:original']['reason'] == 'missing_seed'       # control seed 1 is absent
    assert decisions['P15:C:consensus']['reason'] == 'missing_seed'
    assert not any(key.startswith('P22:C:') and '_vs_' in key for key in decisions)
    assert len(decisions) == 2 * 4 + 3 * 2
    runs = [dict(experiment='P22', arm=arm, seed=seed) for arm in ('C', 'M') for seed in (19, 1)]
    for run in runs:
        folder = tmp_path/p.model_key(run)
        folder.mkdir()
        for name, rows in (control if run['arm'] == 'C' else treated).items():
            (folder/f'development_{name}.json').write_text(json.dumps(rows))
        (folder/'summary.json').write_text(json.dumps(dict(model=p.model_key(run))))
        (folder/'complete.json').write_text(json.dumps(dict(model=p.model_key(run), exit_code=0, phases=16)))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(runs=runs)))
    p.main(['--manifest', str(manifest), '--out', str(tmp_path), '--summarize'])
    printed = json.loads(capsys.readouterr().out)
    assert printed['P22:M:original_vs_C:original'] and not printed['P22:M:consensus'] and len(printed) == 7
    assert len(json.loads((tmp_path/'summary.json').read_text())['detectors']) == 4
    (tmp_path/'P22_M_s1/complete.json').write_text(json.dumps(dict(model='P22_M_s1', exit_code=1, phases=16)))
    with pytest.raises(ValueError):
        p.summarize(tmp_path, runs)


def test_detector_identity_check_accepts_a_preserved_run_and_rejects_changes(tmp_path):
    module = SimpleNamespace(run_config=lambda arm, seed: dict(arm=arm, seed=seed, betas=(.9, .999)))
    run = dict(experiment='P15', arm='J', seed=19, directory=str(tmp_path))
    files = {'manifest.json': json.dumps(dict(weights_sha256='w', cohort_sha256='c',
                                              config=dict(arm='J', seed=19, betas=[.9, .999]))),
             'complete.json': json.dumps(dict(epochs=130, exit_code=0)), 'last.pth': 'checkpoint',
             'evaluation.json': '[]', 'fit.json': '[]',
             'loss.jsonl': '\n'.join(json.dumps(dict(epoch=e)) for e in range(1, 131))}
    for name, text in files.items():
        (tmp_path/name).write_text(text)
    preserved = {name: dict(bytes=(tmp_path/name).stat().st_size, sha256=sha(tmp_path/name)) for name in files}
    (tmp_path/'preservation_manifest.json').write_text(json.dumps(preserved))
    manifest, preservation = p.validate_detector(run, module, 'w', 'c')
    assert manifest['config']['arm'] == 'J' and preservation['last.pth']['bytes'] == 10
    with pytest.raises(AssertionError):
        p.validate_detector(dict(run, arm='C'), module, 'w', 'c')
    with pytest.raises(AssertionError):
        p.validate_detector(run, module, 'w', 'other cohort')
    (tmp_path/'last.pth').write_text('checkpoinT')
    with pytest.raises(AssertionError):
        p.validate_detector(run, module, 'w', 'c')


def test_configuration_is_the_single_frozen_recipe():
    config = p.run_config('cpu')
    assert config['fills'] == ['white', 'mirror'] and config['thresholds'] == [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    assert config['consensus'] == dict(alignments='fifteen_shifted', score=.5, radii_px=[2., 3., 4.],
                                       votes=list(range(1, 16)), position='mean_of_points',
                                       selection='training_sample_F1_all_nuclei',
                                       ties='white_fill,smaller_radius,more_votes')
    assert config['training'] == 'none' and config['training_sample'] == 240 and config['matching_radius_px'] == 3
    assert config['threshold_selection'] == dict(alignments='fifteen_shifted', fills='both_pooled',
                                                 criterion='training_sample_F1_all_nuclei', ties='closest_to_0.5')
    assert json.loads(json.dumps(config)) == config and len(config['phases']) == 16
    assert p.RUNS[:4] == [('P22', 'C', 19), ('P22', 'C', 1), ('P22', 'M', 19), ('P22', 'M', 1)]
    assert p.RUNS[4:] == [('P15', 'C', 19), ('P15', 'C', 1), ('P15', 'J', 19), ('P15', 'J', 1)]
