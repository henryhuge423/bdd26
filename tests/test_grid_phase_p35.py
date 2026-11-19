"""Behavior checks for the sixteen-alignment evaluation of frozen detectors (P35)."""
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
spec = importlib.util.spec_from_file_location('grid_phase_p35', scripts/'evaluate_grid_phase_p35.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from proposal_p3_utils import gt_objects

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


def scene(centers=((40, 60), (100, 130), (200, 90))):
    """White image with one dark pixel per nucleus center; 5 x 5 annotated squares."""
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    inst = np.zeros((256, 256), dtype=np.int32)
    typ = np.zeros((256, 256), dtype=np.uint8)
    for number, (y, x) in enumerate(centers, start=1):
        image[y, x] = 0
        inst[y - 2:y + 3, x - 2:x + 3] = number
        typ[inst == number] = 4
    return image, gt_objects(inst, typ)


def test_phases_cover_one_token_period_and_include_the_trained_alignment():
    assert len(p.PHASES) == 16 == len(set(p.PHASES)) and p.PHASES[p.ORIGINAL] == (0, 0)
    assert {(dy % 16, dx % 16) for dy, dx in p.PHASES} == {(a, b) for a in (0, 4, 8, 12) for b in (0, 4, 8, 12)}
    assert max(abs(v) for phase in p.PHASES for v in phase) <= 8


def test_realigned_maps_return_to_original_coordinates_with_exact_validity():
    image, _ = scene()
    logits, valid = p.phase_maps(Equivariant(), image, CPU)
    assert logits.shape == valid.shape == (16, 256, 256) and valid[p.ORIGINAL].all()
    plain = Equivariant()(torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255)[0].numpy()
    assert np.array_equal(logits[p.ORIGINAL], plain)
    for k, (dy, dx) in enumerate(p.PHASES):
        assert valid[k].sum() == (256 - abs(dy)) * (256 - abs(dx))
        assert np.allclose(logits[k][valid[k]], plain[valid[k]])
    k = p.PHASES.index((-8, 4))                      # content moved up and right
    assert not valid[k, :8].any() and not valid[k, :, -4:].any() and valid[k, 8:, :-4].all()


def test_equivariant_detector_gives_identical_decoders_and_full_alignment_counts():
    image, gt = scene()
    logits, valid = p.phase_maps(Equivariant(), image, CPU)
    maps = p.ensemble_maps(logits, valid)
    assert np.allclose(maps['mean_probability'], maps['original']) and np.allclose(maps['mean_logit'], maps['original'])
    scores, phase_scores, objects, _, points = p.inspect_image(gt, logits, valid, 'toy', 'development', 5)
    assert points['original'] == points['mean_probability'] == points['mean_logit'] == [g[1] for g in gt]
    assert all(s['matched'] == 3 for s in phase_scores) and scores['original']['dead_hit'] == 3
    assert [o['phases_matched'] for o in objects] == [16, 16, 16] and all(o['valid_phases'] == 16 for o in objects)
    assert all(o['matched_other_phases'] == 15 and o['phases_near_center_above_threshold'] == 16 for o in objects)


def test_grid_bound_detector_is_exposed_by_the_alignment_counts():
    # Center rows 34 and 99 respond at the trained alignment (34 % 16 = 2, 99 % 16 = 3); row 200 % 16 = 8 does not.
    image, gt = scene(((34, 60), (99, 130), (200, 90)))
    logits, valid = p.phase_maps(GridBound(), image, CPU)
    scores, phase_scores, objects, maps, points = p.inspect_image(gt, logits, valid, 'toy', 'development', 5)
    assert [o['matched_original'] for o in objects] == [True, True, False]
    # A row responds when (row + dy) % 16 < 4: one of the four row steps for every nucleus here.
    assert [o['phases_matched'] for o in objects] == [4, 4, 4]
    assert [o['matched_other_phases'] for o in objects] == [3, 3, 4]
    assert scores['original']['matched'] == 2 and sorted(s['matched'] for s in phase_scores) == [0] * 8 + [1] * 4 + [2] * 4
    assert objects[0]['near_center_score_phase_max'] > .99 and objects[0]['near_center_score_phase_min'] < .01
    assert abs(objects[0]['near_center_score_mean_probability'] - .25) < .01
    assert scores['mean_probability']['candidates'] == 0 and not any(o['matched_mean_probability'] for o in objects)
    distribution = p.phase_distribution(objects)['interior_dead']
    assert distribution['histogram'][4] == 3 and distribution['sometimes'] == 3 and distribution['never'] == 0
    assert distribution['matched_at_any_alignment'] == 3 and distribution['matched_original'] == 2
    assert distribution['between_object_variance_share'] == 0


def test_nuclei_near_the_border_report_fewer_valid_alignments_and_are_left_out_of_the_distribution():
    image, gt = scene(((5, 60), (100, 130)))
    logits, valid = p.phase_maps(Equivariant(), image, CPU)
    _, _, objects, _, _ = p.inspect_image(gt, logits, valid, 'toy', 'development', 5)
    assert objects[0]['valid_phases'] == 12 and objects[0]['phases_matched'] == 12 and objects[1]['valid_phases'] == 16
    assert p.phase_distribution(objects)['all']['objects'] == 1


def test_between_object_share_separates_nucleus_effects_from_alignment_effects():
    def rows(counts):
        return [dict(cls=1, interior=True, valid_phases=16, phases_matched=k, matched_original=k > 0,
                     matched_mean_probability=k > 8, matched_mean_logit=k > 8) for k in counts]
    assert p.phase_distribution(rows([0, 0, 16, 16]))['all']['between_object_variance_share'] == 1
    assert p.phase_distribution(rows([8, 8, 8, 8]))['all']['between_object_variance_share'] == 0
    mixed = p.phase_distribution(rows([0, 4, 12, 16]))['all']
    assert 0 < mixed['between_object_variance_share'] < 1 and mixed['never'] == 1 and mixed['always'] == 1
    assert p.phase_distribution(rows([16, 16]))['all']['between_object_variance_share'] is None


def toy_fold():
    a, gta = scene()
    b, gtb = scene(((34, 60), (99, 130), (200, 90)))
    return SimpleNamespace(images=[a, b, a], gt=[gta, gtb, gta], tissue=np.asarray(['Colon', 'Lung', 'Colon']))


def test_complete_run_writes_every_record_and_leaves_the_detector_unchanged(tmp_path):
    fold = toy_fold()
    selection = dict(development_images=[1], training_images=[2], training_figures=[2])
    historical = dict(development={0: dict(matched=3, total=3, candidates=3, dead_hit=3, dead_total=3),
                                   1: dict(matched=3, total=3, candidates=3, dead_hit=3, dead_total=3)})
    model = GridBound()
    results, summary = p.run_detector('toy', model, fold, [0, 1], selection, tmp_path/'toy', CPU, historical)
    dev = summary['roles']['development']
    assert dev['decoders']['original']['counts']['matched'] == 2 and dev['images'] == 2
    assert dev['stored_original_comparison']['images_with_different_counts'] == 2
    assert len(dev['single_phases']) == 16 and dev['single_phases'][p.ORIGINAL]['counts'] == dev['decoders']['original']['counts']
    assert dev['alignment_distribution']['interior_dead']['objects'] == 6
    assert summary['roles']['training']['stored_original_comparison'] is None and summary['frozen_detector_unchanged']
    for name in ('development_original.json', 'development_mean_probability.json', 'development_mean_logit.json',
                 'development_single_phases.json', 'development_objects.csv', 'training_objects.csv', 'summary.json',
                 'figures/toy_development_0001.png', 'figures/toy_training_0002.png', 'maps/toy_development_0001.npz'):
        assert (tmp_path/'toy'/name).is_file(), name
    from PIL import Image
    assert Image.open(tmp_path/'toy/figures/toy_development_0001.png').size[0] > 2600
    saved = np.load(tmp_path/'toy/maps/toy_development_0001.npz')
    assert saved['logits'].shape == (16, 256, 256) and saved['valid'].dtype == bool
    with pytest.raises(FileExistsError):
        p.run_detector('toy', model, fold, [0, 1], selection, tmp_path/'toy', CPU)


def test_nonfinite_detector_output_stops_the_evaluation():
    class Broken(torch.nn.Module):
        def forward(self, x):
            return x.mean(1) * float('nan')
    with pytest.raises(FloatingPointError):
        p.phase_maps(Broken(), scene()[0], CPU)


def count_rows(dead_hit, matched, candidates):
    return [dict(image=i, tissue='Colon' if i < 6 else 'Lung', matched=matched, total=20, candidates=candidates,
                 dead_hit=dead_hit, dead_total=10, small_hit=0, small_total=0) for i in range(12)]


def test_paired_decisions_apply_the_original_rule_with_the_original_decoder_as_control():
    good = dict(original=count_rows(5, 14, 18), mean_probability=count_rows(7, 15, 18), mean_logit=count_rows(5, 14, 18))
    decisions = p.paired_decisions({('M', 19): good, ('M', 1): good, ('C', 19): good})
    assert decisions['M:mean_probability']['pass'] and not decisions['M:mean_logit']['pass']
    assert decisions['C:mean_probability'] == dict({'pass': False, 'reason': 'missing_seed'},
                                                   details=decisions['C:mean_probability']['details'])
    assert decisions['M:mean_probability']['details']['19']['delta'] == pytest.approx(.2)


def test_summary_combines_completed_detectors_and_rejects_incomplete_ones(tmp_path, capsys):
    runs = [dict(experiment='P22', arm='M', seed=19), dict(experiment='P22', arm='M', seed=1)]
    for run in runs:
        folder = tmp_path/p.model_key(run)
        folder.mkdir()
        for name, rows in (('original', count_rows(5, 14, 18)), ('mean_probability', count_rows(7, 15, 18)),
                           ('mean_logit', count_rows(5, 14, 18))):
            (folder/f'development_{name}.json').write_text(json.dumps(rows))
        (folder/'summary.json').write_text(json.dumps(dict(model=p.model_key(run))))
        (folder/'complete.json').write_text(json.dumps(dict(model=p.model_key(run), exit_code=0, phases=16)))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(runs=runs)))
    p.main(['--manifest', str(manifest), '--out', str(tmp_path), '--summarize'])
    assert json.loads(capsys.readouterr().out) == {'M:mean_probability': True, 'M:mean_logit': False}
    assert len(json.loads((tmp_path/'summary.json').read_text())['detectors']) == 2
    (tmp_path/'P22_M_s1/complete.json').write_text(json.dumps(dict(model='P22_M_s1', exit_code=0, phases=4)))
    with pytest.raises(ValueError):
        p.summarize(tmp_path, runs)


def test_configuration_is_the_single_frozen_recipe():
    config = p.run_config('cpu')
    assert config['phases'] == [[dy, dx] for dy in (-8, -4, 0, 4) for dx in (-8, -4, 0, 4)]
    assert config['training'] == 'none' and config['threshold'] == .5 and config['matching_radius_px'] == 3
    assert config['decoders'] == ['original', 'mean_probability', 'mean_logit'] and config['fill'] == 'white'
