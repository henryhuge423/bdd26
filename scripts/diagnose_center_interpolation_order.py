"""Fixed CPU counterfactual on completed D32 maps: sigmoid before interpolation.

Analyze the original eleven development images and label-selected training
maps for all twelve models. No network inference, optimization or threshold
selection is performed. Results describe this selected-image diagnostic.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

import diagnose_proposal_center_maps as d
from proposal_p3_utils import gt_objects


def interpolate_scores(coarse):
    """Apply sigmoid to FP32 coarse logits, then the original bilinear operator."""
    tensor = torch.from_numpy(np.asarray(coarse, dtype=np.float32))[None, None]
    return F.interpolate(tensor.sigmoid(), scale_factor=2, mode='bilinear',
                         align_corners=False)[0, 0].numpy()


def original_scores(fine):
    return torch.from_numpy(np.asarray(fine, dtype=np.float32)).sigmoid().numpy()


def point_records(points, gt, inst, scores, image, condition):
    pairs = d.match_points(points, gt)
    assigned = {int(p): int(g) for g, p in pairs}
    return [dict(image=int(image), condition=condition, point_index=i, y=y, x=x,
                 score=float(scores[y, x]), assigned_gt=assigned.get(i),
                 mask_instance_id=int(inst[y, x]), matched=i in assigned)
            for i, (y, x) in enumerate(points)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    source, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.23.5'
    torch.set_num_threads(4)
    complete = json.loads((source/'complete.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    assert complete['models'] == 12 and complete['exit_code'] == 0
    assert len(summary['runs']) == 12
    expected = {f'P{number}_{arm}_s{seed}' for number, arms in ((22, 'CM'), (29, 'CB'), (31, 'CG'))
                for arm in arms for seed in (19, 1)}
    assert {r['model'] for r in summary['runs']} == expected
    assert all(r['hooks_bit_equal'] and r['interpolation_bit_equal'] for r in summary['runs'])
    manifest = json.loads((source/'preservation_manifest.json').read_text())
    input_hashes = {}

    def verify(rel):
        p = source/rel
        e = manifest[rel]
        assert p.stat().st_size == e['bytes'] and d.sha(p) == e['sha256'], rel
        input_hashes[rel] = e
        return p

    verify('summary.json')
    arrays = np.load(verify('selected_native_arrays.npz'))
    selection = json.loads(verify('selection.json').read_text())
    indices = {int(i): j for j, i in enumerate(arrays['ids'])}
    training = sorted(set(selection['crowded_small_training']) | set(selection['training_examples']))
    assert len(selection['audit']) == 11
    for name in ('records', 'maps', 'figures'):
        (out/name).mkdir()
    shutil.copyfile(verify('selected_native_arrays.npz'), out/'selected_native_arrays.npz')
    d.write_json(out/'selection.json', selection)
    models, all_objects = [], []
    for model in summary['runs']:
        key = model['model']
        print(json.dumps(dict(event='run_start', model=key)), flush=True)
        model_objects, paired_objects, points_table, role_summaries = [], [], [], {}
        for role, ids in (('development', selection['audit']), ('training', training)):
            baseline_rows = json.loads(verify(f'records/{key}_{role}.json').read_text())
            expected = {r['image']: r for r in baseline_rows}
            scores_by_condition = {'original': [], 'sigmoid_before_interpolation': []}
            for image in ids:
                native = indices[image]
                rgb, inst, types = arrays['images'][native], arrays['inst'][native], arrays['types'][native]
                gt = gt_objects(inst, types)
                name = f'{key}_{role}_{image:04d}'
                maps = np.load(verify('maps/'+name+'.npz'))
                coarse, fine = maps['coarse_logits'], maps['fine_logits']
                original, alternative = original_scores(fine), interpolate_scores(coarse)
                observations = {}
                for condition, probability in (('original', original), ('sigmoid_before_interpolation', alternative)):
                    trace, points = d.trace_peaks(probability)
                    score = d.score_points(points, gt)
                    if condition == 'original':
                        assert np.array_equal(trace['retained_ids'], maps['retained_peak_ids'])
                        assert all(expected[image][k] == v for k, v in score.items()), (key, image, score)
                    scores_by_condition[condition].append(dict(image=int(image), **score))
                    logits = fine if condition == 'original' else torch.logit(torch.from_numpy(probability).double().clamp(1e-12, 1-1e-12)).numpy()
                    rows, pairs = d.inspect_objects(coarse, logits, inst, gt, points, trace, image, key, role)
                    for row in rows:
                        row['condition'] = condition
                    model_objects.extend(rows)
                    observations[condition] = {r['gt_index']: r for r in rows}
                    points_table.extend(point_records(points, gt, inst, probability, image, condition))
                    if condition != 'original':
                        np.savez_compressed(out/'maps'/(name+'.npz'), original_probability=original,
                            sigmoid_before_interpolation_probability=alternative,
                            retained_peak_ids=trace['retained_ids'])
                        if role == 'development' or image in selection['training_examples']:
                            d.render_panel(rgb, inst, gt, points, coarse, logits,
                                out/'figures'/(name+'.png'),
                                f'{key}: {role} image {image}; sigmoid before interpolation')
                assert observations['original'].keys() == observations['sigmoid_before_interpolation'].keys()
                for index, old in observations['original'].items():
                    new = observations['sigmoid_before_interpolation'][index]
                    paired_objects.append(dict(model=key, role=role, image=int(image), gt_index=index,
                        instance_id=old['instance_id'], area=old['area'], neighbors12=old['neighbors12'],
                        original_matched=old['matched'], alternative_matched=new['matched'],
                        gained=not old['matched'] and new['matched'], lost=old['matched'] and not new['matched'],
                        original_max_logit_r3=old['max_logit_r3'], alternative_max_logit_r3=new['max_logit_r3'],
                        original_interpolation_flag=old['coarse_positive_fine_subthreshold'],
                        original_displaced_in_mask=old['in_mask_displaced_points'] > 0))
            d.write_json(out/'records'/f'{key}_{role}.json', scores_by_condition)
            role_summaries[role] = {condition: d.aggregate(rows) for condition, rows in scores_by_condition.items()}
        d.write_json(out/'records'/f'{key}_objects.json', model_objects)
        d.write_csv(out/'records'/f'{key}_points.csv', points_table)
        all_objects.extend(paired_objects)
        provenance = dict(model=key, source_last_checkpoint_sha256=model['last_checkpoint_sha256'],
            original_count_replay_equal=True, selected_image_results=role_summaries,
            elapsed_seconds=time.monotonic()-started)
        models.append(provenance)
        print(json.dumps(dict(event='run_complete', **provenance)), flush=True)
    d.write_csv(out/'object_transitions.csv', all_objects)
    d.write_json(out/'summary.json', dict(experiment='D33', models=models, input_hashes=input_hashes,
        conditions=['original', 'sigmoid_before_interpolation'], selection=selection,
        scope='Eleven fixed development images with all 145 interior Dead nuclei, plus label-selected training maps.',
        frozen_decoder=dict(threshold=.5, local_maximum_size=3, cap=512, unique_radius_px=3),
        network_inference=False, optimizer_updates=0,
        notes=['The two conditions differ only in the order of sigmoid and final interpolation.',
               'Selected-image precision has its own denominator; original full-development metrics remain fixed.',
               'This is an intervention on completed model outputs, chosen after D32.']))
    payload = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=d.sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()}
    d.write_json(out/'preservation_manifest.json', payload)
    d.write_json(out/'complete.json', dict(experiment='D33', models=12, exit_code=0,
        elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
