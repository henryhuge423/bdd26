"""Measure frozen center detectors without training or changing their decoder.

Use a JSON manifest with twelve run directories: P22 C/M, P29 C/B and P31
C/G, each for seeds 19 and 1. Machine paths belong in that ignored manifest.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import maximum_filter
import torch
import torch.nn.functional as F

from proposal_p3_utils import decode_peaks, score_points, split_indices, aggregate
from train_proposal_p3 import Fold, WEIGHT_SHA, DATA_SHA, load_adapted
from p5_contracts import TISSUE_SHA, validate_cohort
from train_proposal_p21 import match_points, select_examples


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interpolation_contributors(y, x, height=128, width=128):
    """Coarse cells and combined weights for one fine-grid pixel (edge clamp)."""
    cy, cx = (y + .5) / 2 - .5, (x + .5) / 2 - .5
    iy, ix = math.floor(cy), math.floor(cx)
    wy, wx = cy - iy, cx - ix
    cells = {}
    for dy, ay in ((0, 1 - wy), (1, wy)):
        for dx, ax in ((0, 1 - wx), (1, wx)):
            key = (min(height - 1, max(0, iy + dy)),
                   min(width - 1, max(0, ix + dx)))
            cells[key] = cells.get(key, 0.) + ay * ax
    return cells


def shift_image(image, dy, dx):
    """Integer translation with white fill; no wrap and no label reassignment."""
    result = np.full_like(image, 255)
    height, width = image.shape[:2]
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return result


def trace_peaks(prob):
    """Record all local maxima, threshold survivors, and original top-512 order."""
    prob = np.asarray(prob, dtype=np.float64)
    ids = np.flatnonzero(prob == maximum_filter(prob, size=3, mode='constant', cval=-1))
    order = ids[np.lexsort((ids, -prob.ravel()[ids]))]
    eligible = order[prob.ravel()[order] >= .5]
    selected = eligible[:512]
    points = [(int(v // prob.shape[1]), int(v % prob.shape[1])) for v in selected]
    if points != decode_peaks(prob):
        raise AssertionError('peak trace differs from the frozen decoder')
    return dict(local_ids=order, threshold_ids=eligible, retained_ids=selected), points


def region_max(array, y, x, radius):
    yy, xx = np.mgrid[max(0, y-radius):min(array.shape[0], y+radius+1),
                      max(0, x-radius):min(array.shape[1], x+radius+1)]
    keep = (yy-y)**2 + (xx-x)**2 <= radius**2
    sy, sx = yy[keep], xx[keep]
    values = array[sy, sx]
    index = int(values.argmax())
    return float(values[index]), (int(sy[index]), int(sx[index])), (sy, sx)


def inspect_objects(coarse, fine, inst, gt, points, trace, image, model_key, role):
    prob = 1 / (1 + np.exp(-np.clip(fine.astype(np.float64), -700, 700)))
    pairs = match_points(points, gt)
    hits = {int(g) for g, _ in pairs}
    instance_ids = np.unique(inst)
    instance_ids = instance_ids[instance_ids != 0]
    assert len(instance_ids) == len(gt)
    peak_set = set(map(int, trace['local_ids']))
    eligible_set = set(map(int, trace['threshold_ids']))
    retained_set = set(map(int, trace['retained_ids']))
    records = []
    for index, (cls, (y, x), interior, area) in enumerate(gt):
        if cls != 4 or not interior:
            continue
        r3, best, (ys, xs) = region_max(fine, y, x, 3)
        r6, _, _ = region_max(fine, y, x, 6)
        r12, _, _ = region_max(fine, y, x, 12)
        region_ids = set(map(int, ys * fine.shape[1] + xs))
        support = set()
        for fy, fx in zip(ys, xs):
            support.update(interpolation_contributors(int(fy), int(fx)))
        coarse_max = max(float(coarse[cy, cx]) for cy, cx in support)
        oid = int(instance_ids[index])
        mask = inst == oid
        my, mx = np.nonzero(mask)
        mask_index = int(fine[my, mx].argmax())
        mask_y, mask_x = int(my[mask_index]), int(mx[mask_index])
        own = [(py, px) for py, px in points if inst[py, px] == oid]
        displaced = [p for p in own if (p[0]-y)**2 + (p[1]-x)**2 > 9]
        eligible_count = len(region_ids & eligible_set)
        retained_count = len(region_ids & retained_set)
        pos = np.asarray([g[1] for g in gt], dtype=float)
        distances = ((pos - (y, x))**2).sum(1)
        distances[index] = np.inf
        records.append(dict(model=model_key, role=role, image=int(image), gt_index=index,
            instance_id=oid, center_y=y, center_x=x, area=area,
            neighbors12=int((distances <= 144).sum()), matched=index in hits,
            center_logit=float(fine[y, x]), center_score=float(prob[y, x]),
            max_logit_r3=r3, max_logit_r6=r6, max_logit_r12=r12,
            best_r3_y=best[0], best_r3_x=best[1],
            coarse_support_max_logit_r3=coarse_max,
            coarse_positive_fine_subthreshold=coarse_max >= 0 and r3 < 0,
            local_maxima_r3=len(region_ids & peak_set),
            threshold_peaks_r3=eligible_count, retained_peaks_r3=retained_count,
            top512_removed_nearby_peak=eligible_count > retained_count,
            strong_r3_without_eligible_peak=r3 >= 0 and eligible_count == 0,
            retained_nearby_but_unmatched=retained_count > 0 and index not in hits,
            mask_max_logit=float(fine[mask_y, mask_x]), mask_max_y=mask_y, mask_max_x=mask_x,
            mask_max_distance_px=math.hypot(mask_y-y, mask_x-x),
            in_mask_displaced_points=len(displaced),
            nearest_in_mask_point_px=min((math.hypot(py-y, px-x) for py, px in own), default=None)))
    return records, pairs


class Capture:
    """Read-only hooks; never replace forward values."""
    def __init__(self, model):
        self.coarse = None
        self.features = None
        self.handles = [model.head.out.register_forward_hook(self.output),
                        model.head.register_forward_pre_hook(self.inputs)]

    def output(self, module, args, value):
        self.coarse = value.detach()

    def inputs(self, module, args):
        self.features = [f.detach() for f in args[0]]

    def close(self):
        for handle in self.handles:
            handle.remove()


def validate_run(run, module, weights_digest, cohort):
    root = Path(run['directory'])
    manifest = json.loads((root/'manifest.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    preservation = json.loads((root/'preservation_manifest.json').read_text())
    assert complete['epochs'] == 130 and complete['exit_code'] == 0
    assert manifest['weights_sha256'] == weights_digest == WEIGHT_SHA
    assert manifest['cohort_sha256'] == cohort
    assert manifest['config'] == module.run_config(run['arm'], run['seed'], 4)
    epochs = [json.loads(row)['epoch'] for row in (root/'loss.jsonl').read_text().splitlines()]
    assert epochs == list(range(1, 131))
    for rel in ('last.pth', 'manifest.json', 'evaluation.json', 'fit.json', 'complete.json', 'loss.jsonl'):
        file = root/rel
        assert file.stat().st_size == preservation[rel]['bytes'], str(file)
        assert sha(file) == preservation[rel]['sha256'], str(file)
    return manifest, preservation


def render_panel(image, inst, gt, points, coarse, fine, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    axes[0].imshow(image)
    for cls, (y, x), interior, _ in gt:
        axes[0].plot(x, y, 'o', ms=5, mfc='none', mec='#ffbf00' if cls == 4 and interior else '#00bbdd')
    axes[0].set_title('Image and annotated centers')
    axes[1].imshow(1/(1+np.exp(-np.clip(coarse.astype(float), -700, 700))), vmin=0, vmax=1, cmap='magma')
    axes[1].set_title('128-grid score before interpolation')
    axes[2].imshow(1/(1+np.exp(-np.clip(fine.astype(float), -700, 700))), vmin=0, vmax=1, cmap='magma')
    axes[2].set_title('256-grid score after interpolation')
    axes[3].imshow(image)
    pairs = match_points(points, gt)
    hits, assigned = {int(a) for a, _ in pairs}, {int(b) for _, b in pairs}
    for index, (_, (y, x), _, _) in enumerate(gt):
        axes[3].plot(x, y, 'o', ms=6, mfc='none', mec='#44cc44' if index in hits else '#ffbf00')
    for index, (y, x) in enumerate(points):
        axes[3].plot(x, y, '+' if index in assigned else 'x', ms=5, color='#44cc44' if index in assigned else '#ff4444')
    axes[3].set_title('Unique matches, misses and extra points')
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title + '\nFirst panel: yellow = interior Dead; last panel: yellow = missed, green = matched, red = extra', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert spec['experiment'] == 'D32' and spec['fold'] == 1 and spec['batch_size'] == 4
    expected_runs = {(f'P{number}', arm, seed) for number, arms in ((22, 'CM'), (29, 'CB'), (31, 'CG'))
                     for arm in arms for seed in (19, 1)}
    assert len(spec['runs']) == 12
    assert {(r['experiment'], r['arm'], r['seed']) for r in spec['runs']} == expected_runs
    assert torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.23.5'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info()[0] >= 20*1024**3
    torch.set_num_threads(4)
    data_root, weights = Path(spec['data']), Path(spec['weights'])
    inputs = {rel: sha(data_root/rel) for rel in DATA_SHA if rel.startswith('fold1/')}
    assert inputs == {rel: value for rel, value in DATA_SHA.items() if rel.startswith('fold1/')}
    assert sha(data_root/'fold1/tissue.npy') == TISSUE_SHA
    weights_digest = sha(weights)
    assert weights_digest == WEIGHT_SHA
    fold = Fold(data_root, 1)
    inst_type = np.load(data_root/'fold1/type.npy', mmap_mode='r')
    fit, audit = split_indices(fold.tissue, [any(g[0] == 4 and g[2] for g in row) for row in fold.gt])
    cohort = validate_cohort(fit, audit)
    assert len(fit) == 2134 and len(audit) == 522
    dead_audit = [i for i in audit if any(g[0] == 4 and g[2] for g in fold.gt[i])]
    assert len(dead_audit) == 11 and sum(sum(g[0] == 4 and g[2] for g in fold.gt[i]) for i in audit) == 145
    crowded_fit = []
    for i in fit:
        centers = np.asarray([g[1] for g in fold.gt[i]], dtype=float)
        for j, g in enumerate(fold.gt[i]):
            if g[0] == 4 and g[2] and g[3] < 67:
                distance = ((centers - g[1])**2).sum(1)
                distance[j] = np.inf
                if np.any(distance <= 144):
                    crowded_fit.append(i)
                    break
    assert len(crowded_fit) == 13
    fit_dead = [i for i in fit if any(g[0] == 4 and g[2] for g in fold.gt[i])]
    examples = set(select_examples(fit, fold.gt, 4))
    save_images = set(dead_audit) | set(crowded_fit) | examples
    selected = dict(audit=dead_audit, crowded_small_training=crowded_fit,
                    training_examples=sorted(examples), training_dead_images=fit_dead)
    assert selected == spec['selection'], 'GT-only selection changed after freeze'
    for folder in ('maps', 'figures', 'features', 'records'):
        (out/folder).mkdir()
    np.savez_compressed(out/'selected_native_arrays.npz', ids=np.asarray(sorted(save_images)),
                        images=np.asarray(fold.images[sorted(save_images)]),
                        inst=np.asarray(fold.inst[sorted(save_images)]),
                        types=np.asarray(inst_type[sorted(save_images)]))
    write_json(out/'selection.json', selected)
    all_objects, shift_rows, summaries = [], [], []
    for run in spec['runs']:
        key = f"{run['experiment']}_{run['arm']}_s{run['seed']}"
        print(json.dumps(dict(event='run_start', model=key)), flush=True)
        module = importlib.import_module('train_proposal_' + run['experiment'].lower())
        root = Path(run['directory'])
        manifest, preservation = validate_run(run, module, weights_digest, cohort)
        assert manifest['input_sha256'] == dict(inputs, **{'fold1/tissue.npy': TISSUE_SHA})
        checkpoint = torch.load(root/'last.pth', map_location='cpu', weights_only=False)
        assert checkpoint['completed_epoch'] == 130 and checkpoint['config'] == manifest['config']
        model = module.Proposal(str(weights), True, run['arm'])
        load_adapted(model, checkpoint['model'])
        del checkpoint
        # Membership was validated by load_adapted before making inference read-only.
        model.requires_grad_(False).cuda().eval()
        source_file = Path(importlib.import_module('nucseg.cellvit.model').__file__).resolve()
        assert source_file.is_relative_to(Path(__file__).resolve().parent.parent/'src')
        test_x = torch.from_numpy(np.asarray(fold.images[audit[:4]]).copy()).permute(0,3,1,2).float().cuda()/255
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            plain = model(test_x)
        capture = Capture(model)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hooked = model(test_x)
        assert torch.equal(plain, hooked), 'hooks changed model output'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            rebuilt = F.interpolate(capture.coarse, size=(256,256), mode='bilinear', align_corners=False)[:,0]
        assert torch.equal(rebuilt, hooked), 'captured output is not the actual final interpolation input'
        assert capture.coarse.dtype == torch.bfloat16 and hooked.dtype == torch.float32
        del plain, hooked, rebuilt, test_x
        rows, training_rows, objects, point_rows = [], [], [], []
        shift_base = {}
        for role, ids in (('development', audit), ('training', fit_dead + sorted(examples - set(fit_dead)))):
            original = {r['image']: r for r in json.loads((root/('evaluation.json' if role == 'development' else 'fit.json')).read_text())}
            for start in range(0, len(ids), 4):
                batch = ids[start:start+4]
                x = torch.from_numpy(np.asarray(fold.images[batch]).copy()).permute(0,3,1,2).float().cuda()/255
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                assert torch.isfinite(logits).all() and torch.isfinite(capture.coarse).all()
                fine_batch = logits.float().cpu().numpy()
                coarse_batch = capture.coarse[:,0].float().cpu().numpy()
                features = [f.float().cpu().numpy() for f in capture.features] if set(batch) & save_images else None
                for index, image in enumerate(batch):
                    fine, coarse = fine_batch[index], coarse_batch[index]
                    prob = torch.from_numpy(fine).sigmoid().numpy()
                    trace, points = trace_peaks(prob)
                    score = score_points(points, fold.gt[image])
                    differences = {name: [original[image][name], score[name]] for name in score if original[image][name] != score[name]}
                    row = dict(image=int(image), tissue=str(fold.tissue[image]), **score,
                               historical_count_equal=not differences, historical_count_differences=differences)
                    (rows if role == 'development' else training_rows).append(row)
                    records, pairs = inspect_objects(coarse, fine, fold.inst[image], fold.gt[image], points, trace, image, key, role)
                    objects.extend(records)
                    if role == 'development':
                        assigned = {int(p): int(g) for g,p in pairs}
                        for point_index, (py,px) in enumerate(points):
                            oid = int(fold.inst[image][py,px])
                            point_rows.append(dict(image=int(image), point_index=point_index, y=py, x=px,
                                score=float(prob[py,px]), assigned_gt=assigned.get(point_index), mask_instance_id=oid,
                                matched=point_index in assigned, inside_mask=oid>0))
                    if image in save_images:
                        basename = f'{key}_{role}_{image:04d}'
                        np.savez_compressed(out/'maps'/(basename+'.npz'), coarse_logits=coarse, fine_logits=fine,
                            local_peak_ids=trace['local_ids'], threshold_peak_ids=trace['threshold_ids'],
                            retained_peak_ids=trace['retained_ids'])
                        np.savez_compressed(out/'features'/(basename+'.npz'), **{f'normalized_tap_{tap}':features[k][index].astype(np.float16) for k,tap in enumerate((6,12,18,24))})
                        if role == 'development' or image in examples:
                            render_panel(fold.images[image],fold.inst[image],fold.gt[image],points,coarse,fine,
                                out/'figures'/(basename+'.png'),f'{key}: {role} image {image}, {fold.tissue[image]}')
                    if image in dead_audit:
                        shift_base[image] = dict(fine=fine.copy(), records=records)
                del x, logits
        # Each shift uses the same four-image grouping for all models. Border-near
        # targets are omitted from shift statistics using a fixed 16-pixel margin.
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
            for start in range(0,len(dead_audit),4):
                batch = dead_audit[start:start+4]
                translated = np.asarray([shift_image(fold.images[i],dy,dx) for i in batch])
                x = torch.from_numpy(translated).permute(0,3,1,2).float().cuda()/255
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    fine_batch = model(x).float().cpu().numpy()
                coarse_batch = capture.coarse[:,0].float().cpu().numpy()
                for index,image in enumerate(batch):
                    shifted = fine_batch[index]
                    np.savez_compressed(out/'maps'/f'{key}_shift_{dy}_{dx}_{image:04d}.npz',
                                        coarse_logits=coarse_batch[index],fine_logits=shifted)
                    for record in shift_base[image]['records']:
                        y,xx = record['center_y'],record['center_x']
                        if not (16 <= y < 240 and 16 <= xx < 240):
                            continue
                        value, peak, _ = region_max(shifted,y+dy,xx+dx,3)
                        shift_rows.append(dict(model=key,image=image,gt_index=record['gt_index'],dy=dy,dx=dx,
                            original_max_logit_r3=record['max_logit_r3'],shifted_max_logit_r3=value,
                            score_threshold_crossing=(value >= 0)!=(record['max_logit_r3'] >= 0),
                            mapped_peak_y=peak[0]-dy,mapped_peak_x=peak[1]-dx,
                            peak_displacement_px=math.hypot(peak[0]-dy-record['best_r3_y'],peak[1]-dx-record['best_r3_x'])))
                del x
        capture.close()
        summary = dict(model=key,source_module_sha256=sha(module.__file__),last_checkpoint_sha256=preservation['last.pth']['sha256'],
            last_checkpoint_bytes=preservation['last.pth']['bytes'],config=manifest['config'],cohort_sha256=cohort,
            hooks_bit_equal=True,interpolation_bit_equal=True,development=aggregate(rows),
            historical_count_mismatch_images=sum(not r['historical_count_equal'] for r in rows),
            selected_training_count_mismatch_images=sum(not r['historical_count_equal'] for r in training_rows),
            inference_peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
        write_json(out/'records'/(key+'_development.json'), rows)
        write_json(out/'records'/(key+'_training.json'),training_rows)
        write_json(out/'records'/(key+'_provenance.json'),summary)
        write_csv(out/'records'/(key+'_points.csv'),point_rows)
        all_objects.extend(objects)
        summaries.append(summary)
        write_json(out/'records'/(key+'_objects.json'),objects)
        print(json.dumps(dict(event='run_complete',**summary)),flush=True)
        del model,capture,shift_base,features
        torch.cuda.empty_cache()
    write_csv(out/'objects.csv',all_objects)
    write_csv(out/'shift_stability.csv',shift_rows)
    write_json(out/'summary.json',dict(experiment='D32',runs=summaries,selection=selected,
        notes=['Coarse support maxima and final maxima have different spatial support.',
               'A coarse-positive/fine-subthreshold flag identifies a measured interpolation pattern.',
               'Feature arrays use float16 storage; actual logits are retained as float32.',
               'Shift results are exploratory and do not alter the original evaluation.',
               'Historical count differences are retained; original published metrics remain authoritative.']))
    payload = {str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'preservation_manifest.json',payload)
    write_json(out/'complete.json',dict(experiment='D32',models=12,exit_code=0,elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
