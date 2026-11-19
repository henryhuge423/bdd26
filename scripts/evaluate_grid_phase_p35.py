"""P35: frozen P22 detectors evaluated at sixteen alignments to the 16-pixel token grid.

Every image is translated by each (rows, columns) pair of PHASE_STEPS x PHASE_STEPS
with white fill, passed through the frozen detector, and the score map is moved
back to the original coordinates. Nothing is trained.

Decoders, all with the original threshold, peak rule and matching rule:

* original         - the map of phase (0, 0): the frozen detector as trained and stored;
* mean_probability - mean of the realigned probabilities over the phases valid at a pixel;
* mean_logit       - sigmoid of the mean of the realigned logits over the valid phases.

Each single phase is also decoded alone, which gives, for every annotated
nucleus, the number of alignments at which it is matched.

Each invocation handles one detector. Machine paths belong in the ignored JSON
manifest, not in this file.
"""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from proposal_p3_utils import decode_peaks, score_points, split_indices, aggregate, admission
from train_proposal_p21 import match_points
from train_center_offset_p34 import (sha, write_json, write_csv, model_key, shift_array, state_fingerprint,
                                     frozen_selection, totals, COUNT_KEYS)

EXPERIMENT = 'P35'
PHASE_STEPS = (-8, -4, 0, 4)   # per axis; one 16-pixel token period sampled every four pixels
PHASES = tuple((dy, dx) for dy in PHASE_STEPS for dx in PHASE_STEPS)
ORIGINAL = PHASES.index((0, 0))
ENSEMBLES = ('mean_probability', 'mean_logit')
ROLES = ('development', 'training')
CENTER_RADIUS = 3              # matching radius of the project, native pixels


def run_config(device='cpu'):
    return dict(experiment=EXPERIMENT, phases=[list(p) for p in PHASES], fill='white',
                decoders=['original', *ENSEMBLES], single_phase_decoding=True, threshold=.5,
                matching_radius_px=CENTER_RADIUS, roles=list(ROLES), detector='frozen', training='none',
                precision='cpu_float32' if device == 'cpu' else 'cuda_bfloat16_autocast')


def phase_maps(model, image, device):
    """Realigned logits (phases, H, W) and validity (phases, H, W) of one image."""
    image = np.asarray(image)
    height, width = image.shape[:2]
    x = torch.from_numpy(np.stack([shift_array(image, dy, dx, 255) for dy, dx in PHASES]))
    x = x.permute(0, 3, 1, 2).float().to(device) / 255
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        moved = model(x).float().cpu().numpy()
    if moved.shape != (len(PHASES), height, width) or not np.isfinite(moved).all():
        raise FloatingPointError('unexpected or nonfinite detector output')
    logits = np.zeros_like(moved)
    valid = np.zeros(moved.shape, dtype=bool)
    for k, (dy, dx) in enumerate(PHASES):
        logits[k] = shift_array(moved[k], -dy, -dx, 0)
        valid[k] = shift_array(np.ones((height, width), dtype=bool), -dy, -dx, False)
    return logits, valid


def probability(logits):
    return 1. / (1. + np.exp(-np.asarray(logits, dtype=np.float64)))


def ensemble_maps(logits, valid):
    """Probability map of every decoder; pixels are averaged over their valid phases only."""
    count = valid.sum(0)
    assert count.min() >= 1, 'phase (0, 0) is valid everywhere'
    prob = probability(logits) * valid
    return dict(original=prob[ORIGINAL], mean_probability=prob.sum(0) / count,
                mean_logit=probability((logits * valid).sum(0) / count))


def inspect_image(gt, logits, valid, key, role, image):
    """Per-image counts of every decoder and of every single phase; per-object records."""
    maps = ensemble_maps(logits, valid)
    single = probability(logits) * valid
    points = {name: decode_peaks(m) for name, m in maps.items()}
    assert points['original'] == decode_peaks(single[ORIGINAL])
    scores = {name: score_points(v, gt) for name, v in points.items()}
    hits = {name: {int(g) for g, _ in match_points(v, gt)} for name, v in points.items()}
    phase_scores, phase_hits = [], []
    for k in range(len(PHASES)):
        found = decode_peaks(single[k])
        phase_scores.append(score_points(found, gt))
        phase_hits.append({int(g) for g, _ in match_points(found, gt)})
    assert phase_scores[ORIGINAL] == scores['original'] and phase_hits[ORIGINAL] == hits['original']
    height, width = logits.shape[1:]
    objects = []
    for index, (cls, (y, x), interior, area) in enumerate(gt):
        window = (slice(max(0, y - CENTER_RADIUS), min(height, y + CENTER_RADIUS + 1)),
                  slice(max(0, x - CENTER_RADIUS), min(width, x + CENTER_RADIUS + 1)))
        near = single[(slice(None),) + window].reshape(len(PHASES), -1).max(1)
        usable = valid[:, y, x]
        matched = [index in h for h in phase_hits]
        row = dict(model=key, role=role, image=int(image), gt_index=index, cls=int(cls), interior=bool(interior),
                   area=int(area), center_y=int(y), center_x=int(x), valid_phases=int(usable.sum()),
                   phases_matched=int(sum(matched)), matched_original=index in hits['original'],
                   matched_other_phases=int(sum(matched) - matched[ORIGINAL]),
                   near_center_score_original=float(near[ORIGINAL]),
                   near_center_score_phase_mean=float(near[usable].mean()),
                   near_center_score_phase_min=float(near[usable].min()),
                   near_center_score_phase_max=float(near[usable].max()),
                   phases_near_center_above_threshold=int((near[usable] >= .5).sum()))
        for name in ENSEMBLES:
            row[f'matched_{name}'] = index in hits[name]
            row[f'near_center_score_{name}'] = float(maps[name][window].max())
        objects.append(row)
    return scores, phase_scores, objects, maps, points


def render_phases(image, gt, maps, points, objects, path, title):
    """Large panels: annotation with alignment counts, original decoder, both ensemble decoders."""
    from PIL import Image, ImageDraw, ImageFont
    panels = [('Annotated centers: matched at k of 16 alignments', None), ('Original decoder (trained alignment)', 'original'),
              ('Mean probability over 16 alignments', 'mean_probability'), ('Mean logit over 16 alignments', 'mean_logit')]
    side, gap, left, top = 640, 36, 28, 150
    canvas = Image.new('RGB', (len(panels) * side + (len(panels) - 1) * gap + 2 * left, top + side + 110), 'white')
    draw = ImageDraw.Draw(canvas)
    scale = side / image.shape[0]
    font = ImageFont.load_default()

    def label(position, text, size=2):
        box = font.getbbox(text)
        tile = Image.new('RGB', (box[2] - box[0] + 4, box[3] - box[1] + 4), 'white')
        ImageDraw.Draw(tile).text((2 - box[0], 2 - box[1]), text, fill='black', font=font)
        canvas.paste(tile.resize((tile.width * size, tile.height * size), Image.Resampling.NEAREST), position)

    def xy(col, pos):
        return left + col * (side + gap) + (pos[1] + .5) * scale, top + (pos[0] + .5) * scale

    def ring(col, pos, color, radius=8):
        cx, cy = xy(col, pos)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=3)

    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((side, side), Image.Resampling.NEAREST)
    counts = []
    for col, (text, name) in enumerate(panels):
        canvas.paste(rgb, (left + col * (side + gap), top))
        label((left + col * (side + gap), 112), text)
        if name is None:
            for (cls, center, interior, _), row in zip(gt, objects):
                k = row['phases_matched']
                ring(0, center, '#44cc44' if k >= 12 else '#ffbf00' if k >= 4 else '#ff4444')
                cx, cy = xy(0, center)
                draw.text((cx + 9, cy - 6), str(k), fill='#ffffff' if cls == 4 and interior else '#00e5ff', font=font)
            continue
        pairs = match_points(points[name], gt)
        hit, used = {int(g) for g, _ in pairs}, {int(p) for _, p in pairs}
        for index, (_, center, _, _) in enumerate(gt):
            ring(col, center, '#44cc44' if index in hit else '#ffbf00')
        for j, pos in enumerate(points[name]):
            cx, cy = xy(col, pos)
            color = '#44cc44' if j in used else '#ff4444'
            if j in used:
                draw.line((cx - 7, cy, cx + 7, cy), fill=color, width=3)
                draw.line((cx, cy - 7, cx, cy + 7), fill=color, width=3)
            else:
                draw.line((cx - 7, cy - 7, cx + 7, cy + 7), fill=color, width=3)
                draw.line((cx - 7, cy + 7, cx + 7, cy - 7), fill=color, width=3)
        counts.append(f'{text.lower()}: {len(points[name])} points, {len(pairs)} matched')
    label((left, 16), title, 3)
    label((left, 64), 'First panel: ring color and number = alignments (of 16) at which the nucleus is matched; '
                      'green 12-16, yellow 4-11, red 0-3; white number = interior Dead nucleus.')
    label((left, top + side + 14), 'Other panels: rings = annotated centers (green matched, yellow missed); '
                                   'green plus = matched point, red cross = extra point. Matching radius = 3 pixels.')
    label((left, top + side + 56), f'Annotated nuclei = {len(gt)}; ' + '; '.join(counts))
    canvas.save(path, format='PNG', dpi=(150, 150))


def phase_distribution(objects):
    """How many nuclei are matched at 0, 1, ..., 16 alignments (nuclei valid at every alignment)."""
    groups = {'all': objects, 'interior_dead': [o for o in objects if o['cls'] == 4 and o['interior']]}
    result = {}
    for group, rows in groups.items():
        rows = [o for o in rows if o['valid_phases'] == len(PHASES)]
        histogram = np.bincount([o['phases_matched'] for o in rows], minlength=len(PHASES) + 1)
        rate = np.asarray([o['phases_matched'] / len(PHASES) for o in rows], dtype=float)
        mean = float(rate.mean()) if len(rows) else None
        # Share of the match variance that belongs to the nucleus (between objects) rather than to the alignment.
        between = float(rate.var() / (mean * (1 - mean))) if len(rows) and 0 < mean < 1 else None
        result[group] = dict(objects=len(rows), histogram=histogram.tolist(), mean_match_rate=mean,
                             never=int(histogram[0]), always=int(histogram[-1]),
                             sometimes=int(histogram[1:-1].sum()), between_object_variance_share=between,
                             matched_original=int(sum(o['matched_original'] for o in rows)),
                             matched_at_any_alignment=int(sum(o['phases_matched'] > 0 for o in rows)),
                             **{f'matched_{name}': int(sum(o[f'matched_{name}'] for o in rows)) for name in ENSEMBLES})
    return result


def evaluate_role(model, source, ids, device, key, role, historical=None, save_images=(), out=None):
    rows = {name: [] for name in ('original', *ENSEMBLES)}
    phase_rows = [[] for _ in PHASES]
    objects, differing, forward_seconds = [], 0, 0.
    stored = {k: 0 for k in COUNT_KEYS}
    for image in ids:
        tick = time.monotonic()
        logits, valid = phase_maps(model, source.images[image], device)
        forward_seconds += time.monotonic() - tick
        gt = source.gt[image]
        scores, phase_scores, records, maps, points = inspect_image(gt, logits, valid, key, role, image)
        identity = dict(image=int(image), tissue=str(source.tissue[image]))
        for name in rows:
            rows[name].append(dict(identity, **scores[name]))
        for k, s in enumerate(phase_scores):
            phase_rows[k].append(dict(identity, **s))
        objects.extend(records)
        if historical is not None:
            differing += any(historical[image][k] != scores['original'][k] for k in COUNT_KEYS)
            for k in COUNT_KEYS:
                stored[k] += historical[image][k]
        if image in save_images:
            base = f'{key}_{role}_{image:04d}'
            np.savez_compressed(out / 'maps' / (base + '.npz'), logits=logits.astype(np.float16), valid=valid)
            render_phases(np.asarray(source.images[image]), gt, maps, points, records, out / 'figures' / (base + '.png'),
                          f'{key}: {role} image {image}, {source.tissue[image]}')
    comparison = None
    if historical is not None:
        comparison = dict(images=len(ids), images_with_different_counts=int(differing), stored_counts=stored,
                          recomputed_counts=totals(rows['original']))
    return rows, phase_rows, objects, dict(comparison=comparison, forward_seconds=forward_seconds)


def run_detector(key, model, fold, audit, selection, out, device, historical=None):
    """Evaluate one frozen detector at every alignment on both roles."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    for folder in ('maps', 'figures'):
        (out / folder).mkdir()
    started = time.monotonic()
    model.requires_grad_(False).eval()
    frozen_before = state_fingerprint(model)
    sources = dict(development=audit, training=selection['training_images'])
    figures = dict(development=set(selection['development_images']), training=set(selection['training_figures']))
    results, roles = {}, {}
    for role in ROLES:
        ids = sources[role]
        stored = None if historical is None or role not in historical else historical[role]
        rows, phase_rows, objects, info = evaluate_role(model, fold, ids, device, key, role, stored, figures[role], out)
        results[role] = rows
        for name, value in rows.items():
            write_json(out / f'{role}_{name}.json', value)
        write_json(out / f'{role}_single_phases.json', dict(phases=[list(p) for p in PHASES], rows=phase_rows))
        write_csv(out / f'{role}_objects.csv', objects)
        roles[role] = dict(
            images=len(ids),
            decoders={name: dict(counts=totals(value), rates=aggregate(value)) for name, value in rows.items()},
            single_phases=[dict(phase=list(p), counts=totals(r), rates=aggregate(r)) for p, r in zip(PHASES, phase_rows)],
            alignment_distribution=phase_distribution(objects), stored_original_comparison=info['comparison'],
            forward_seconds_per_image=info['forward_seconds'] / max(1, len(ids)))
        print(json.dumps(dict(event='role', model=key, role=role, seconds=time.monotonic() - started,
                              decoders={name: v['counts'] for name, v in roles[role]['decoders'].items()})), flush=True)
    assert state_fingerprint(model) == frozen_before, 'frozen detector changed'
    summary = dict(model=key, config=run_config(device.type), roles=roles, frozen_detector_fingerprint=frozen_before,
                   frozen_detector_unchanged=True, elapsed_seconds=time.monotonic() - started)
    write_json(out / 'summary.json', summary)
    return results, summary


def paired_decisions(results):
    """Original two-seed rule per detector arm: original decoder versus each ensemble decoder."""
    decisions = {}
    for arm in sorted({a for a, _ in results}):
        for name in ENSEMBLES:
            data = {str(seed): dict(F1=results[(a, seed)]['original'], T1=results[(a, seed)][name])
                    for (a, seed) in results if a == arm}
            decisions[f'{arm}:{name}'] = admission(data)
    return decisions


def summarize(root, runs):
    """Combine the finished detectors from their small result files."""
    root = Path(root)
    results, summaries = {}, []
    for run in runs:
        folder = root / model_key(run)
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['exit_code'] != 0 or complete['phases'] != len(PHASES) or complete['model'] != model_key(run):
            raise ValueError(f'{model_key(run)} is not a completed {EXPERIMENT} run')
        results[(run['arm'], run['seed'])] = {name: json.loads((folder / f'development_{name}.json').read_text())
                                              for name in ('original', *ENSEMBLES)}
        summaries.append(json.loads((folder / 'summary.json').read_text()))
    combined = dict(experiment=EXPERIMENT, detectors=summaries, decisions=paired_decisions(results))
    write_json(root / 'summary.json', combined)
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-index', type=int, help='evaluate this detector of the manifest')
    mode.add_argument('--summarize', action='store_true', help='combine the finished detectors')
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.manifest).read_text())
    if args.summarize:
        combined = summarize(args.out, spec['runs'])
        print(json.dumps({key: value['pass'] for key, value in combined['decisions'].items()}, indent=1))
        return
    from train_proposal_p3 import Fold, WEIGHT_SHA, DATA_SHA, load_adapted
    from p5_contracts import TISSUE_SHA, validate_cohort
    from diagnose_proposal_center_maps import validate_run
    started = time.monotonic()
    device = torch.device(spec['device'])
    assert spec['experiment'] == EXPERIMENT and spec['fold'] == 1 and spec['config'] == run_config(device.type)
    assert [(r['experiment'], r['arm'], r['seed']) for r in spec['runs']] == \
        [('P22', 'C', 19), ('P22', 'C', 1), ('P22', 'M', 19), ('P22', 'M', 1)]
    run = spec['runs'][args.run_index]
    key = model_key(run)
    assert torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.23.5'
    if device.type == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    else:
        available = len(os.sched_getaffinity(0))
        assert available >= spec['threads'], f'{available} usable CPUs for {spec["threads"]} threads'
    torch.set_num_threads(spec['threads'])
    data_root, weights = Path(spec['data']), Path(spec['weights'])
    inputs = {rel: sha(data_root / rel) for rel in DATA_SHA if rel.startswith('fold1/')}
    assert inputs == {rel: value for rel, value in DATA_SHA.items() if rel.startswith('fold1/')}
    assert sha(data_root / 'fold1/tissue.npy') == TISSUE_SHA
    weights_digest = sha(weights)
    assert weights_digest == WEIGHT_SHA
    fold = Fold(data_root, 1)
    fit, audit = split_indices(fold.tissue, [any(g[0] == 4 and g[2] for g in row) for row in fold.gt])
    cohort = validate_cohort(fit, audit)
    assert len(fit) == 2134 and len(audit) == 522
    selection = frozen_selection(fold, fit, audit)
    assert selection == spec['selection'], 'annotation-only image selection changed after freeze'
    module = importlib.import_module('train_proposal_p22')
    root = Path(run['directory'])
    manifest, preservation = validate_run(run, module, weights_digest, cohort)
    assert manifest['input_sha256'] == dict(inputs, **{'fold1/tissue.npy': TISSUE_SHA})
    checkpoint = torch.load(root / 'last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['completed_epoch'] == 130 and checkpoint['config'] == manifest['config']
    model = module.Proposal(str(weights), True, run['arm'])
    load_adapted(model, checkpoint['model'])
    del checkpoint
    model.requires_grad_(False).to(device).eval()
    historical = {role: {r['image']: r for r in json.loads((root / name).read_text())}
                  for role, name in (('development', 'evaluation.json'), ('training', 'fit.json'))}
    out = Path(args.out) / key
    print(json.dumps(dict(event='run_start', model=key, device=device.type, threads=spec['threads'],
                          usable_cpus=len(os.sched_getaffinity(0)), setup_seconds=time.monotonic() - started)),
          flush=True)
    _, summary = run_detector(key, model, fold, audit, selection, out, device, historical)
    write_json(out / 'provenance.json', dict(
        model=key, last_checkpoint_sha256=preservation['last.pth']['sha256'],
        last_checkpoint_bytes=preservation['last.pth']['bytes'], cohort_sha256=cohort, weights_sha256=weights_digest,
        input_sha256=inputs, selection=selection, script_sha256=sha(__file__), torch=torch.__version__,
        numpy=np.__version__, threads=spec['threads']))
    payload = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'preservation_manifest.json', payload)
    write_json(out / 'complete.json', dict(experiment=EXPERIMENT, model=key, phases=len(PHASES), exit_code=0,
                                           elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(event='run_complete', model=key,
                          development={n: v['counts'] for n, v in summary['roles']['development']['decoders'].items()},
                          elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
