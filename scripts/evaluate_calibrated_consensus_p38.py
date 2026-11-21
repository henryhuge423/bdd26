"""P38: image-separated quality calibration of preserved low-score point groups.

No detector forward pass. Fit six-feature ridge logistic scores on half of the
P36 training images; select a cutoff on the other half. All eight rules are
written before development records are opened. Evaluate the frozen rules using
the original one-to-one matching and retain every image, including empty ones.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from analyze_alignment_misses_d37 import read_objects, read_peaks
from evaluate_alignment_consensus_p36 import (SHIFTED, ORIGINAL, VALID, RUNS, retained,
    consensus_groups, consensus_points, score_points, match_points, aggregate, admission)

EXPERIMENT = 'P38'
FEATURES = ('maximum_score_logit', 'mean_score_logit', 'support_fraction',
            'position_rms_px', 'score_standard_deviation', 'maximum_score_minus_mean')
CUTOFFS = tuple(i / 20 for i in range(1, 20))
RADIUS = 4.
FILL = 'mirror'
FLOOR = .1
PRECISION_FLOOR = .75
RIDGE = 1.
FIXED_IMAGES = (178, 776, 1037, 1482)
DECODERS = ('original', 'p36_threshold', 'p36_consensus', 'support_only', 'calibrated')
COUNTS = ('matched', 'total', 'candidates', 'dead_hit', 'dead_total')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('output rows are required')
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def calibration_split(images):
    images = [int(i) for i in images]
    if len(images) != 240 or len(set(images)) != 240:
        raise ValueError('P38 requires the same 240 distinct P36 training images')
    order = sorted(images, key=lambda i: hashlib.sha256(f'p38-20261120:{i}'.encode()).hexdigest())
    return sorted(order[:120]), sorted(order[120:])


def low_score_groups(phases):
    """Seed-radius grouping at 0.1; each alignment contributes its highest score once.

    Group ownership follows P36's earliest seed within four pixels. The seed
    remains fixed. Per-alignment representatives determine position and score
    features, preventing multiple peaks from the same alignment counting as
    independent evidence. The trained alignment is excluded.
    """
    pooled = sorted((-float(s), int(k), int(y), int(x)) for k in SHIFTED
                    for y, x, s in phases[k] if s >= FLOOR)
    seeds = np.empty((len(pooled), 2), dtype=float)
    members = []
    for neg, phase, y, x in pooled:
        near = ((seeds[:len(members)] - (y, x)) ** 2).sum(1) <= RADIUS ** 2
        if near.any():
            j = int(near.argmax())
            # Descending score and coordinate order fix ties deterministically.
            members[j].setdefault(phase, (y, x, -neg))
        else:
            seeds[len(members)] = (y, x)
            members.append({phase: (y, x, -neg)})
    result = []
    for seed, m in zip(seeds, members):
        # Normalize support over alignments valid at this fixed seed. A point
        # near a border may be valid at its own coordinate but not at the seed.
        m = {k:v for k,v in m.items() if VALID[k, int(seed[0]), int(seed[1])]}
        xy = np.asarray([v[:2] for v in m.values()], dtype=float)
        scores = np.asarray([v[2] for v in m.values()], dtype=float)
        position = xy.mean(0)
        usable = int(VALID[list(SHIFTED), int(seed[0]), int(seed[1])].sum())
        if not usable or len(m) > usable:
            raise ValueError('invalid phase support at the group seed')
        logit = lambda s: float(np.log(np.clip(s, 1e-6, 1-1e-6) / (1-np.clip(s, 1e-6, 1-1e-6))))
        features = [logit(scores.max()), logit(scores.mean()), len(m)/usable,
                    float(np.sqrt(((xy-position)**2).sum(1).mean())), float(scores.std()),
                    float(scores.max()-scores.mean())]
        result.append(dict(position=position.tolist(), support=len(m), usable=usable,
            phase_mask=sum(1 << k for k in m), seed_y=int(seed[0]), seed_x=int(seed[1]),
            features=features))
    return result


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))


def logistic_fit(features, labels):
    x = np.asarray(features, dtype=float).reshape(-1, len(FEATURES))
    y = np.asarray(labels, dtype=float)
    if not len(y) or len(x) != len(y) or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError('finite features and binary labels are required')
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    xx = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    beta = np.zeros(xx.shape[1]); penalty = np.diag([0.] + [RIDGE]*len(FEATURES))
    # Proper intercept stabilization for constant-label samples, without changing
    # real-sample fitting: fail closed rather than claim a calibrated classifier.
    if len(np.unique(y)) != 2:
        raise ValueError('both matched and unmatched groups are required for calibration')
    objective = lambda b: float((np.logaddexp(0, xx@b)-y*(xx@b)).sum() + .5*b@penalty@b)
    converged = False
    for iteration in range(100):
        p = sigmoid(xx@beta)
        gradient = xx.T@(p-y) + penalty@beta
        hessian = xx.T@((p*(1-p))[:, None]*xx) + penalty
        step = np.linalg.solve(hessian, gradient)
        previous = objective(beta)
        alpha = 1.
        while alpha >= 2**-20 and objective(beta-alpha*step) > previous + 1e-10:
            alpha *= .5
        if alpha < 2**-20:
            raise ArithmeticError('calibration line search failed')
        beta -= alpha*step
        if np.max(np.abs(alpha*step)) < 1e-7:
            converged = True; break
    if not converged or not np.isfinite(beta).all():
        raise ArithmeticError('calibration failed to converge')
    return dict(features=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                ridge=RIDGE, iterations=iteration+1, converged=True, examples=len(y), positives=int(y.sum()),
                objective=objective(beta))


def qualities(groups, model):
    if not groups:
        return np.empty(0, dtype=float)
    x = np.asarray([g['features'] for g in groups])
    return sigmoid(np.column_stack((np.ones(len(x)), (x-np.asarray(model['mean']))/model['scale'])) @ model['beta'])


def group_points(groups, name, rule, q=None):
    if rule.get('reject_all'):
        return []
    if name == 'support_only':
        return [g['position'] for g in groups if g['support']*15 >= rule['votes']*g['usable']]
    if name == 'calibrated':
        return [g['position'] for g, value in zip(groups, q) if value >= rule['cutoff']]
    raise ValueError(name)


def choose_rule(records, name):
    """F1 selection subject to precision >= .75, including all empty images.

    Ties favor the higher cutoff / more support. If no nonempty rule is
    admissible, reject all and record selection failure explicitly.
    """
    candidates = [dict(votes=v) for v in range(1, 16)] if name == 'support_only' else [dict(cutoff=t) for t in CUTOFFS]
    grid = []
    for rule in candidates:
        total = matched = points = 0
        for groups, gt, q in records:
            s = score_points(group_points(groups, name, rule, q), gt)
            total += s['total']; matched += s['matched']; points += s['candidates']
        precision = matched/points if points else None
        grid.append(dict(**rule, matched=matched, points=points, total=total, precision=precision,
                         f1=2*matched/(points+total) if points+total else 0.))
    eligible = [r for r in grid if r['points'] > 0 and r['precision'] >= PRECISION_FLOOR]
    if not eligible:
        return dict(reject_all=True, selection_failure='no_nonempty_rule_at_precision_floor'), grid
    selected = max(eligible, key=lambda r: (r['f1'], r.get('votes', r.get('cutoff'))))
    return selected, grid


def load_role(folder, role, input_log):
    manifest = json.loads((folder/'preservation_manifest.json').read_text())
    names = [f'{role}_{part}' for part in ('original.json', 'objects.csv', 'points.npz')]
    for name in names:
        p = folder/name; h = digest(p); expected = manifest[name]
        if h != expected['sha256'] or p.stat().st_size != expected['bytes']:
            raise ValueError(f'preserved input changed: {p}')
        input_log[folder.name+'/'+name] = dict(sha256=h, bytes=p.stat().st_size)
    rows = json.loads((folder/f'{role}_original.json').read_text())
    ids = [int(r['image']) for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != (240 if role == 'training' else 522):
        raise ValueError('P36 image coverage changed')
    objects = read_objects(folder/f'{role}_objects.csv')
    if not set(objects).issubset(ids):
        raise ValueError('object images differ from recorded image list')
    for i in ids:
        objects.setdefault(i, [])
    peaks = read_peaks(folder/f'{role}_points.npz', ids)
    gt = {i: [(r['cls'], (r['center_y'], r['center_x']), r['interior'], r['area']) for r in objects[i]] for i in ids}
    for row in rows:
        actual = score_points(retained(peaks[row['image']]['white'][ORIGINAL]), gt[row['image']])
        if any(actual[k] != row[k] for k in COUNTS):
            raise ValueError('original count replay differs')
    return ids, objects, peaks, gt, rows


def fit_selection(folder, out, input_log):
    ids, objects, peaks, gt, _ = load_role(folder, 'training', input_log)
    fit, select = calibration_split(ids)
    groups = {i: low_score_groups(peaks[i][FILL]) for i in ids}
    x, y = [], []
    for i in fit:
        hits = {int(p) for _, p in match_points([g['position'] for g in groups[i]], gt[i])}
        for j, g in enumerate(groups[i]):
            x.append(g['features']); y.append(int(j in hits))
    model = logistic_fit(x, y)
    qs = {i: qualities(groups[i], model) for i in ids}
    records = [(groups[i], gt[i], qs[i]) for i in select]
    rules, grids = {}, {}
    for name in ('support_only', 'calibrated'):
        rules[name], grids[name] = choose_rule(records, name)
    result = dict(experiment=EXPERIMENT, model=folder.name, calibration_images=fit, selection_images=select,
        image_sets_disjoint=True, detector_training_images=True, detector_forward_passes=0,
        development_records_opened=0, fit=model, rules=rules, selection_grid=grids,
        fixed=dict(fill=FILL, radius_px=RADIUS, record_floor=FLOOR, alignments=list(SHIFTED),
            representative='unweighted mean of highest-score point per alignment', precision_floor=PRECISION_FLOOR))
    write_json(out/'selection.json', result)
    return result


def render(image, gt, displays, path, title):
    from PIL import Image, ImageDraw, ImageFont
    side = 640; scale = side/256; margin = 28; gap = 36; top = 115; height = 750
    canvas = Image.new('RGB', (2*side+2*margin+gap, 2*height+top), 'white')
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    d.text((margin, 12), title, font=font, fill='black')
    d.text((margin, 47), 'Cyan: annotated center; green +: matched; red x: extra; yellow: missed.', font=font, fill='black')
    rgb = Image.fromarray(np.asarray(image, np.uint8)).resize((side, side), Image.Resampling.NEAREST)
    for panel, (label, points) in enumerate(displays):
        left = margin+(panel%2)*(side+gap); base = top+(panel//2)*height
        canvas.paste(rgb, (left, base+50))
        pairs = match_points(points, gt); hit = {int(g) for g, _ in pairs}; used = {int(p) for _, p in pairs}
        d.text((left, base), f'{label}: {len(pairs)}/{len(gt)} matched, {len(points)-len(pairs)} extra', font=font, fill='black')
        xy = lambda p: (left+(p[1]+.5)*scale, base+50+(p[0]+.5)*scale)
        for j, (cls, center, interior, _) in enumerate(gt):
            x, y = xy(center); color = '#00c7e5' if panel == 0 else ('#19b453' if j in hit else '#ffbf00')
            d.ellipse((x-7,y-7,x+7,y+7), outline=color, width=2)
            if cls == 4 and interior:
                d.text((x+5,y-20), str(j), font=font, fill='#ffbf00', stroke_width=1, stroke_fill='black')
        for j, p in enumerate(points):
            x,y=xy(p); color='#19b453' if j in used else '#ff3654'
            if j in used:
                d.line((x-6,y,x+6,y), fill=color, width=3);d.line((x,y-6,x,y+6),fill=color,width=3)
            else:
                d.line((x-6,y-6,x+6,y+6),fill=color,width=3);d.line((x-6,y+6,x+6,y-6),fill=color,width=3)
    canvas.save(path, dpi=(150,150))


def evaluate_role(folder, out, role, selection, input_log, images=None):
    ids, objects, peaks, gt, original_rows = load_role(folder, role, input_log)
    tissue = {r['image']: r['tissue'] for r in original_rows}
    manifest = json.loads((folder/'preservation_manifest.json').read_text())
    p = folder/'selected_rule.json'
    if digest(p) != manifest[p.name]['sha256']:
        raise ValueError('P36 selection changed')
    input_log[folder.name+'/'+p.name] = manifest[p.name]
    old = json.loads(p.read_text()); cr = old['consensus']; threshold = old['threshold']['threshold']
    expected = {}
    for name in ('selected_threshold', 'consensus'):
        p = folder/f'{role}_{name}.json'
        if digest(p) != manifest[p.name]['sha256']:
            raise ValueError('P36 comparison changed')
        expected[name] = {r['image']: r for r in json.loads(p.read_text())}
        input_log[folder.name+'/'+p.name] = manifest[p.name]
    rows = {name: [] for name in DECODERS}; point_rows = []; object_rows = []
    for i in ids:
        groups = low_score_groups(peaks[i][FILL]); q = qualities(groups, selection['fit'])
        displays = dict(original=retained(peaks[i]['white'][ORIGINAL]),
            p36_threshold=retained(peaks[i]['white'][ORIGINAL], threshold),
            p36_consensus=consensus_points(consensus_groups(peaks[i][cr['fill']], cr['radius']), cr['votes']))
        for name in ('support_only', 'calibrated'):
            displays[name] = group_points(groups, name, selection['rules'][name], q)
        hits = {}
        for name, pp in displays.items():
            counts = score_points(pp, gt[i]); hits[name] = {int(g) for g, _ in match_points(pp, gt[i])}
            rows[name].append(dict(image=i, tissue=tissue[i], **counts))
            if name in ('p36_threshold', 'p36_consensus'):
                previous = expected['selected_threshold' if name == 'p36_threshold' else 'consensus'][i]
                if any(counts[k] != previous[k] for k in COUNTS):
                    raise ValueError('P36 selected count replay differs')
        accepted = {tuple(p) for p in displays['calibrated']}
        for j, (g, value) in enumerate(zip(groups, q)):
            point_rows.append(dict(image=i, group_index=j, y=g['position'][0], x=g['position'][1],
                score=float(value), support=g['support'], usable=g['usable'], phase_mask=g['phase_mask'],
                accepted=tuple(g['position']) in accepted, **dict(zip(FEATURES, g['features']))))
        for j, obj in enumerate(objects[i]):
            object_rows.append(dict(image=i, gt_index=j, cls=obj['cls'], center_y=obj['center_y'],
                center_x=obj['center_x'], area=obj['area'], neighbors12=obj['neighbors12'], border=obj['border'],
                interior=obj['interior'], **{name: j in hits[name] for name in DECODERS}))
        if role == 'development' and i in FIXED_IMAGES and folder.name in ('P22_M_s19', 'P15_J_s19') and images is not None:
            render(images[i], gt[i], [('Annotated centers', []), ('P36 consensus', displays['p36_consensus']),
                ('Low-score support rule', displays['support_only']), ('Calibrated consensus', displays['calibrated'])],
                out/'figures'/f'{folder.name}_{i:04d}.png', f'{folder.name}: development image {i}')
    for name, value in rows.items():
        write_json(out/f'{role}_{name}.json', value)
    write_csv(out/f'{role}_groups.csv', point_rows)
    write_csv(out/f'{role}_objects.csv', object_rows)
    transitions = {name: dict(gained=sum(r[name] and not r['original'] for r in object_rows),
        lost=sum(r['original'] and not r[name] for r in object_rows)) for name in DECODERS[1:]}
    return dict(images=len(ids), decoders={name: dict(counts={k:sum(r[k] for r in rr) for k in COUNTS},
        rates=aggregate(rr)) for name, rr in rows.items()}, transitions=transitions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); input_log = {}; selections = {}; folders = []
    keys = [f'{e}_{a}_s{s}' for e,a,s in RUNS]
    for key in keys:
        folder = args.root/key
        complete = json.loads((folder/'complete.json').read_text())
        if complete['model'] != key or complete['exit_code'] != 0 or complete['phases'] != 16:
            raise ValueError('P36 scientific completion missing')
        out = args.out/key;out.mkdir();(out/'figures').mkdir()
        selections[key] = fit_selection(folder, out, input_log);folders.append(folder)
        if key != keys[0] and any(selections[key][name] != selections[keys[0]][name]
                for name in ('calibration_images', 'selection_images')):
            raise ValueError('training calibration membership differs across detectors')
        print(json.dumps(dict(event='selected',model=key,rules=selections[key]['rules'],seconds=time.monotonic()-started)),flush=True)
    selection_hashes = {key: digest(args.out/key/'selection.json') for key in keys}
    write_json(args.out/'selection_complete.json', dict(models=keys, development_records_opened=0,
        selection_sha256=selection_hashes, script_sha256=digest(__file__)))
    # Only now open development images and predictions.
    image_hash = digest(args.images)
    for folder in folders:
        provenance = json.loads((folder/'provenance.json').read_text())
        if image_hash != provenance['input_sha256']['fold1/images.npy']:
            raise ValueError('image source differs from preserved P36 input')
    images = np.load(args.images, mmap_mode='r')
    summaries = {}
    for folder in folders:
        key=folder.name;out=args.out/key
        roles={role:evaluate_role(folder,out,role,selections[key],input_log,images)
            for role in ('training','development')}
        summaries[key]=dict(model=key,roles=roles,selection=selections[key],selection_sha256=selection_hashes[key])
        write_json(out/'summary.json',summaries[key])
        print(json.dumps(dict(event='evaluated',model=key,development=roles['development'],seconds=time.monotonic()-started)),flush=True)
    decisions={}
    for experiment,arm in (('P15','J'),('P22','M')):
        for name in ('support_only','calibrated'):
            paired={str(seed):dict(F1=json.loads((args.out/f'{experiment}_C_s{seed}'/f'development_{name}.json').read_text()),
                T1=json.loads((args.out/f'{experiment}_{arm}_s{seed}'/f'development_{name}.json').read_text())) for seed in (19,1)}
            decisions[f'{experiment}:{arm}:vs_C:{name}']=admission(paired)
    for experiment,arm in sorted({(e,a) for e,a,_ in RUNS}):
        for comparator in ('original','p36_consensus','support_only'):
            paired={str(seed):dict(F1=json.loads((args.out/f'{experiment}_{arm}_s{seed}'/f'development_{comparator}.json').read_text()),
                T1=json.loads((args.out/f'{experiment}_{arm}_s{seed}'/'development_calibrated.json').read_text())) for seed in (19,1)}
            decisions[f'{experiment}:{arm}:calibrated_vs_{comparator}']=admission(paired)
    result=dict(experiment=EXPERIMENT,detectors=summaries,decisions=decisions,inputs=input_log,
        image_sha256=image_hash,elapsed_seconds=time.monotonic()-started,
        primary='P15:J:vs_C:calibrated',secondary='P22:M:vs_C:calibrated',detector_forward_passes=0)
    write_json(args.out/'P38_RESULTS.json',result)
    payload={str(p.relative_to(args.out)):dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(args.out.rglob('*')) if p.is_file()}
    write_json(args.out/'preservation_manifest.json',payload)
    write_json(args.out/'complete.json',dict(experiment=EXPERIMENT,models=8,exit_code=0,
        figures=sum(1 for p in args.out.rglob('*.png')),elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
