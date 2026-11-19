"""P36: alignment-resolved point records, threshold selection and consensus decoding of frozen detectors.

Every image is passed through a frozen detector at the sixteen alignments of
P35, once with white fill and once with mirrored fill of the vacated strip.
Every local maximum with a score of at least RECORD_THRESHOLD is stored with
its position in original coordinates. Nothing is trained.

Decoders, all scored with the unchanged three-pixel one-to-one matching rule:

* original           - alignment (0, 0), score >= 0.5: the detector as trained and stored;
* selected_threshold - alignment (0, 0) at the score threshold selected below;
* consensus          - retained points (score >= 0.5) of the fifteen shifted
                       alignments are grouped around the highest-scoring point
                       within `radius` pixels; a group is kept when the share of
                       the alignments valid at its first point that contribute
                       to it reaches `votes`/15; its position is the mean of
                       its points.

The score threshold and the consensus rule (fill, radius, votes) are selected
on a fixed sample of training images at the fifteen shifted alignments, by the
F1 score over all annotated nuclei, and written to disk before the detector
reads any development image; they are then applied unchanged. Each invocation
handles one detector. Machine paths belong in the ignored JSON manifest, not in
this file.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from proposal_p3_utils import decode_peaks, score_points, split_indices, aggregate, admission
from train_proposal_p21 import match_points
from train_center_offset_p34 import (sha, write_json, write_csv, model_key, shift_array, state_fingerprint,
                                     frozen_selection, COUNT_KEYS)
from evaluate_grid_phase_p35 import PHASES, ORIGINAL, probability

EXPERIMENT = 'P36'
FILLS = ('white', 'mirror')
RADII = (2., 3., 4.)
VOTES = tuple(range(1, 16))
THRESHOLDS = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
RECORD_THRESHOLD = .1
SHIFTED = tuple(k for k in range(len(PHASES)) if k != ORIGINAL)
TRAINING_SAMPLE = 240
NEAR_RADIUS = 6.               # window for the displacement statistics, twice the matching radius
ROLES = ('training', 'development')
NEW_DECODERS = ('selected_threshold', 'consensus')
MODULES = {'P22': 'train_proposal_p22', 'P15': 'train_proposal_p15'}
CONTROL_ARM = 'C'
RUNS = [('P22', 'C', 19), ('P22', 'C', 1), ('P22', 'M', 19), ('P22', 'M', 1),
        ('P15', 'C', 19), ('P15', 'C', 1), ('P15', 'J', 19), ('P15', 'J', 1)]
BANDS = (('0-16', 0, 16), ('16-24', 16, 24), ('24-32', 24, 32), ('32-128', 32, 129))
VALID = np.stack([shift_array(np.ones((256, 256), dtype=bool), -dy, -dx, False) for dy, dx in PHASES])


def run_config(device='cpu'):
    return dict(experiment=EXPERIMENT, phases=[list(p) for p in PHASES], fills=list(FILLS),
                record_threshold=RECORD_THRESHOLD, thresholds=list(THRESHOLDS),
                threshold_selection=dict(alignments='fifteen_shifted', fills='both_pooled',
                                         criterion='training_sample_F1_all_nuclei', ties='closest_to_0.5'),
                consensus=dict(alignments='fifteen_shifted', score=.5, radii_px=list(RADII), votes=list(VOTES),
                               position='mean_of_points', selection='training_sample_F1_all_nuclei',
                               ties='white_fill,smaller_radius,more_votes'),
                training_sample=TRAINING_SAMPLE, matching_radius_px=3, near_radius_px=NEAR_RADIUS,
                roles=list(ROLES), detector='frozen', training='none',
                precision='cpu_float32' if device == 'cpu' else 'cuda_bfloat16_autocast')


def sample_training(fit):
    """Fixed sample of training images, chosen by a hash of the image number only."""
    ordered = sorted(fit, key=lambda i: hashlib.sha256(f'p36-20261119:{int(i)}'.encode()).hexdigest())
    return sorted(int(i) for i in ordered[:TRAINING_SAMPLE])


def mirror_shift(array, dy, dx):
    """Integer translation; the vacated strip shows the mirrored image content."""
    array = np.asarray(array)
    pad = max(abs(dy), abs(dx))
    if pad == 0:
        return array.copy()
    padded = np.pad(array, ((pad, pad), (pad, pad)) + ((0, 0),) * (array.ndim - 2), mode='reflect')
    return padded[pad - dy:pad - dy + array.shape[0], pad - dx:pad - dx + array.shape[1]].copy()


def aligned_maps(model, image, device, fill):
    """Logits of the sixteen alignments, moved back to original coordinates (zero where invalid)."""
    image = np.asarray(image)
    moved = [shift_array(image, dy, dx, 255) if fill == 'white' else mirror_shift(image, dy, dx) for dy, dx in PHASES]
    x = torch.from_numpy(np.stack(moved)).permute(0, 3, 1, 2).float().to(device) / 255
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        out = model(x).float().cpu().numpy()
    if out.shape != VALID.shape or not np.isfinite(out).all():
        raise FloatingPointError('unexpected or nonfinite detector output')
    return np.stack([shift_array(out[k], -dy, -dx, 0) for k, (dy, dx) in enumerate(PHASES)])


def record_peaks(logits):
    """Per alignment: local maxima with score >= RECORD_THRESHOLD as (y, x, score), in decoder order."""
    prob = probability(logits) * VALID
    peaks = [[(y, x, float(prob[k][y, x])) for y, x in decode_peaks(prob[k], RECORD_THRESHOLD)]
             for k in range(len(PHASES))]
    assert retained(peaks[ORIGINAL]) == decode_peaks(prob[ORIGINAL]), 'recorded peaks differ from the original decoder'
    return peaks


def retained(points, threshold=.5):
    return [(y, x) for y, x, s in points if s >= threshold]


def collect(model, source, ids, device):
    """Forward passes only: peaks[image][fill][alignment] = [(y, x, score), ...]."""
    peaks, started = {}, time.monotonic()
    for image in ids:
        peaks[image] = {fill: record_peaks(aligned_maps(model, source.images[image], device, fill)) for fill in FILLS}
    return peaks, time.monotonic() - started


def consensus_groups(points, radius):
    """Group the retained points of the fifteen shifted alignments around the highest-scoring point.

    Points are visited in descending score order; a point joins the earliest
    group whose first point lies within `radius` pixels, otherwise it starts a
    group. `usable` counts the shifted alignments valid at the first point.
    """
    pooled = sorted((-s, k, y, x) for k in SHIFTED for y, x, s in points[k] if s >= .5)
    seeds = np.empty((len(pooled), 2))
    members, phases = [], []
    for _, k, y, x in pooled:
        near = (seeds[:len(members), 0] - y) ** 2 + (seeds[:len(members), 1] - x) ** 2 <= radius * radius
        if near.any():
            j = int(near.argmax())
            members[j].append((y, x))
            phases[j].add(k)
        else:
            seeds[len(members)] = (y, x)
            members.append([(y, x)])
            phases.append({k})
    return [dict(position=tuple(float(v) for v in np.mean(m, axis=0)), votes=len(p), points=len(m),
                 usable=int(VALID[list(SHIFTED), m[0][0], m[0][1]].sum())) for m, p in zip(members, phases)]


def consensus_points(groups, votes):
    """Groups supported by at least votes/15 of the alignments valid at their first point."""
    return [g['position'] for g in groups if g['votes'] * len(SHIFTED) >= votes * g['usable']]


def triplet(points, gt):
    s = score_points(points, gt)
    return s['matched'], s['candidates'], s['dead_hit']


def grid(peaks, source, ids):
    """Per-image counts (matched, points, interior Dead matched) of every threshold and consensus rule.

    `thresholds` belongs to the trained alignment; `shifted_thresholds` is the
    sum over the fifteen shifted alignments, per fill.
    """
    n = len(ids)
    table = dict(images=np.asarray(ids, np.int32), total=np.zeros(n, np.int32), dead_total=np.zeros(n, np.int32),
                 thresholds=np.zeros((n, len(THRESHOLDS), 3), np.int32),
                 shifted_thresholds=np.zeros((n, len(FILLS), len(THRESHOLDS), 3), np.int32),
                 consensus=np.zeros((n, len(FILLS), len(RADII), len(VOTES), 3), np.int32))
    for row, image in enumerate(ids):
        gt = source.gt[image]
        empty = score_points([], gt)
        table['total'][row], table['dead_total'][row] = empty['total'], empty['dead_total']
        for t, threshold in enumerate(THRESHOLDS):
            table['thresholds'][row, t] = triplet(retained(peaks[image]['white'][ORIGINAL], threshold), gt)
        for f, fill in enumerate(FILLS):
            for t, threshold in enumerate(THRESHOLDS):
                table['shifted_thresholds'][row, f, t] = np.sum(
                    [triplet(retained(peaks[image][fill][k], threshold), gt) for k in SHIFTED], axis=0)
            for r, radius in enumerate(RADII):
                groups = consensus_groups(peaks[image][fill], radius)
                for v, votes in enumerate(VOTES):
                    table['consensus'][row, f, r, v] = triplet(consensus_points(groups, votes), gt)
    return table


def f1_score(matched, points, nuclei):
    return 2 * matched / (points + nuclei) if points + nuclei else 0.


def select_threshold(table):
    """Score threshold with the highest F1 over all nuclei at the shifted alignments; ties: closest to 0.5."""
    counts = table['shifted_thresholds'].sum((0, 1), dtype=np.int64)
    nuclei = int(table['total'].sum()) * len(SHIFTED) * len(FILLS)
    scores = [f1_score(int(m), int(c), nuclei) for m, c, _ in counts]
    best = max(range(len(THRESHOLDS)), key=lambda t: (scores[t], -abs(THRESHOLDS[t] - .5)))
    return dict(threshold=THRESHOLDS[best], f1=scores[best], matched=int(counts[best, 0]), points=int(counts[best, 1]),
                nuclei=nuclei)


def select_rule(table):
    """Consensus rule with the highest F1 over all nuclei; ties: white fill, smaller radius, more votes."""
    counts = table['consensus'].sum(0, dtype=np.int64)
    total, best = int(table['total'].sum()), None
    for f, fill in enumerate(FILLS):
        for r, radius in enumerate(RADII):
            for v in reversed(range(len(VOTES))):
                matched, candidates = int(counts[f, r, v, 0]), int(counts[f, r, v, 1])
                score = f1_score(matched, candidates, total)
                if best is None or score > best['f1']:
                    best = dict(fill=fill, radius=radius, votes=VOTES[v], f1=score, matched=matched, points=candidates,
                                nuclei=total)
    return best


def rule_index(rule):
    return FILLS.index(rule['fill']), RADII.index(rule['radius']), VOTES.index(rule['votes'])


def rule_rows(table, source, name, index):
    """Per-image rows in the form used by the paired two-seed rule."""
    rows = []
    for row, image in enumerate(table['images'].tolist()):
        matched, candidates, dead_hit = (int(v) for v in table[name][(row,) + tuple(index)])
        rows.append(dict(image=image, tissue=str(source.tissue[image]), matched=matched, total=int(table['total'][row]),
                         candidates=candidates, dead_hit=dead_hit, dead_total=int(table['dead_total'][row])))
    return rows


def totals(rows):
    return {k: int(sum(r[k] for r in rows)) for k in COUNT_KEYS}


def nearest_offsets(points, centers, radius=NEAR_RADIUS):
    """For every center: displacement (dy, dx) of the nearest point within `radius`, else NaN."""
    out = np.full((len(centers), 2), np.nan)
    if len(points) and len(centers):
        delta = np.asarray(points, float)[None] - np.asarray(centers, float)[:, None]
        distance = (delta ** 2).sum(-1)
        j = distance.argmin(1)
        rows = np.arange(len(centers))
        ok = distance[rows, j] <= radius * radius
        out[ok] = delta[rows, j][ok]
    return out


def describe(peaks, source, ids, rule, key, role, figures=(), out=None):
    """Per-alignment counts, per-object records and figures for one role under the selected rule."""
    single = np.zeros((len(ids), len(FILLS), len(PHASES), 3), np.int32)
    objects = []
    for row, image in enumerate(ids):
        gt = source.gt[image]
        centers = [g[1] for g in gt]
        hits, offsets = {}, {}
        for f, fill in enumerate(FILLS):
            hits[fill], offsets[fill] = [], []
            for k in range(len(PHASES)):
                points = retained(peaks[image][fill][k])
                single[row, f, k] = triplet(points, gt)
                hits[fill].append({int(g) for g, _ in match_points(points, gt)})
                offsets[fill].append(nearest_offsets(points, centers))
        chosen = consensus_points(consensus_groups(peaks[image][rule['fill']], rule['radius']), rule['votes'])
        agreed = {int(g) for g, _ in match_points(chosen, gt)}
        position = np.asarray(centers, float).reshape(-1, 2)
        neighbors = (((position[:, None] - position[None]) ** 2).sum(-1) <= 144).sum(1) - 1
        for index, (cls, (y, x), interior, area) in enumerate(gt):
            first = offsets['white'][ORIGINAL][index]
            record = dict(model=key, role=role, image=int(image), gt_index=index, cls=int(cls), interior=bool(interior),
                          area=int(area), center_y=int(y), center_x=int(x), border=int(min(y, x, 255 - y, 255 - x)),
                          neighbors12=int(neighbors[index]),
                          valid_phases=int(VALID[:, y, x].sum()), matched_original=index in hits['white'][ORIGINAL],
                          matched_consensus=index in agreed,
                          original_dy=None if np.isnan(first[0]) else float(first[0]),
                          original_dx=None if np.isnan(first[1]) else float(first[1]))
            for fill in FILLS:
                moved = np.stack([offsets[fill][k][index] for k in SHIFTED])
                seen = moved[~np.isnan(moved[:, 0])]
                record.update({f'phases_{fill}': int(sum(1 << k for k in range(len(PHASES)) if index in hits[fill][k])),
                               f'shifted_matched_{fill}': int(sum(index in hits[fill][k] for k in SHIFTED)),
                               f'shifted_near_{fill}': int(len(seen)),
                               f'mean_dy_{fill}': float(seen[:, 0].mean()) if len(seen) else None,
                               f'mean_dx_{fill}': float(seen[:, 1].mean()) if len(seen) else None,
                               f'spread2_{fill}': float(seen.var(0, ddof=1).sum()) if len(seen) > 1 else None})
            objects.append(record)
        if image in figures:
            cloud = [p for k in SHIFTED for p in retained(peaks[image][rule['fill']][k])]
            render(np.asarray(source.images[image]), gt, retained(peaks[image]['white'][ORIGINAL]), cloud, chosen,
                   out / 'figures' / f'{key}_{role}_{image:04d}.png',
                   f'{key}: {role} image {image}, {source.tissue[image]}', rule)
    return single, objects


def object_summary(objects):
    """Detection by distance to the border, alignment counts and the displacement decomposition."""
    full = [o for o in objects if o['valid_phases'] == len(PHASES)]
    result = dict(objects=len(objects), valid_at_every_alignment=len(full), border_bands={})
    for name, low, high in BANDS:
        rows = [o for o in full if low <= o['border'] < high]
        result['border_bands'][name] = dict(
            objects=len(rows), original=float(np.mean([o['matched_original'] for o in rows])) if rows else None,
            **{fill: float(np.mean([o[f'shifted_matched_{fill}'] for o in rows]) / len(SHIFTED)) if rows else None
               for fill in FILLS})
    central = [o for o in full if o['border'] >= 24]
    result['central_objects'] = len(central)
    for fill in FILLS:
        # Displacement statistics use isolated nuclei, so that the nearest point belongs to the nucleus itself.
        seen = [o for o in central if o['neighbors12'] == 0 and o[f'spread2_{fill}'] is not None]
        spread = np.asarray([o[f'spread2_{fill}'] for o in seen], float)
        count = np.asarray([o[f'shifted_near_{fill}'] for o in seen], float)
        mean2 = np.asarray([o[f'mean_dy_{fill}'] ** 2 + o[f'mean_dx_{fill}'] ** 2 for o in seen], float)
        matched = np.asarray([o[f'shifted_matched_{fill}'] for o in central], dtype=np.int64)
        result[fill] = dict(
            histogram=np.bincount(matched, minlength=len(SHIFTED) + 1).tolist(),
            isolated_objects_with_two_or_more_points=len(seen),
            alignment_jitter_px2=float(spread.mean()) if len(seen) else None,
            stable_displacement_px2=float((mean2 - spread / count).mean()) if len(seen) else None)
    return result


def render(image, gt, original, cloud, chosen, path, title, rule):
    """Large panels: annotation, original decoder, points of the shifted alignments, consensus decoder."""
    from PIL import Image, ImageDraw, ImageFont
    panels = [('Image and annotated centers', None), ('Original decoder (trained alignment)', original),
              (f"Points of the 15 shifted alignments ({rule['fill']} fill)", None), ('Consensus decoder', chosen)]
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

    def ring(col, pos, color, radius=8, width=3):
        cx, cy = xy(col, pos)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=width)

    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((side, side), Image.Resampling.NEAREST)
    counts = []
    for col, (text, shown) in enumerate(panels):
        canvas.paste(rgb, (left + col * (side + gap), top))
        label((left + col * (side + gap), 112), text)
        if col == 0:
            for cls, center, interior, _ in gt:
                ring(0, center, '#ffbf00' if cls == 4 and interior else '#00bbdd')
        elif shown is None:
            for _, center, _, _ in gt:
                ring(col, center, '#ffffff', radius=3 * scale, width=1)
            for pos in cloud:
                cx, cy = xy(col, pos)
                draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='#3366ff')
        else:
            pairs = match_points(shown, gt)
            hit, used = {int(g) for g, _ in pairs}, {int(p) for _, p in pairs}
            for index, (_, center, _, _) in enumerate(gt):
                ring(col, center, '#44cc44' if index in hit else '#ffbf00')
            for j, pos in enumerate(shown):
                cx, cy = xy(col, pos)
                if j in used:
                    draw.line((cx - 7, cy, cx + 7, cy), fill='#44cc44', width=3)
                    draw.line((cx, cy - 7, cx, cy + 7), fill='#44cc44', width=3)
                else:
                    draw.line((cx - 7, cy - 7, cx + 7, cy + 7), fill='#ff4444', width=3)
                    draw.line((cx - 7, cy + 7, cx + 7, cy - 7), fill='#ff4444', width=3)
            counts.append(f'{text.lower()}: {len(shown)} points, {len(pairs)} matched')
    label((left, 16), title, 3)
    label((left, 64), 'First panel: yellow ring = interior Dead nucleus, cyan = other nuclei. Third panel: white circle = '
                      'three-pixel matching radius, blue dot = one point of one alignment.')
    label((left, top + side + 14), 'Second and fourth panel: ring = annotated center (green matched, yellow missed); '
                                   'green plus = matched point, red cross = extra point.')
    label((left, top + side + 56), f"Annotated nuclei = {len(gt)}; consensus rule: radius {rule['radius']:g} px, at least "
                                   f"{rule['votes']} of 15 alignments; " + '; '.join(counts))
    canvas.save(path, format='PNG', dpi=(150, 150))


def save_points(path, peaks, ids):
    columns = dict(image=[], fill=[], phase=[], y=[], x=[], score=[])
    for image in ids:
        for f, fill in enumerate(FILLS):
            for k, points in enumerate(peaks[image][fill]):
                for y, x, s in points:
                    for name, value in zip(columns, (image, f, k, y, x, s)):
                        columns[name].append(value)
    kinds = dict(image=np.int32, fill=np.uint8, phase=np.uint8, y=np.uint8, x=np.uint8, score=np.float32)
    np.savez_compressed(path, **{name: np.asarray(values, kinds[name]) for name, values in columns.items()})


def run_detector(key, model, fold, training, audit, development_figures, out, device, historical=None):
    """Record every alignment, select threshold and consensus rule on training images, then evaluate both roles."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    (out / 'figures').mkdir()
    started = time.monotonic()
    model.requires_grad_(False).eval()
    frozen_before = state_fingerprint(model)
    ids = dict(training=list(training), development=list(audit))
    figures = dict(training=set(ids['training'][:4]), development=set(development_figures))
    peaks, tables, seconds = {}, {}, {}
    peaks['training'], seconds['training'] = collect(model, fold, ids['training'], device)
    tables['training'] = grid(peaks['training'], fold, ids['training'])
    rule, cut = select_rule(tables['training']), select_threshold(tables['training'])
    write_json(out / 'selected_rule.json', dict(consensus=rule, threshold=cut,
                                                selected_on='training sample, fifteen shifted alignments',
                                                training_images=len(ids['training']), development_images_read=0))
    print(json.dumps(dict(event='rule', model=key, seconds=time.monotonic() - started, consensus=rule, threshold=cut)),
          flush=True)
    peaks['development'], seconds['development'] = collect(model, fold, ids['development'], device)
    tables['development'] = grid(peaks['development'], fold, ids['development'])
    assert state_fingerprint(model) == frozen_before, 'frozen detector changed'
    half, results, roles = THRESHOLDS.index(.5), {}, {}
    for role in ROLES:
        table = tables[role]
        rows = dict(original=rule_rows(table, fold, 'thresholds', (half,)),
                    selected_threshold=rule_rows(table, fold, 'thresholds', (THRESHOLDS.index(cut['threshold']),)),
                    consensus=rule_rows(table, fold, 'consensus', rule_index(rule)))
        table['single'], objects = describe(peaks[role], fold, ids[role], rule, key, role, figures[role], out)
        assert np.array_equal(table['single'][:, 0, ORIGINAL], table['thresholds'][:, half])
        assert np.array_equal(table['single'][:, :, list(SHIFTED)].sum(2), table['shifted_thresholds'][:, :, half])
        assert sum(o['matched_consensus'] for o in objects) == totals(rows['consensus'])['matched']
        np.savez_compressed(out / f'{role}_counts.npz', **table)
        save_points(out / f'{role}_points.npz', peaks[role], ids[role])
        for name, value in rows.items():
            write_json(out / f'{role}_{name}.json', value)
        write_csv(out / f'{role}_objects.csv', objects)
        comparison = None
        if historical is not None and role in historical:
            stored = [historical[role][r['image']] for r in rows['original']]
            comparison = dict(images=len(stored), stored_counts=totals(stored), recomputed_counts=totals(rows['original']),
                              images_with_different_counts=int(sum(any(s[k] != r[k] for k in COUNT_KEYS)
                                                                   for s, r in zip(stored, rows['original']))))
        results[role] = rows
        roles[role] = dict(
            images=len(ids[role]),
            decoders={name: dict(counts=totals(value), rates=aggregate(value)) for name, value in rows.items()},
            thresholds=[dict(threshold=t, counts=table['thresholds'][:, i].sum(0).tolist())
                        for i, t in enumerate(THRESHOLDS)],
            shifted_thresholds=table['shifted_thresholds'].sum(0).tolist(),
            consensus_grid=table['consensus'].sum(0).tolist(), single_phases=table['single'].sum(0).tolist(),
            nuclei=int(table['total'].sum()), interior_dead=int(table['dead_total'].sum()),
            objects=object_summary(objects), stored_original_comparison=comparison,
            forward_seconds_per_image=seconds[role] / max(1, len(ids[role])))
        print(json.dumps(dict(event='role', model=key, role=role, seconds=time.monotonic() - started,
                              decoders={name: v['counts'] for name, v in roles[role]['decoders'].items()})), flush=True)
    summary = dict(model=key, config=run_config(device.type), rule=rule, threshold=cut, roles=roles,
                   frozen_detector_fingerprint=frozen_before, frozen_detector_unchanged=True,
                   elapsed_seconds=time.monotonic() - started)
    write_json(out / 'summary.json', summary)
    return results, summary


def paired_decisions(results):
    """Original two-seed rule.

    Per recipe: the original decoder against each new decoder of the same
    detector. Per treatment recipe: every decoder against the original decoder
    of the control detector of the same experiment and seed.
    """
    decisions = {}
    for experiment, arm in sorted({k[:2] for k in results}):
        seeds = {seed: rows for (e, a, seed), rows in results.items() if (e, a) == (experiment, arm)}
        for name in NEW_DECODERS:
            decisions[f'{experiment}:{arm}:{name}'] = admission(
                {str(seed): dict(F1=rows['original'], T1=rows[name]) for seed, rows in seeds.items()})
        if arm != CONTROL_ARM:
            for name in ('original',) + NEW_DECODERS:
                decisions[f'{experiment}:{arm}:{name}_vs_{CONTROL_ARM}:original'] = admission(
                    {str(seed): dict(F1=results[(experiment, CONTROL_ARM, seed)]['original'], T1=rows[name])
                     for seed, rows in seeds.items() if (experiment, CONTROL_ARM, seed) in results})
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
        results[(run['experiment'], run['arm'], run['seed'])] = {
            name: json.loads((folder / f'development_{name}.json').read_text()) for name in ('original',) + NEW_DECODERS}
        summaries.append(json.loads((folder / 'summary.json').read_text()))
    combined = dict(experiment=EXPERIMENT, detectors=summaries, decisions=paired_decisions(results))
    write_json(root / 'summary.json', combined)
    return combined


def validate_detector(run, module, weights_digest, cohort):
    """Identity of a completed 130-epoch run: configuration, cohort, weights and preserved files."""
    root = Path(run['directory'])
    manifest = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    preservation = json.loads((root / 'preservation_manifest.json').read_text())
    expected = module.run_config(run['arm'], run['seed'], 4) if run['experiment'] == 'P22' \
        else module.run_config(run['arm'], run['seed'])
    assert complete['epochs'] == 130 and complete['exit_code'] == 0
    assert manifest['weights_sha256'] == weights_digest and manifest['cohort_sha256'] == cohort
    assert manifest['config'] == json.loads(json.dumps(expected)), 'stored configuration differs from the recipe'
    epochs = [json.loads(row)['epoch'] for row in (root / 'loss.jsonl').read_text().splitlines()]
    assert epochs == list(range(1, 131))
    for rel in ('last.pth', 'manifest.json', 'evaluation.json', 'fit.json', 'complete.json', 'loss.jsonl'):
        assert (root / rel).stat().st_size == preservation[rel]['bytes'] and sha(root / rel) == preservation[rel]['sha256'], rel
    return manifest, preservation


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
    started = time.monotonic()
    device = torch.device(spec['device'])
    assert spec['experiment'] == EXPERIMENT and spec['fold'] == 1 and spec['config'] == run_config(device.type)
    assert [(r['experiment'], r['arm'], r['seed']) for r in spec['runs']] == RUNS
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
    selection = dict(development_images=frozen_selection(fold, fit, audit)['development_images'],
                     training_sample=sample_training(fit))
    assert selection == spec['selection'], 'annotation-only image selection changed after freeze'
    module = importlib.import_module(MODULES[run['experiment']])
    root = Path(run['directory'])
    manifest, preservation = validate_detector(run, module, weights_digest, cohort)
    assert manifest['input_sha256'] == dict(inputs, **{'fold1/tissue.npy': TISSUE_SHA})
    checkpoint = torch.load(root / 'last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['completed_epoch'] == 130
    assert json.loads(json.dumps(checkpoint['config'])) == manifest['config']
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
    _, summary = run_detector(key, model, fold, selection['training_sample'], audit, selection['development_images'],
                              out, device, historical)
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
    print(json.dumps(dict(event='run_complete', model=key, rule=summary['rule'], threshold=summary['threshold'],
                          development={n: v['counts'] for n, v in summary['roles']['development']['decoders'].items()},
                          elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
