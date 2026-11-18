"""P34: center-offset readout trained on frozen, completed P22 detectors.

The detector, its scores, its retained peaks and their order stay fixed. A small
head reads the detector's last 64-channel feature map and predicts, for every
pixel inside an annotated nucleus, the vector to the annotated center. Three
decoders are compared on the same peaks:

* original  - the frozen decoder output (must replay the stored counts);
* refined   - each retained peak moved by the offset read at that peak;
* merged    - refined peaks, with later peaks removed when they land within
              MERGE_RADIUS pixels of an earlier, higher-scoring refined peak.

Machine paths belong in the ignored JSON manifest, not in this file.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from proposal_p3_utils import (decode_peaks, score_points, split_indices, aggregate,
                               admission, augment_pair)
from train_proposal_p21 import match_points, select_examples

EXPERIMENT = 'P34'
EPOCHS = 20
BATCH = 16
EVAL_BATCH = 4
HEAD_LR = 3e-4
WEIGHT_DECAY = .01
MAX_SHIFT = 16.      # one UNI token cell, in native pixels
MERGE_RADIUS = 2.    # native pixels, measured between refined positions
DECODERS = ('original', 'refined', 'merged')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_config():
    return dict(experiment=EXPERIMENT, epochs=EPOCHS, batch=BATCH, eval_batch=EVAL_BATCH,
                head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY, schedule='cosine20_floor0.1',
                loss='per_object_mean_L1', supervision='pixels_inside_annotated_instances',
                augmentation='original_D4_photometric', max_shift_px=MAX_SHIFT,
                merge_radius_px=MERGE_RADIUS, head='conv3x3-64,conv3x3-64,conv1x1-2,zero_output',
                feature='input_of_detector_output_convolution', detector='frozen')


# ---------------------------------------------------------------- targets

def center_fields(inst, gt):
    """Per-pixel linear index of the owning nucleus center (-1 outside) and 1/area."""
    inst = np.asarray(inst).astype(np.int64)
    ids = np.unique(inst)
    ids = ids[ids != 0]
    if len(ids) != len(gt):
        raise ValueError('instance map and annotated objects disagree')
    width = inst.shape[1]
    centers = np.full(int(inst.max()) + 1 if inst.size else 1, -1, np.int64)
    weights = np.zeros(len(centers), np.float32)
    for oid, (_, (y, x), _, area) in zip(ids, gt):
        if inst[y, x] != oid or area != int((inst == oid).sum()):
            raise ValueError('annotated center or area does not belong to its instance')
        centers[oid] = y * width + x
        weights[oid] = 1. / area
    return centers[inst], weights[inst]


def training_sample(image, inst, gt, rng=None, fields=None):
    """Image, (2,H,W) offset target in pixels, per-pixel weight, object count.

    The original augmentation function moves the image, the center-index map and
    a pixel-identity map together; the target is rebuilt from the moved maps, so
    it never depends on a hand-written rotation formula. `fields` may carry the
    validated result of center_fields for this image.
    """
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
    centers, weights = center_fields(inst, gt) if fields is None else fields
    height, width = centers.shape
    if height != width:
        raise ValueError('square native images required')
    identity = torch.arange(height * width, dtype=torch.float64).reshape(height, width)
    packed = torch.stack([torch.from_numpy(centers).double(), identity,
                          torch.from_numpy(weights).double()])
    if rng is not None:
        x, packed = augment_pair(x, packed, rng)
    centers_moved = packed[0].round().long()
    identity_moved = packed[1].round().long()
    position = torch.empty(height * width, dtype=torch.long)
    position[identity_moved.reshape(-1)] = torch.arange(height * width)
    inside = centers_moved >= 0
    center_position = position[centers_moved.clamp(min=0)]
    grid = torch.arange(height * width).reshape(height, width)
    target = torch.stack([(center_position // width - grid // width).float(),
                          (center_position % width - grid % width).float()]) * inside
    return x, target, packed[2].float() * inside, len(gt)


def offset_loss(pred, target, weight, objects):
    """Mean over objects of the mean absolute offset error (dy plus dx) inside the object."""
    return ((pred.float() - target).abs().sum(1) * weight).sum() / max(1, objects)


# ---------------------------------------------------------------- model parts

class OffsetHead(nn.Module):
    def __init__(self, channels=64, width=64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features):
        coarse = self.out(self.body(features.float()))
        return F.interpolate(coarse, size=(256, 256), mode='bilinear', align_corners=False)


class FeatureTap:
    """Read-only hook on the input of the detector's output convolution."""
    def __init__(self, model):
        self.features = None
        self.handle = model.head.out.register_forward_pre_hook(self._read)

    def _read(self, module, args):
        self.features = args[0].detach()

    def close(self):
        self.handle.remove()


def state_fingerprint(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- decoders

def refine_points(points, offsets, max_shift=MAX_SHIFT):
    """Move each retained peak by the offset read at that peak (bounded, in-image)."""
    height, width = offsets.shape[1:]
    refined = []
    for y, x in points:
        dy, dx = float(offsets[0, y, x]), float(offsets[1, y, x])
        if not (math.isfinite(dy) and math.isfinite(dx)):
            raise FloatingPointError('nonfinite offset')
        norm = math.hypot(dy, dx)
        if norm > max_shift:
            dy, dx = dy * max_shift / norm, dx * max_shift / norm
        refined.append((min(height - 1., max(0., y + dy)), min(width - 1., max(0., x + dx))))
    return refined


def merge_points(points, radius=MERGE_RADIUS):
    """Greedy merge in score order: indices kept, and the kept owner of every point."""
    kept, owner = [], []
    for index, (y, x) in enumerate(points):
        hit = next((k for k in kept
                    if (points[k][0] - y) ** 2 + (points[k][1] - x) ** 2 <= radius * radius), None)
        if hit is None:
            kept.append(index)
            owner.append(index)
        else:
            owner.append(hit)
    return kept, owner


def decode_all(prob, offsets):
    points = decode_peaks(prob)
    refined = refine_points(points, offsets)
    kept, owner = merge_points(refined)
    return points, refined, kept, owner


def inspect_image(gt, inst, prob, offsets, decoded, model_key, role, image):
    """Per-object and per-point records for one image under the three decoders."""
    points, refined, kept, owner = decoded
    merged = [refined[k] for k in kept]
    pairs = dict(original=match_points(points, gt), refined=match_points(refined, gt),
                 merged=match_points(merged, gt))
    hits = {name: {int(g) for g, _ in value} for name, value in pairs.items()}
    inst = np.asarray(inst)
    ids = np.unique(inst)
    ids = ids[ids != 0]
    centers = np.asarray([g[1] for g in gt], dtype=float).reshape(-1, 2)
    owners = [int(inst[y, x]) for y, x in points]
    objects = []
    for index, (cls, (y, x), interior, area) in enumerate(gt):
        oid = int(ids[index])
        ys, xs = np.nonzero(inst == oid)
        vote_y, vote_x = ys + offsets[0, ys, xs], xs + offsets[1, ys, xs]
        error = np.hypot(vote_y - y, vote_x - x)
        distance = ((centers - (y, x)) ** 2).sum(1)
        distance[index] = np.inf
        own = [j for j, o in enumerate(owners) if o == oid]
        before = min((math.hypot(points[j][0] - y, points[j][1] - x) for j in own), default=None)
        after = min((math.hypot(refined[j][0] - y, refined[j][1] - x) for j in own), default=None)
        objects.append(dict(model=model_key, role=role, image=int(image), gt_index=index,
            instance_id=oid, cls=int(cls), interior=bool(interior), area=int(area),
            center_y=int(y), center_x=int(x), neighbors12=int((distance <= 144).sum()),
            matched_original=index in hits['original'], matched_refined=index in hits['refined'],
            matched_merged=index in hits['merged'],
            center_vote_error_px=float(math.hypot(offsets[0, y, x], offsets[1, y, x])),
            mean_vote_error_px=float(error.mean()), median_vote_error_px=float(np.median(error)),
            pooled_vote_error_px=float(math.hypot(vote_y.mean() - y, vote_x.mean() - x)),
            points_inside_mask=len(own), nearest_inside_point_before_px=before,
            nearest_inside_point_after_px=after))
    assigned = {name: {int(p): int(g) for g, p in value} for name, value in pairs.items()}
    merged_index = {k: m for m, k in enumerate(kept)}
    point_rows = []
    for j, (y, x) in enumerate(points):
        point_rows.append(dict(model=model_key, role=role, image=int(image), point_index=j, y=y, x=x,
            score=float(prob[y, x]), offset_y=float(offsets[0, y, x]), offset_x=float(offsets[1, y, x]),
            refined_y=refined[j][0], refined_x=refined[j][1], kept=j in merged_index,
            merged_into=int(owner[j]), mask_instance_id=owners[j],
            gt_original=assigned['original'].get(j), gt_refined=assigned['refined'].get(j),
            gt_merged=assigned['merged'].get(merged_index[j]) if j in merged_index else None))
    scores = dict(original=score_points(points, gt), refined=score_points(refined, gt),
                  merged=score_points(merged, gt))
    for name in DECODERS:
        assert scores[name]['matched'] == len(pairs[name])
    return scores, objects, point_rows


# ---------------------------------------------------------------- figures

def render_comparison(image, gt, decoded, path, title):
    """Three large panels: annotation, original points, refined and merged points."""
    from PIL import Image, ImageDraw, ImageFont
    points, refined, kept, _ = decoded
    merged = [refined[k] for k in kept]
    side, gap, left, top = 768, 36, 28, 150
    canvas = Image.new('RGB', (3 * side + 2 * gap + 2 * left, top + side + 110), 'white')
    draw = ImageDraw.Draw(canvas)
    scale = side / image.shape[0]

    def label(position, text, size=2):
        font = ImageFont.load_default()
        box = font.getbbox(text)
        tile = Image.new('RGB', (box[2] - box[0] + 4, box[3] - box[1] + 4), 'white')
        ImageDraw.Draw(tile).text((2 - box[0], 2 - box[1]), text, fill='black', font=font)
        canvas.paste(tile.resize((tile.width * size, tile.height * size), Image.Resampling.NEAREST), position)

    def origin(col):
        return left + col * (side + gap)

    def xy(col, pos):
        return origin(col) + (pos[1] + .5) * scale, top + (pos[0] + .5) * scale

    def ring(col, pos, color, radius=8):
        cx, cy = xy(col, pos)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=3)

    def cross(col, pos, color, radius=7, plus=False):
        cx, cy = xy(col, pos)
        if plus:
            draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=3)
            draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=3)
        else:
            draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, width=3)
            draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=color, width=3)

    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((side, side), Image.Resampling.NEAREST)
    for col in range(3):
        canvas.paste(rgb, (origin(col), top))
    for cls, center, interior, _ in gt:
        ring(0, center, '#ffbf00' if cls == 4 and interior else '#00bbdd')
    counts = []
    for col, shown in ((1, points), (2, merged)):
        pairs = match_points(shown, gt)
        hit, used = {int(g) for g, _ in pairs}, {int(p) for _, p in pairs}
        for index, (_, center, _, _) in enumerate(gt):
            ring(col, center, '#44cc44' if index in hit else '#ffbf00')
        if col == 2:
            for j, (before, after) in enumerate(zip(points, refined)):
                if math.hypot(before[0] - after[0], before[1] - after[1]) >= 1:
                    draw.line(xy(2, before) + xy(2, after), fill='#3366ff', width=2)
                if j not in kept:
                    cross(2, after, '#888888')
        for j, pos in enumerate(shown):
            cross(col, pos, '#44cc44' if j in used else '#ff4444', plus=j in used)
        counts.append((len(shown), len(pairs)))
    label((left, 16), title, 3)
    label((left, 64), 'Rings: annotated centers. Left: yellow = interior Dead, cyan = other nuclei. '
                      'Middle and right: green = matched, yellow = missed.')
    for col, text in enumerate(('Image and annotated centers', 'Original decoder',
                                'Offset-refined and merged decoder')):
        label((origin(col), 112), text)
    label((left, top + side + 14), 'Points: green plus = matched, red cross = extra, grey cross = merged away; '
                                   'blue line = movement of at least one pixel. Matching radius = 3 native pixels.')
    label((left, top + side + 56), f'Annotated nuclei = {len(gt)}; original: {counts[0][0]} points, {counts[0][1]} matched; '
                                   f'refined and merged: {counts[1][0]} points, {counts[1][1]} matched')
    canvas.save(path, format='PNG', dpi=(150, 150))


# ---------------------------------------------------------------- training and evaluation

def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def forward_frozen(model, tap, x, device):
    with torch.no_grad(), autocast(device):
        logits = model(x)
    return logits, tap.features


def train_head(model, tap, head, fold, fit, seed, device, out, epochs=EPOCHS, epoch_limit=None):
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, betas=(.9, .999), weight_decay=WEIGHT_DECAY)
    history, witness, cache = [], {}, {}

    def fields(i):
        if i not in cache:
            centers, weights = center_fields(fold.inst[i], fold.gt[i])
            cache[i] = (centers.astype(np.int32), weights)
        return cache[i][0].astype(np.int64), cache[i][1]

    for epoch in range(epochs):
        head.train()
        factor = .1 + .9 * (1 + math.cos(math.pi * epoch / max(1, epochs - 1))) / 2
        for group in opt.param_groups:
            group['lr'] = HEAD_LR * factor
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 123])).permutation(fit).tolist()
        total, seen, started = 0., 0, time.monotonic()
        for start in range(0, len(order), BATCH):
            selected = order[start:start + BATCH]
            samples = [training_sample(fold.images[i], fold.inst[i], fold.gt[i],
                                       np.random.default_rng(np.random.SeedSequence([seed, epoch, int(i), 987])),
                                       fields(i))
                       for i in selected]
            x = torch.stack([s[0] for s in samples]).to(device)
            target = torch.stack([s[1] for s in samples]).to(device)
            weight = torch.stack([s[2] for s in samples]).to(device)
            objects = sum(s[3] for s in samples)
            _, features = forward_frozen(model, tap, x, device)
            loss = offset_loss(head(features), target, weight, objects)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite offset loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
            if epoch == 0 and start == 0:
                before = head.out.weight.detach().clone()
            opt.step()
            if epoch == 0 and start == 0:
                witness = dict(first_loss=float(loss.detach()), first_gradient_norm=float(norm),
                               first_output_weight_change=float((head.out.weight.detach() - before).abs().max()))
                assert witness['first_gradient_norm'] > 0 and witness['first_output_weight_change'] > 0
            total += float(loss.detach()) * len(selected)
            seen += len(selected)
        row = dict(epoch=epoch + 1, train_loss=total / max(1, seen), lr=HEAD_LR * factor,
                   epoch_seconds=time.monotonic() - started)
        history.append(row)
        with (out / 'loss.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(dict(event='epoch', **row)), flush=True)
        if epoch == 0 and epoch_limit is not None and row['epoch_seconds'] > epoch_limit:
            raise TimeoutError(f"first epoch took {row['epoch_seconds']:.1f}s; limit {epoch_limit}s")
    return history, witness


def evaluate_role(model, tap, head, fold, ids, device, model_key, role, historical=None,
                  save_images=(), out=None, point_images=None):
    head.eval()
    rows = {name: [] for name in DECODERS}
    objects, point_rows, mismatches = [], [], 0
    for start in range(0, len(ids), EVAL_BATCH):
        batch = ids[start:start + EVAL_BATCH]
        x = torch.from_numpy(np.asarray(fold.images[batch]).copy()).permute(0, 3, 1, 2).float().to(device) / 255
        logits, features = forward_frozen(model, tap, x, device)
        with torch.no_grad():
            offsets = head(features).float().cpu().numpy()
        fine = logits.float().cpu().numpy()
        assert np.isfinite(fine).all() and np.isfinite(offsets).all()
        for index, image in enumerate(batch):
            prob = torch.from_numpy(fine[index]).sigmoid().numpy()
            decoded = decode_all(prob, offsets[index])
            scores, records, points = inspect_image(fold.gt[image], fold.inst[image], prob, offsets[index],
                                                    decoded, model_key, role, image)
            if historical is not None:
                differences = {k: [historical[image][k], v] for k, v in scores['original'].items()
                               if historical[image][k] != v}
                mismatches += bool(differences)
            for name in DECODERS:
                rows[name].append(dict(image=int(image), tissue=str(fold.tissue[image]), **scores[name]))
            objects.extend(records)
            if point_images is None or image in point_images:
                point_rows.extend(points)
            if image in save_images:
                base = f'{model_key}_{role}_{image:04d}'
                np.savez_compressed(out / 'maps' / (base + '.npz'), fine_logits=fine[index],
                                    offsets=offsets[index].astype(np.float16))
                render_comparison(fold.images[image], fold.gt[image], decoded, out / 'figures' / (base + '.png'),
                                  f'{model_key}: {role} image {image}, {fold.tissue[image]}')
    return rows, objects, point_rows, mismatches


def error_summary(objects):
    def quantiles(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return None
        return dict(n=int(len(values)), mean=float(values.mean()),
                    **{f'q{int(q * 100)}': float(np.quantile(values, q)) for q in (.5, .75, .9, .95)},
                    within3=float((values <= 3).mean()), within1p5=float((values <= 1.5).mean()))
    groups = {'all': objects, 'interior_dead': [o for o in objects if o['cls'] == 4 and o['interior']]}
    return {name: {field: quantiles([o[field] for o in rows])
                   for field in ('center_vote_error_px', 'mean_vote_error_px', 'pooled_vote_error_px')}
            for name, rows in groups.items()}


def run_model(model_key, seed, model, fold, fit, audit, selection, out, device, historical=None,
              epochs=EPOCHS, epoch_limit=None):
    """Train one head on one frozen detector, then evaluate the three decoders."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('maps', 'figures'):
        (out / folder).mkdir()
    started = time.monotonic()
    model.requires_grad_(False).eval()
    frozen_before = state_fingerprint(model)
    probe = torch.from_numpy(np.asarray(fold.images[audit[:EVAL_BATCH]]).copy()).permute(0, 3, 1, 2).float().to(device) / 255
    with torch.no_grad(), autocast(device):
        plain = model(probe)
    tap = FeatureTap(model)
    hooked, features = forward_frozen(model, tap, probe, device)
    assert torch.equal(plain, hooked), 'hook changed the detector output'
    assert features.shape[1:] == (64, 128, 128), tuple(features.shape)
    torch.manual_seed(seed)
    head = OffsetHead().to(device)
    with torch.no_grad():
        initial = head(features).float().cpu().numpy()
    start_points = decode_peaks(hooked[0].float().sigmoid().cpu().numpy())
    assert refine_points(start_points, initial[0]) == [(float(y), float(x)) for y, x in start_points], \
        'zero-initialized head must leave every peak in place'
    history, witness = train_head(model, tap, head, fold, fit, seed, device, out, epochs, epoch_limit)
    assert history[-1]['train_loss'] < history[0]['train_loss'], 'offset loss did not decrease'
    torch.save(head.state_dict(), out / 'offset_head.pth')
    reloaded = OffsetHead().to(device)
    reloaded.load_state_dict(torch.load(out / 'offset_head.pth', map_location=device, weights_only=True))
    head.eval(), reloaded.eval()
    with torch.no_grad():
        assert torch.equal(head(features), reloaded(features)), 'saved head does not reload exactly'
    save = set(selection['development_images']) | set(selection['training_images'])
    results, summary = {}, {}
    for role, ids, saved_points in (('development', audit, None), ('training', fit, set(selection['training_images']))):
        rows, objects, points, mismatches = evaluate_role(
            model, tap, reloaded, fold, ids, device, model_key, role,
            None if historical is None else historical[role], save, out, saved_points)
        # The development replay is the control of the comparison and must be exact.
        if historical is not None and mismatches and role == 'development':
            raise AssertionError(f'{model_key}: {mismatches} development images differ from the stored original counts')
        results[role] = rows
        for name in DECODERS:
            write_json(out / f'{role}_{name}.json', rows[name])
        write_csv(out / f'{role}_objects.csv', objects)
        write_csv(out / f'{role}_points.csv', points)
        summary[role] = dict(decoders={name: aggregate(rows[name]) for name in DECODERS},
                             counts={name: {k: int(sum(r[k] for r in rows[name]))
                                            for k in ('matched', 'total', 'candidates', 'dead_hit', 'dead_total')}
                                     for name in DECODERS},
                             offset_error=error_summary(objects),
                             original_count_replay_equal=None if historical is None else mismatches == 0,
                             original_count_mismatch_images=None if historical is None else int(mismatches))
    tap.close()
    assert state_fingerprint(model) == frozen_before, 'frozen detector changed'
    summary.update(model=model_key, seed=seed, config=run_config(), witness=witness,
                   head_parameters=sum(p.numel() for p in head.parameters()),
                   head_sha256=sha(out / 'offset_head.pth'), frozen_detector_fingerprint=frozen_before,
                   frozen_detector_unchanged=True, zero_start_identity=True, hook_bit_equal=True,
                   reload_equal=True, epochs=len(history), first_epoch_loss=history[0]['train_loss'],
                   last_epoch_loss=history[-1]['train_loss'],
                   train_seconds=sum(h['epoch_seconds'] for h in history),
                   elapsed_seconds=time.monotonic() - started)
    write_json(out / 'summary.json', summary)
    return results, summary


def paired_decisions(results):
    """Original two-seed rule applied to each frozen detector arm: original versus each new decoder."""
    decisions = {}
    for arm in sorted({key[0] for key in results}):
        for name in ('refined', 'merged'):
            data = {str(seed): dict(F1=results[(arm, seed)]['development']['original'],
                                    T1=results[(arm, seed)]['development'][name])
                    for (a, seed) in results if a == arm}
            decisions[f'{arm}_{name}'] = admission(data)
    return decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from train_proposal_p3 import Fold, WEIGHT_SHA, DATA_SHA, load_adapted
    from p5_contracts import TISSUE_SHA, validate_cohort
    from diagnose_proposal_center_maps import validate_run
    spec = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert spec['experiment'] == EXPERIMENT and spec['fold'] == 1 and spec['config'] == run_config()
    assert [(r['experiment'], r['arm'], r['seed']) for r in spec['runs']] == \
        [('P22', 'C', 19), ('P22', 'C', 1), ('P22', 'M', 19), ('P22', 'M', 1)]
    assert torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.23.5'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info()[0] >= 20 * 1024 ** 3
    torch.set_num_threads(4)
    device = torch.device('cuda')
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
    dead_audit = [i for i in audit if any(g[0] == 4 and g[2] for g in fold.gt[i])]
    selection = dict(development_images=sorted(set(dead_audit) | set(select_examples(audit, fold.gt, 8))),
                     training_images=sorted(select_examples(fit, fold.gt, 4)))
    assert selection == spec['selection'], 'annotation-only image selection changed after freeze'
    # Annotation-only geometry of the training images, recorded before any evaluation.
    nearest = []
    for i in fit:
        centers = np.asarray([g[1] for g in fold.gt[i]], dtype=float).reshape(-1, 2)
        if len(centers) > 1:
            distance = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
            np.fill_diagonal(distance, np.inf)
            nearest.extend(distance.min(1).tolist())
    nearest = np.asarray(nearest)
    geometry = dict(training_objects_with_neighbor=int(len(nearest)),
                    nearest_center_within_merge_radius=int((nearest <= MERGE_RADIUS).sum()),
                    nearest_center_within_3px=int((nearest <= 3).sum()),
                    nearest_center_within_6px=int((nearest <= 6).sum()),
                    nearest_center_q01_px=float(np.quantile(nearest, .01)))
    write_json(out / 'training_center_geometry.json', geometry)
    write_json(out / 'selection.json', selection)
    results, summaries = {}, []
    module = importlib.import_module('train_proposal_p22')
    for run in spec['runs']:
        key = f"P22_{run['arm']}_s{run['seed']}"
        print(json.dumps(dict(event='run_start', model=key)), flush=True)
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
        result, summary = run_model(key, run['seed'], model, fold, fit, audit, selection, out / key, device,
                                    historical, EPOCHS, spec['first_epoch_limit_seconds'])
        summary.update(last_checkpoint_sha256=preservation['last.pth']['sha256'], cohort_sha256=cohort,
                       peak_memory_bytes=torch.cuda.max_memory_allocated())
        results[(run['arm'], run['seed'])] = result
        summaries.append(summary)
        print(json.dumps(dict(event='run_complete', model=key,
                              development=summary['development']['counts'],
                              elapsed_seconds=summary['elapsed_seconds'])), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(out / 'summary.json', dict(experiment=EXPERIMENT, config=run_config(), models=summaries,
                                          decisions=paired_decisions(results), selection=selection,
                                          training_center_geometry=geometry))
    payload = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'preservation_manifest.json', payload)
    write_json(out / 'complete.json', dict(experiment=EXPERIMENT, models=len(summaries), exit_code=0,
                                           elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
