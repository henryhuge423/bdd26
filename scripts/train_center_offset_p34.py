"""P34: center-offset readout trained on frozen, completed P22 detectors.

The detector, its scores, its retained peaks and their order stay fixed. A small
head reads the detector's last 64-channel feature map and predicts, for every
pixel inside an annotated nucleus, the vector to the annotated center. Two
readouts are trained per detector from the same initial weights, with the same
image order and the same rotation, flip and color draws:

* same_augmentation - only the augmentation used to train the detector, which
                      keeps every image at the alignment to the 16-pixel token
                      grid that the detector met during its own training;
* with_translation  - the same, preceded by a random integer translation of up
                      to TRANSLATION pixels per axis with white fill.

Three decoders are compared on identical retained peaks:

* original - the frozen decoder output;
* refined  - each retained peak moved by the offset read at that peak;
* merged   - refined peaks, with later peaks removed when they land within
             MERGE_RADIUS pixels of an earlier, higher-scoring refined peak.

Each invocation handles one detector. Machine paths belong in the ignored JSON
manifest, not in this file.
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
from scipy.ndimage import find_objects
import torch
from torch import nn
import torch.nn.functional as F

from proposal_p3_utils import (decode_peaks, score_points, split_indices, aggregate,
                               admission, augment_pair, gt_objects)
from train_proposal_p21 import match_points, select_examples

EXPERIMENT = 'P34'
EPOCHS = 12
BATCH = 16
EVAL_BATCH = 4
HEAD_LR = 3e-4
WEIGHT_DECAY = .01
MAX_SHIFT = 16.            # one UNI token cell, in native pixels
MERGE_RADIUS = 2.          # native pixels, measured between refined positions
TRANSLATION = 8            # training translation per axis: integers from -8 to 8
DIAGNOSTIC_SHIFT = (5, 3)  # fixed translation (rows, columns) of training images at evaluation
READOUTS = ('same_augmentation', 'with_translation')
NEW_DECODERS = ('refined', 'merged')
ROLES = ('development', 'training', 'training_shifted')
COUNT_KEYS = ('matched', 'total', 'candidates', 'dead_hit', 'dead_total')


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


def run_config(device='cpu'):
    return dict(experiment=EXPERIMENT, epochs=EPOCHS, batch=BATCH, eval_batch=EVAL_BATCH,
                head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY, schedule='cosine_floor0.1',
                loss='per_object_mean_L1', supervision='pixels_inside_annotated_instances',
                readouts=list(READOUTS), shared_augmentation='original_D4_photometric',
                translation_px=TRANSLATION, translation_fill='white',
                diagnostic_shift_px=list(DIAGNOSTIC_SHIFT), max_shift_px=MAX_SHIFT,
                merge_radius_px=MERGE_RADIUS, head='conv3x3-64,conv3x3-64,conv1x1-2,zero_output',
                feature='input_of_detector_output_convolution', detector='frozen',
                precision='cpu_float32' if device == 'cpu' else 'cuda_bfloat16_autocast')


def model_key(run):
    return f"{run['experiment']}_{run['arm']}_s{run['seed']}"


# ---------------------------------------------------------------- geometry and targets

def shift_array(array, dy, dx, fill):
    """Integer translation without wrap-around; vacated pixels take `fill`."""
    array = np.asarray(array)
    result = np.full_like(array, fill)
    height, width = array.shape[:2]
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    if sy0 < sy1 and sx0 < sx1:
        result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = array[sy0:sy1, sx0:sx1]
    return result


def instance_fields(inst, gt=None):
    """Per-pixel linear index of the owning nucleus center (-1 outside), 1/area, object count.

    The center follows the project's annotation rule: the mask pixel nearest to
    the mask centroid. With `gt`, every center and area must equal the stored
    annotation.
    """
    inst = np.asarray(inst).astype(np.int64)
    height, width = inst.shape
    centers = np.full((height, width), -1, np.int64)
    weights = np.zeros((height, width), np.float32)
    count = 0
    for oid, box in enumerate(find_objects(inst), start=1):
        if box is None:
            continue
        ys, xs = np.nonzero(inst[box] == oid)
        ys, xs = ys + box[0].start, xs + box[1].start
        j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
        if gt is not None and (count >= len(gt) or tuple(gt[count][1]) != (int(ys[j]), int(xs[j]))
                               or gt[count][3] != len(ys)):
            raise ValueError('annotated center or area differs from the instance map')
        centers[ys, xs] = ys[j] * width + xs[j]
        weights[ys, xs] = 1. / len(ys)
        count += 1
    if gt is not None and count != len(gt):
        raise ValueError('instance map and annotated objects disagree')
    return centers, weights, count


def training_sample(image, fields, rng=None):
    """Image, (2,H,W) offset target in pixels, per-pixel weight, object count.

    The original augmentation function moves the image, the center-index map and
    a pixel-identity map together; the target is rebuilt from the moved maps, so
    it never depends on a hand-written rotation formula.
    """
    centers, weights, count = fields
    x = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255
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
    return x, target, packed[2].float() * inside, count


def translation_draw(seed, epoch, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(index), 3434]))
    dy, dx = rng.integers(-TRANSLATION, TRANSLATION + 1, size=2)
    return int(dy), int(dx)


def readout_sample(image, inst, gt, seed, epoch, index, translate):
    """One training sample; both readouts share the rotation, flip and color draws."""
    if translate:
        dy, dx = translation_draw(seed, epoch, index)
        image = shift_array(image, dy, dx, 255)
        fields = instance_fields(shift_array(inst, dy, dx, 0))
    else:
        fields = instance_fields(inst, gt)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(index), 987]))
    return training_sample(image, fields, rng)


def offset_loss(pred, target, weight, objects):
    """Mean over objects of the mean absolute offset error (dy plus dx) inside the object."""
    return ((pred.float() - target).abs().sum(1) * weight).sum() / max(1, objects)


class ShiftedView:
    """Translated copies of selected images, annotated again on the moved label maps.

    A nucleus counts as interior only when it was interior in the original image
    and stays clear of the border after the move, so that nuclei cut by the
    original image border never enter the interior group.
    """
    def __init__(self, fold, types, ids, dy, dx):
        self.images, self.inst, self.gt = {}, {}, {}
        for i in ids:
            original = np.asarray(fold.inst[i])
            labels = np.unique(original)
            interior = {int(oid): bool(g[2]) for oid, g in zip(labels[labels != 0], fold.gt[i])}
            self.images[i] = shift_array(fold.images[i], dy, dx, 255)
            self.inst[i] = shift_array(original, dy, dx, 0)
            moved = np.unique(self.inst[i])
            rows = gt_objects(self.inst[i], shift_array(types[i], dy, dx, 0))
            self.gt[i] = [(cls, center, inside and interior[int(oid)], area)
                          for oid, (cls, center, inside, area) in zip(moved[moved != 0], rows)]
        self.tissue = fold.tissue


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


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def forward_frozen(model, tap, x, device):
    with torch.no_grad(), autocast(device):
        logits = model(x)
    return logits, tap.features


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


def decode_readout(points, offsets):
    refined = refine_points(points, offsets)
    kept, owner = merge_points(refined)
    return dict(offsets=offsets, refined=refined, kept=kept, owner=owner)


def inspect_image(gt, inst, prob, points, decoded, key, role, image):
    """Per-image counts, per-object records and per-point records for every decoder."""
    inst = np.asarray(inst)
    ids = np.unique(inst)
    ids = ids[ids != 0]
    if len(ids) != len(gt):
        raise ValueError('instance map and annotated objects disagree')
    shown = {'original': points}
    for name, d in decoded.items():
        shown[(name, 'refined')] = d['refined']
        shown[(name, 'merged')] = [d['refined'][k] for k in d['kept']]
    pairs = {k: match_points(v, gt) for k, v in shown.items()}
    scores = {k: score_points(v, gt) for k, v in shown.items()}
    for k in shown:
        assert scores[k]['matched'] == len(pairs[k])
    hits = {k: {int(g) for g, _ in v} for k, v in pairs.items()}
    centers = np.asarray([g[1] for g in gt], dtype=float).reshape(-1, 2)
    owners = [int(inst[y, x]) for y, x in points]
    objects = []
    for index, (cls, (y, x), interior, area) in enumerate(gt):
        oid = int(ids[index])
        ys, xs = np.nonzero(inst == oid)
        distance = ((centers - (y, x)) ** 2).sum(1)
        distance[index] = np.inf
        own = [j for j, o in enumerate(owners) if o == oid]
        row = dict(model=key, role=role, image=int(image), gt_index=index, instance_id=oid,
                   cls=int(cls), interior=bool(interior), area=int(area), center_y=int(y), center_x=int(x),
                   neighbors12=int((distance <= 144).sum()), points_inside_mask=len(own),
                   matched_original=index in hits['original'],
                   nearest_inside_point_px=min((math.hypot(points[j][0] - y, points[j][1] - x) for j in own),
                                               default=None))
        for name, d in decoded.items():
            off = d['offsets']
            vote_y, vote_x = ys + off[0, ys, xs], xs + off[1, ys, xs]
            row.update({f'matched_refined_{name}': index in hits[(name, 'refined')],
                        f'matched_merged_{name}': index in hits[(name, 'merged')],
                        f'center_vote_error_px_{name}': float(math.hypot(off[0, y, x], off[1, y, x])),
                        f'mean_vote_error_px_{name}': float(np.hypot(vote_y - y, vote_x - x).mean()),
                        f'pooled_vote_error_px_{name}': float(math.hypot(vote_y.mean() - y, vote_x.mean() - x)),
                        f'nearest_inside_point_refined_px_{name}': min(
                            (math.hypot(d['refined'][j][0] - y, d['refined'][j][1] - x) for j in own), default=None)})
        objects.append(row)
    assigned = {k: {int(p): int(g) for g, p in v} for k, v in pairs.items()}
    merged_index = {name: {k: m for m, k in enumerate(d['kept'])} for name, d in decoded.items()}
    point_rows = []
    for j, (y, x) in enumerate(points):
        row = dict(model=key, role=role, image=int(image), point_index=j, y=y, x=x, score=float(prob[y, x]),
                   mask_instance_id=owners[j], gt_original=assigned['original'].get(j))
        for name, d in decoded.items():
            row.update({f'offset_y_{name}': float(d['offsets'][0, y, x]), f'offset_x_{name}': float(d['offsets'][1, y, x]),
                        f'refined_y_{name}': d['refined'][j][0], f'refined_x_{name}': d['refined'][j][1],
                        f'kept_{name}': j in merged_index[name], f'merged_into_{name}': int(d['owner'][j]),
                        f'gt_refined_{name}': assigned[(name, 'refined')].get(j),
                        f'gt_merged_{name}': assigned[(name, 'merged')].get(merged_index[name][j])
                        if j in merged_index[name] else None})
        point_rows.append(row)
    return scores, objects, point_rows


# ---------------------------------------------------------------- figures

def render_comparison(image, gt, points, decoded, path, title):
    """Large panels: annotation, original decoder, and the merged decoder of every readout."""
    from PIL import Image, ImageDraw, ImageFont
    names = {'same_augmentation': 'Offset readout, same augmentation',
             'with_translation': 'Offset readout, translated training'}
    panels = [('Image and annotated centers', None), ('Original decoder', None)]
    panels += [(names.get(name, name), d) for name, d in decoded.items()]
    side, gap, left, top = 640, 36, 28, 150
    canvas = Image.new('RGB', (len(panels) * side + (len(panels) - 1) * gap + 2 * left, top + side + 110), 'white')
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

    def cross(col, pos, color, plus=False, radius=7):
        cx, cy = xy(col, pos)
        if plus:
            draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=3)
            draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=3)
        else:
            draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, width=3)
            draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=color, width=3)

    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((side, side), Image.Resampling.NEAREST)
    counts = []
    for col, (text, d) in enumerate(panels):
        canvas.paste(rgb, (origin(col), top))
        label((origin(col), 112), text)
        if col == 0:
            for cls, center, interior, _ in gt:
                ring(0, center, '#ffbf00' if cls == 4 and interior else '#00bbdd')
            continue
        shown = points if d is None else [d['refined'][k] for k in d['kept']]
        pairs = match_points(shown, gt)
        hit, used = {int(g) for g, _ in pairs}, {int(p) for _, p in pairs}
        for index, (_, center, _, _) in enumerate(gt):
            ring(col, center, '#44cc44' if index in hit else '#ffbf00')
        if d is not None:
            for j, (before, after) in enumerate(zip(points, d['refined'])):
                if math.hypot(before[0] - after[0], before[1] - after[1]) >= 1:
                    draw.line(xy(col, before) + xy(col, after), fill='#3366ff', width=2)
                if j not in d['kept']:
                    cross(col, after, '#888888')
        for j, pos in enumerate(shown):
            cross(col, pos, '#44cc44' if j in used else '#ff4444', plus=j in used)
        counts.append(f'{len(shown)} points, {len(pairs)} matched')
    label((left, 16), title, 3)
    label((left, 64), 'Rings: annotated centers. First panel: yellow = interior Dead, cyan = other nuclei. '
                      'Other panels: green = matched, yellow = missed.')
    label((left, top + side + 14), 'Points: green plus = matched, red cross = extra, grey cross = merged away; '
                                   'blue line = movement of at least one pixel. Matching radius = 3 native pixels.')
    label((left, top + side + 56), f'Annotated nuclei = {len(gt)}; ' + '; '.join(
        f'{text.lower()}: {count}' for (text, _), count in zip(panels[1:], counts)))
    canvas.save(path, format='PNG', dpi=(150, 150))


# ---------------------------------------------------------------- training and evaluation

def train_head(model, tap, head, fold, fit, seed, device, log, translate, epochs=EPOCHS, epoch_limit=None):
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, betas=(.9, .999), weight_decay=WEIGHT_DECAY)
    history, witness = [], {}
    for epoch in range(epochs):
        head.train()
        factor = .1 + .9 * (1 + math.cos(math.pi * epoch / max(1, epochs - 1))) / 2
        for group in opt.param_groups:
            group['lr'] = HEAD_LR * factor
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 123])).permutation(fit).tolist()
        total, seen, started = 0., 0, time.monotonic()
        for start in range(0, len(order), BATCH):
            selected = order[start:start + BATCH]
            samples = [readout_sample(fold.images[i], fold.inst[i], fold.gt[i], seed, epoch, i, translate)
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
        with Path(log).open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(dict(event='epoch', readout='with_translation' if translate else 'same_augmentation',
                              **row)), flush=True)
        if epoch == 0 and epoch_limit is not None and row['epoch_seconds'] > epoch_limit:
            raise TimeoutError(f"first epoch took {row['epoch_seconds']:.1f}s; limit {epoch_limit}s")
    return history, witness


def evaluate_role(model, tap, heads, source, ids, device, key, role, historical=None,
                  save_images=(), out=None, point_images=None, maps=True):
    """One pass of the frozen detector; every readout head decodes the same retained peaks."""
    for head in heads.values():
        head.eval()
    rows = {'original': [], **{name: {decoder: [] for decoder in NEW_DECODERS} for name in heads}}
    objects, point_rows, differing, forward_seconds = [], [], 0, 0.
    stored = {k: 0 for k in COUNT_KEYS}
    for start in range(0, len(ids), EVAL_BATCH):
        batch = ids[start:start + EVAL_BATCH]
        x = torch.from_numpy(np.stack([np.asarray(source.images[i]) for i in batch]))
        x = x.permute(0, 3, 1, 2).float().to(device) / 255
        tick = time.monotonic()
        logits, features = forward_frozen(model, tap, x, device)
        with torch.no_grad():
            offsets = {name: head(features).float().cpu().numpy() for name, head in heads.items()}
        fine = logits.float().cpu().numpy()
        forward_seconds += time.monotonic() - tick
        assert np.isfinite(fine).all() and all(np.isfinite(v).all() for v in offsets.values())
        for index, image in enumerate(batch):
            gt, inst = source.gt[image], source.inst[image]
            prob = torch.from_numpy(fine[index]).sigmoid().numpy()
            points = decode_peaks(prob)
            decoded = {name: decode_readout(points, offsets[name][index]) for name in heads}
            scores, records, points_out = inspect_image(gt, inst, prob, points, decoded, key, role, image)
            if historical is not None:
                differing += any(historical[image][k] != scores['original'][k] for k in COUNT_KEYS)
                for k in COUNT_KEYS:
                    stored[k] += historical[image][k]
            identity = dict(image=int(image), tissue=str(source.tissue[image]))
            rows['original'].append(dict(identity, **scores['original']))
            for name in heads:
                for decoder in NEW_DECODERS:
                    rows[name][decoder].append(dict(identity, **scores[(name, decoder)]))
            objects.extend(records)
            if point_images is None or image in point_images:
                point_rows.extend(points_out)
            if image in save_images:
                base = f'{key}_{role}_{image:04d}'
                if maps:
                    np.savez_compressed(out / 'maps' / (base + '.npz'), fine_logits=fine[index],
                                        **{f'offsets_{name}': offsets[name][index].astype(np.float16) for name in heads})
                render_comparison(source.images[image], gt, points, decoded, out / 'figures' / (base + '.png'),
                                  f'{key}: {role.replace("_", " ")} image {image}, {source.tissue[image]}')
    comparison = None
    if historical is not None:
        comparison = dict(images=len(ids), images_with_different_counts=int(differing), stored_counts=stored,
                          recomputed_counts={k: int(sum(r[k] for r in rows['original'])) for k in COUNT_KEYS})
    return rows, objects, point_rows, dict(comparison=comparison, forward_seconds=forward_seconds)


def totals(rows):
    return {k: int(sum(r[k] for r in rows)) for k in COUNT_KEYS}


def error_summary(objects, name):
    def quantiles(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return None
        return dict(n=int(len(values)), mean=float(values.mean()),
                    **{f'q{int(q * 100)}': float(np.quantile(values, q)) for q in (.5, .75, .9, .95)},
                    within3=float((values <= 3).mean()), within1p5=float((values <= 1.5).mean()))
    groups = {'all': objects, 'interior_dead': [o for o in objects if o['cls'] == 4 and o['interior']]}
    return {group: {field: quantiles([o[f'{field}_{name}'] for o in rows])
                    for field in ('center_vote_error_px', 'mean_vote_error_px', 'pooled_vote_error_px')}
            for group, rows in groups.items()}


def run_detector(key, seed, model, fold, types, fit, audit, selection, out, device, historical=None,
                 epochs=EPOCHS, epoch_limit=None):
    """Train both readouts on one frozen detector, then evaluate all decoders."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    for folder in ('maps', 'figures', 'preflight/figures'):
        (out / folder).mkdir(parents=True)
    started = time.monotonic()
    model.requires_grad_(False).eval()
    frozen_before = state_fingerprint(model)
    probe = torch.from_numpy(np.stack([np.asarray(fold.images[i]) for i in audit[:EVAL_BATCH]]))
    probe = probe.permute(0, 3, 1, 2).float().to(device) / 255
    with torch.no_grad(), autocast(device):
        plain = model(probe)
    tap = FeatureTap(model)
    hooked, features = forward_frozen(model, tap, probe, device)
    assert torch.equal(plain, hooked), 'hook changed the detector output'
    assert features.shape[1:] == (64, 128, 128), tuple(features.shape)
    shifted = ShiftedView(fold, types, selection['training_images'], *DIAGNOSTIC_SHIFT)
    sources = dict(development=(fold, audit), training=(fold, selection['training_images']),
                   training_shifted=(shifted, selection['training_images']))
    figures = dict(development=set(selection['development_images']), training=set(selection['training_figures']),
                   training_shifted=set(selection['training_figures']))

    def sweep(heads, target, maps):
        results, described = {}, {}
        for role in ROLES:
            source, ids = sources[role]
            stored = None if historical is None or role not in historical else historical[role]
            rows, objects, points, info = evaluate_role(model, tap, heads, source, ids, device, key, role,
                                                        stored, figures[role], target, maps=maps)
            results[role] = rows
            write_json(target / f'{role}_original.json', rows['original'])
            for name in READOUTS:
                for decoder in NEW_DECODERS:
                    write_json(target / f'{role}_{name}_{decoder}.json', rows[name][decoder])
            write_csv(target / f'{role}_objects.csv', objects)
            write_csv(target / f'{role}_points.csv', points)
            described[role] = dict(
                images=len(ids), original=dict(counts=totals(rows['original']), rates=aggregate(rows['original'])),
                readouts={name: dict({decoder: dict(counts=totals(rows[name][decoder]),
                                                    rates=aggregate(rows[name][decoder])) for decoder in NEW_DECODERS},
                                     offset_error=error_summary(objects, name)) for name in READOUTS},
                stored_original_comparison=info['comparison'],
                forward_seconds_per_image=info['forward_seconds'] / max(1, len(ids)))
        return results, described

    # Before any training: the annotation rule holds on every training image, and
    # zero-initialized heads leave every retained peak of every evaluated image in place.
    for i in fit:
        instance_fields(fold.inst[i], fold.gt[i])
    blank = {name: OffsetHead().to(device) for name in READOUTS}
    before, preflight = sweep(blank, out / 'preflight', False)
    for role in ROLES:
        for name in READOUTS:
            assert before[role][name]['refined'] == before[role]['original'], 'zero offsets changed the counts'
            assert all(m['candidates'] <= o['candidates']
                       for m, o in zip(before[role][name]['merged'], before[role]['original']))
    print(json.dumps(dict(event='preflight', model=key, seconds=time.monotonic() - started,
                          development=preflight['development']['original']['counts'],
                          stored_comparison=preflight['development']['stored_original_comparison'],
                          forward_seconds_per_image=preflight['development']['forward_seconds_per_image'])), flush=True)
    del blank, before

    heads, training = {}, {}
    for name in READOUTS:
        torch.manual_seed(seed)
        head = OffsetHead().to(device)
        history, witness = train_head(model, tap, head, fold, fit, seed, device, out / f'loss_{name}.jsonl',
                                      name == 'with_translation', epochs, epoch_limit)
        torch.save(head.state_dict(), out / f'offset_head_{name}.pth')
        reloaded = OffsetHead().to(device)
        reloaded.load_state_dict(torch.load(out / f'offset_head_{name}.pth', map_location=device, weights_only=True))
        head.eval(), reloaded.eval()
        with torch.no_grad():
            assert torch.equal(head(features), reloaded(features)), 'saved head does not reload exactly'
        heads[name] = reloaded
        training[name] = dict(witness=witness, epochs=len(history), first_epoch_loss=history[0]['train_loss'],
                              last_epoch_loss=history[-1]['train_loss'],
                              loss_decreased=history[-1]['train_loss'] < history[0]['train_loss'],
                              train_seconds=sum(h['epoch_seconds'] for h in history),
                              head_sha256=sha(out / f'offset_head_{name}.pth'))

    results, roles = sweep(heads, out, True)
    tap.close()
    assert state_fingerprint(model) == frozen_before, 'frozen detector changed'
    summary = dict(model=key, seed=seed, config=run_config(device.type), training=training, roles=roles,
                   preflight=preflight, training_images=len(fit),
                   head_parameters=sum(p.numel() for p in heads[READOUTS[0]].parameters()),
                   frozen_detector_fingerprint=frozen_before, frozen_detector_unchanged=True,
                   zero_start_identity=True, hook_bit_equal=True, reload_equal=True,
                   annotation_rule_verified_images=len(fit), elapsed_seconds=time.monotonic() - started)
    write_json(out / 'summary.json', summary)
    return results, summary


def paired_decisions(results):
    """Original two-seed rule on every frozen detector arm: original decoder versus each new decoder.

    `results[(arm, seed)]` holds the development rows: 'original' and, per readout, each new decoder.
    """
    decisions = {}
    for arm in sorted({a for a, _ in results}):
        for name in READOUTS:
            for decoder in NEW_DECODERS:
                data = {str(seed): dict(F1=results[(a, seed)]['original'], T1=results[(a, seed)][name][decoder])
                        for (a, seed) in results if a == arm}
                decisions[f'{arm}:{name}:{decoder}'] = admission(data)
    return decisions


def summarize(root, runs):
    """Combine the finished detectors: per-detector summaries and the paired two-seed decisions.

    Reads only the small per-detector result files, so it also runs away from
    the compute node on a verified copy of those files.
    """
    root = Path(root)
    results, summaries = {}, []
    for run in runs:
        folder = root / model_key(run)
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['exit_code'] != 0 or complete['epochs'] != EPOCHS or complete['model'] != model_key(run):
            raise ValueError(f'{model_key(run)} is not a completed {EXPERIMENT} run')
        rows = {'original': json.loads((folder / 'development_original.json').read_text())}
        for name in READOUTS:
            rows[name] = {decoder: json.loads((folder / f'development_{name}_{decoder}.json').read_text())
                          for decoder in NEW_DECODERS}
        results[(run['arm'], run['seed'])] = rows
        summaries.append(json.loads((folder / 'summary.json').read_text()))
    combined = dict(experiment=EXPERIMENT, detectors=summaries, decisions=paired_decisions(results))
    write_json(root / 'summary.json', combined)
    return combined


def training_center_geometry(fold, fit):
    """Annotation-only distances between neighbouring centers in the training images."""
    nearest = []
    for i in fit:
        centers = np.asarray([g[1] for g in fold.gt[i]], dtype=float).reshape(-1, 2)
        if len(centers) > 1:
            distance = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
            np.fill_diagonal(distance, np.inf)
            nearest.extend(distance.min(1).tolist())
    nearest = np.asarray(nearest)
    return dict(training_objects_with_neighbor=int(len(nearest)),
                nearest_center_within_merge_radius=int((nearest <= MERGE_RADIUS).sum()),
                nearest_center_within_3px=int((nearest <= 3).sum()),
                nearest_center_within_6px=int((nearest <= 6).sum()),
                nearest_center_q01_px=float(np.quantile(nearest, .01)))


def frozen_selection(fold, fit, audit):
    """Image lists fixed by annotation alone, identical to the lists of the earlier diagnostics."""
    def dead(i):
        return any(g[0] == 4 and g[2] for g in fold.gt[i])
    examples = select_examples(fit, fold.gt, 4)
    fit_dead = [i for i in fit if dead(i)]
    return dict(development_images=sorted({i for i in audit if dead(i)} | set(select_examples(audit, fold.gt, 8))),
                training_images=fit_dead + sorted(set(examples) - set(fit_dead)),
                training_figures=sorted(examples))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-index', type=int, help='train and evaluate this detector of the manifest')
    mode.add_argument('--summarize', action='store_true', help='combine the four finished detectors')
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
    types = np.load(data_root / 'fold1/type.npy', mmap_mode='r')
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
    _, summary = run_detector(key, run['seed'], model, fold, types, fit, audit, selection, out, device,
                              historical, EPOCHS, spec['first_epoch_limit_seconds'])
    write_json(out / 'provenance.json', dict(
        model=key, last_checkpoint_sha256=preservation['last.pth']['sha256'],
        last_checkpoint_bytes=preservation['last.pth']['bytes'], cohort_sha256=cohort, weights_sha256=weights_digest,
        input_sha256=inputs, selection=selection, training_center_geometry=training_center_geometry(fold, fit),
        script_sha256=sha(__file__), torch=torch.__version__, numpy=np.__version__, threads=spec['threads']))
    payload = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'preservation_manifest.json', payload)
    write_json(out / 'complete.json', dict(experiment=EXPERIMENT, model=key, epochs=EPOCHS, readouts=len(READOUTS),
                                           exit_code=0, elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(event='run_complete', model=key,
                          development={'original': summary['roles']['development']['original']['counts'],
                                       **{name: summary['roles']['development']['readouts'][name]['merged']['counts']
                                          for name in READOUTS}},
                          elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
