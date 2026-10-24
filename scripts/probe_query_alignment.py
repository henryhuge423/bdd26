"""Fixed-query, classifier-only train-fold mechanism probe. Never saves trained weights."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def alignment_loss(logits, labels, balanced):
    if logits.ndim != 2 or labels.shape != logits.shape[:1] or labels.dtype != torch.long:
        raise ValueError('Expected flat query logits and integer labels')
    if not len(labels) or int(labels.min()) < 0 or int(labels.max()) >= logits.shape[-1]:
        raise ValueError('Empty or invalid labels')
    fg = labels > 0
    if not bool(fg.any()) or bool(fg.all()):
        raise ValueError('Both foreground and background support required')
    if balanced:
        values = F.cross_entropy(logits.float(), labels, reduction='none')
        return values[fg].mean() + values[~fg].mean()
    weight = logits.new_ones(logits.shape[-1], dtype=torch.float32)
    weight[0] = .1
    return 2 * F.cross_entropy(logits.float(), labels, weight=weight)


def head_logits(head, features):
    context = torch.autocast('cuda', dtype=torch.bfloat16) if features.is_cuda else nullcontext()
    with context:
        return head(features)


def fit_classifier(original, features, labels, balanced, steps=300, lr=3e-4):
    features, labels = features.detach(), labels.detach()
    head = deepcopy(original)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, betas=(.85, .95), weight_decay=1e-4)
    initial = head_logits(head, features).detach().cpu()
    rows = [{'step': 0, 'loss': float(alignment_loss(initial.reshape(-1, initial.shape[-1]), labels.cpu().flatten(), balanced)),
             'initial_logits': initial}]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head_logits(head, features)
        loss = alignment_loss(logits.reshape(-1, logits.shape[-1]), labels.flatten(), balanced)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite classifier loss')
        loss.backward()
        if any(not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
            raise ValueError('Nonfinite classifier gradient')
        optimizer.step()
        if step in (100, steps):
            with torch.no_grad():
                logits = head_logits(head, features)
                value = alignment_loss(logits.reshape(-1, logits.shape[-1]), labels.flatten(), balanced)
            rows.append({'step': step, 'loss': float(value)})
    return head.eval(), rows


def allocation(results):
    details = {}
    for seed in ('19', '1'):
        base, changed = results[seed]['A0']['audit'], results[seed]['A2']
        gain = changed['audit']['best_final_recall50'] - base['best_final_recall50']
        details[seed] = {'audit_final_gain': gain,
                         'audit_fg_change': changed['audit']['foreground_accuracy'] - base['foreground_accuracy'],
                         'fit_fg_accuracy': changed['fit']['foreground_accuracy']}
    passed = all(d['audit_final_gain'] >= .05 and d['audit_fg_change'] >= 0 and
                 d['fit_fg_accuracy'] >= .90 for d in details.values())
    return {'promising': passed, 'per_seed': details,
            'meaning': 'Train-only allocation rule; never a method or generalization claim'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(caches):
    h = hashlib.sha256()
    for cache in caches.values():
        for tensor in (cache['features'], cache['labels'], cache['pred']['mask_embed'], cache['pred']['mask_features']):
            h.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def cache_batch(model, batch):
    from nucseg.setmask.engine import amp
    from nucseg.setmask.matching import make_targets, match_queries
    captured = []
    hook = model.query_head.class_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with amp('cuda'):
            pred = model(batch['img'])
    finally:
        hook.remove()
    features = captured[0]
    if not torch.equal(head_logits(model.query_head.class_head, features), pred['class_logits']):
        raise ValueError('Cached classifier does not reproduce original bf16 logits exactly')
    targets = make_targets(batch['inst_map'], batch['tp_map'])
    pairs = match_queries(pred['class_logits'], pred['mask_embed'], pred['mask_features'], targets)
    labels = torch.zeros(pred['class_logits'].shape[:2], dtype=torch.long, device='cuda')
    ious = []
    for b, (target, (qi, ti)) in enumerate(zip(targets, pairs)):
        labels[b, qi.to('cuda')] = target['classes'][ti].to('cuda')
        truth = (target['inst'][None] == target['ids'][:, None, None]).to('cuda').flatten(1).float()
        area = truth.sum(1)
        chunks = []
        for start in range(0, pred['mask_embed'].shape[1], 16):
            raw = (torch.einsum('qc,chw->qhw', pred['mask_embed'][b, start:start+16].float(),
                               pred['mask_features'][b].float()) / np.sqrt(128) >= 0).flatten(1).float()
            inter = raw @ truth.T
            chunks.append(inter / (raw.sum(1)[:, None] + area[None] - inter).clamp_min(1))
        ious.append(torch.cat(chunks))
    return {'features': features, 'labels': labels, 'pred': {k: v.detach() for k, v in pred.items()},
            'targets': targets, 'ious': ious, 'batch_digests': batch['digest']}


@torch.no_grad()
def measure_classifier(head, cache):
    from nucseg.setmask.decode import decode_set
    from nucseg.visibility_probe import max_iou_table
    logits = head_logits(head, cache['features'])
    labels = cache['labels']; fg = labels > 0
    guess = logits.argmax(-1)
    score, types = logits.float().softmax(-1).max(-1)
    inst, _ = decode_set({**cache['pred'], 'class_logits': logits})
    raw = kept = final = count = accepted = 0
    class_rows = {str(c): {'n': 0, 'best_final_good': 0} for c in range(1, 6)}
    for b, (target, iou) in enumerate(zip(cache['targets'], cache['ious'])):
        n = len(target['ids']); count += n
        chosen = (types[b] > 0) & (score[b] >= .5); accepted += int(chosen.sum())
        raw += int((iou.max(0).values > .5).sum()) if n else 0
        kept += int((iou[chosen].max(0).values > .5).sum()) if n and bool(chosen.any()) else 0
        best = max_iou_table(target['inst'].numpy(), inst[b])
        final += sum(u > .5 for u in best.values())
        for oid, cls in zip(target['ids'].tolist(), target['classes'].tolist()):
            class_rows[str(cls)]['n'] += 1
            class_rows[str(cls)]['best_final_good'] += int(best[oid] > .5)
    return {'n_targets': count, 'foreground_accuracy': float((guess[fg] == labels[fg]).float().mean()),
            'background_accuracy': float((guess[~fg] == 0).float().mean()),
            'original_weighted_ce': float(alignment_loss(logits.flatten(0, 1), labels.flatten(), False)),
            'best_raw_recall50': raw / count, 'best_accepted_raw_recall50': kept / count,
            'best_final_recall50': final / count, 'best_final_good': final,
            'accepted_queries_per_image': accepted / len(inst), 'class_support': class_rows}


def run_probe(snapshot, run_root, out):
    from torch.utils.data import default_collate
    sys.path.insert(0, str(snapshot / 'src')); os.chdir(snapshot)
    from nucseg.setmask.engine import load_config, build_model, move_batch
    from nucseg.setmask.data import PairedDataset, sample_keys
    from nucseg.cellvit.data import cell_tissue_weights
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one admitted GPU required')
    torch.cuda.set_per_process_memory_fraction(.30)
    if shutil.disk_usage(out.parent).free < 20 * 1024**3:
        raise ValueError('20GiB disk reserve required')
    out.mkdir(exist_ok=False)
    ds = PairedDataset(1, True, 19)
    keys = sample_keys(cell_tissue_weights(ds, .85), 19, 0)[:32]
    batches = {name: move_batch(default_collate([ds[k] for k in keys[start:start+16]]), 'cuda', 'Q')
               for name, start in (('fit', 0), ('audit', 16))}
    if set(k[0] for k in keys[:16]) & set(k[0] for k in keys[16:]):
        raise ValueError('Fit/audit original images overlap; do not silently resample')
    manifest = {'fold': 1, 'data_seed': 19, 'sample_keys': keys, 'steps': 300,
                'arms': ['A0', 'A1', 'A2'], 'script_sha256': sha(__file__),
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'checkpoints': {},
                'limits': 'Train fold only; fixed feature/mask/assignment; no saved trained weights; no performance claim'}
    results = {}; started = time.monotonic()
    for seed in (19, 1):
        run = run_root / f'Q_seed{seed}'; cfg = load_config(run)
        if cfg.arm != 'Q' or cfg.seed != seed or cfg.round != 'E7':
            raise ValueError('Expected E7 Q final checkpoint')
        prov = json.loads((run / 'provenance.json').read_text())
        receipt = json.loads((run / 'completion.json').read_text())
        for name in ('config.json', 'provenance.json', 'log.jsonl', 'final.pth'):
            if sha(run / name) != receipt['hashes'][name]:
                raise ValueError('Checkpoint/completion identity differs: ' + name)
        for name, digest in prov['sources'].items():
            if sha(snapshot / name) != digest:
                raise ValueError('Frozen source differs: ' + name)
        for name in ('images', 'inst', 'type', 'tissue'):
            if sha(ds.fold.dir / (name + '.npy')) != prov['inputs']['fold1/' + name + '.npy']:
                raise ValueError('Training input differs')
        if sha(cfg.uni_ckpt) != prov['inputs']['UNI'] or torch.__version__ != prov['torch'] or np.__version__ != prov['numpy']:
            raise ValueError('Pinned weight/environment identity differs')
        state = torch.load(run / 'final.pth', map_location='cpu', weights_only=False, mmap=True)
        if state['config'] != asdict(cfg) or state['provenance'] != prov:
            raise ValueError('Final metadata differs')
        model = build_model(cfg, pretrained=False).eval().to('cuda')
        model.load_state_dict(state['model']); del state
        caches = {name: cache_batch(model, batch) for name, batch in batches.items()}
        head = deepcopy(model.query_head.class_head); del model
        torch.cuda.empty_cache()
        for cache in caches.values():
            if not torch.equal(head_logits(head, cache['features']), cache['pred']['class_logits']):
                raise ValueError('Copied classifier does not start from original 3D logits')
        before = fingerprint(caches); initial = deepcopy(head.state_dict())
        arms = {'A0': {name: measure_classifier(head, cache) for name, cache in caches.items()}}
        for arm, balanced in (('A1', False), ('A2', True)):
            fitted, rows = fit_classifier(head, caches['fit']['features'],
                                          caches['fit']['labels'], balanced)
            if not torch.equal(rows[0]['initial_logits'], caches['fit']['pred']['class_logits'].cpu()):
                raise ValueError('Arms do not start from original logits')
            del rows[0]['initial_logits']
            arms[arm] = {name: measure_classifier(fitted, cache) for name, cache in caches.items()}
            arms[arm]['optimization'] = rows
            if any(not torch.equal(v, initial[k]) for k, v in head.state_dict().items()) or fingerprint(caches) != before:
                raise ValueError('Frozen model/cache changed')
            del fitted
        manifest['checkpoints'][str(seed)] = receipt['hashes']['final.pth']
        manifest.setdefault('batch_digests', {})[str(seed)] = {k: c['batch_digests'] for k, c in caches.items()}
        results[str(seed)] = arms
        with (out / f'seed{seed}.json').open('x') as f:
            json.dump(arms, f, indent=2); f.write('\n')
        print(json.dumps({'seed': seed, 'arms': arms}), flush=True)
        del caches, head; torch.cuda.empty_cache()
        if time.monotonic() - started > 1100:
            raise ValueError('Probe wall budget exhausted')
    payload = {'manifest': manifest, 'results': results, 'allocation': allocation(results),
               'elapsed_seconds': time.monotonic()-started,
               'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3}
    with (out / 'results.json').open('x') as f:
        json.dump(payload, f, indent=2); f.write('\n')
    print(json.dumps(payload['allocation']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run_probe(a.snapshot.resolve(), a.run_root.resolve(), a.out.resolve())
