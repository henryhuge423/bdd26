"""Reproduce P18 small-payload checks, aggregates and original image bootstrap.

Run with pinned project Python/NumPy (original NumPy 1.23.5):
    python analyze_p18_independent.py
Default checks the adjacent independent JSON and prints the reproduced evidence.
Use --out PATH to additionally save the reproduced machine-readable record.
No torch/scipy import, remote access, model inference or admission import occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def aggregate(rows):
    sums = {k: sum(r[k] for r in rows) for k in
            ('matched', 'total', 'candidates', 'dead_hit', 'dead_total', 'small_hit', 'small_total')}
    positive = [r for r in rows if r['dead_total']]
    return dict(**sums, recall=sums['matched']/sums['total'],
                precision=sums['matched']/sums['candidates'],
                dead_recall=sums['dead_hit']/sums['dead_total'] if sums['dead_total'] else None,
                dead_images=len(positive),
                nondeg=sum(r['candidates'] > 0 for r in positive)/len(positive) if positive else None,
                empty_images=sum(r['candidates'] == 0 for r in rows),
                median_candidates=float(np.median([r['candidates'] for r in rows])),
                max_candidates=max(r['candidates'] for r in rows),
                cap_images=sum(r.get('saturated_512', False) for r in rows))


def bootstrap_and_gate(baseline, treatment):
    assert [(r['image'], r['dead_total']) for r in baseline] == [
        (r['image'], r['dead_total']) for r in treatment]
    groups = {}
    for i, row in enumerate(baseline):
        if row['dead_total']:
            groups.setdefault(str(row['tissue']), []).append(i)
    rng = np.random.default_rng(20261018)
    draws = []
    for _ in range(1000):
        ids = np.concatenate([rng.choice(indices, len(indices), replace=True)
                              for tissue, indices in sorted(groups.items())])
        draws.append(sum(treatment[i]['dead_hit'] - baseline[i]['dead_hit'] for i in ids)
                     / sum(baseline[i]['dead_total'] for i in ids))
    interval = np.percentile(draws, [2.5, 97.5]).tolist()
    b, t = aggregate(baseline), aggregate(treatment)
    delta = t['dead_recall'] - b['dead_recall']
    checks = dict(support=t['dead_images'] >= 10 and t['dead_total'] >= 100,
                  absolute_dead_recall=t['dead_recall'] >= .60, gain=delta >= .05,
                  positive_bootstrap_lower=interval[0] > 0,
                  absolute_precision=t['precision'] >= .75,
                  precision_noninferiority=t['precision'] >= b['precision'] - .02,
                  overall_recall_noninferiority=t['recall'] >= b['recall'] - .02,
                  nondegeneracy=t['nondeg'] >= .95)
    images = [dict(image=x['image'], tissue=x['tissue'], dead_total=x['dead_total'],
                   C_hits=x['dead_hit'], H_hits=y['dead_hit'], net=y['dead_hit']-x['dead_hit'])
              for x, y in zip(baseline, treatment) if x['dead_total']]
    tissues = [dict(tissue=name, images=sum(x['tissue'] == name for x in images),
                    **{key: sum(x[key] for x in images if x['tissue'] == name)
                       for key in ('dead_total', 'C_hits', 'H_hits', 'net')})
               for name in sorted(groups)]
    return dict(delta=delta, ci95=interval, net_dead_hits=t['dead_hit']-b['dead_hit'],
                checks=checks, pass_seed=all(checks.values()), dead_images=images,
                dead_tissues=tissues, positive_net_images=sum(x['net'] > 0 for x in images),
                negative_net_images=sum(x['net'] < 0 for x in images),
                unchanged_images=sum(x['net'] == 0 for x in images),
                recall_delta=t['recall']-b['recall'], precision_delta=t['precision']-b['precision'])


def analyze(root):
    collection_path = root/'p18_collection_final.json'
    collection = json.loads(collection_path.read_text(encoding='utf-8'))
    verified = json.loads((root/'p18_verified_final.json').read_text(encoding='utf-8'))
    files = collection['files']
    hashes = {}
    for key, expected in collection['verified'].items():
        payload = files[key].encode('utf-8')
        actual = hashlib.sha256(payload).hexdigest()
        assert len(payload) == expected['bytes'] and actual == expected['sha256'], key
        hashes[key] = actual
    assert len(hashes) == 36
    source_sha = hashlib.sha256(files['source_hashes.json'].encode('utf-8')).hexdigest()
    assert source_sha == verified['source_manifest_sha256']
    load = lambda key: json.loads(files[key])
    identities, runs, raw = [], {}, {}
    for seed in (19, 1):
        for arm in ('C', 'H'):
            tag = f'A_{arm}_s{seed}'
            rows, fit = load(tag+'/evaluation.json'), load(tag+'/fit.json')
            manifest = load(tag+'/manifest.json')
            preservation = load(tag+'/preservation_manifest.json')
            for name in ('evaluation.json', 'fit.json', 'loss.jsonl', 'manifest.json',
                         'status.json', 'summary.json', 'complete.json'):
                payload = files[tag+'/'+name].encode('utf-8')
                assert len(payload) == preservation[name]['bytes']
                assert hashlib.sha256(payload).hexdigest() == preservation[name]['sha256']
            config = manifest['config']
            assert config['arm'] == arm and config['seed'] == seed
            assert config['trainable_encoder_blocks'] == list(range(12, 24))
            assert config['target_sigma'] == 1 and config['target_radius'] == 3
            assert config['augmentation'] == {'kind': 'original'}
            assert len(rows) == 522 and len(fit) == 2134
            assert [r['image'] for r in rows] == manifest['audit']
            assert [r['image'] for r in fit] == manifest['fit']
            assert not set(manifest['fit']) & set(manifest['audit'])
            identities.append({k: manifest[k] for k in
                               ('weights_sha256', 'input_sha256', 'cohort_sha256')})
            logs = [json.loads(line) for line in files[tag+'/loss.jsonl'].splitlines()]
            assert [x['epoch'] for x in logs] == list(range(1, 131))
            complete = load(tag+'/complete.json')
            assert complete['epochs'] == 130 and complete['exit_code'] == 0
            audit, fitting = aggregate(rows), aggregate(fit)
            for kind, values in (('audit', audit), ('fit', fitting)):
                for metric in ('recall', 'precision', 'dead_recall', 'nondeg'):
                    assert values[metric] == verified['runs'][f'{arm}_{seed}'][kind][metric]
            saved_summary = load(tag+'/summary.json')
            assert all(audit[k] == value for k, value in saved_summary.items()
                       if k != 'dead_objects')
            assert audit['dead_total'] == saved_summary['dead_objects']
            training = sum(x['training_seconds'] for x in logs)
            checkpoints = sum(x['checkpoint_seconds'] for x in logs)
            elapsed = complete['elapsed_seconds']
            runs[f'{arm}_{seed}'] = dict(audit=audit, fit=fitting,
                timing=dict(training_seconds=training, epoch_checkpoint_seconds=checkpoints,
                            complete_elapsed_seconds=elapsed,
                            final_epoch_elapsed_seconds=logs[-1]['elapsed_seconds'],
                            other_seconds=elapsed-training-checkpoints,
                            checkpoint_fraction_of_complete=checkpoints/elapsed,
                            mean_training_epoch_seconds=training/130),
                tissues={t: aggregate([r for r in rows if r['tissue'] == t])
                         for t in sorted({r['tissue'] for r in rows})},
                peak_allocated_bytes=max(x['peak_memory'] for x in logs))
            raw[f'{arm}_{seed}'] = rows
    assert all(x == identities[0] for x in identities)
    pairs = {str(seed): bootstrap_and_gate(raw[f'C_{seed}'], raw[f'H_{seed}']) for seed in (19, 1)}
    for seed, pair in pairs.items():
        expected = verified['admission']['details'][seed]
        for key in ('delta', 'ci95', 'pass_seed'):
            assert pair[key] == expected[key], (seed, key)
    record = dict(experiment='P18', independent_reimplementation=True,
        admission_function_imported=False, input_collection_sha256=hashlib.sha256(collection_path.read_bytes()).hexdigest(),
        source_manifest_sha256=source_sha, small_payloads_independently_hashed=len(hashes),
        small_payload_hashes=hashes, identity=identities[0], runs=runs, pairs=pairs,
        original_admission_pass=all(x['pass_seed'] for x in pairs.values()),
        descriptive_two_seed_mean_delta=float(np.mean([x['delta'] for x in pairs.values()])),
        large_checkpoints_independently_rehashed=False,
        object_identity_crossover_unavailable=True)
    record.update(
        accounting=collection['sacct'],
        bootstrap=dict(replicates=1000, seed=20261018,
            unit='tissue-stratified Dead-positive image, sampling with replacement within sorted tissue groups',
            percentiles=[2.5,97.5]),
        comparison_json_sha256=hashlib.sha256((root/'p18_verified_final.json').read_bytes()).hexdigest(),
        config_and_preservation_checks='Both seeds C/H arm, seed, adapted blocks12..23, original augmentation and sigma/radius; seven per-run payload hashes independently agree with preservation manifests.',
        diagnosis='P18 preserves high fitting coverage and modestly improves overall audit point recall, but fails original Dead absolute recall/gain/CI/precision clauses in both seeds; closes exact RGB residual recipe. P19 is independent effective-batch optimization recipe, not continuation or rescue.',
        source_files_collection_verified=collection['source_files_verified'])
    independent = root/'p18_independent_analysis.json'
    if independent.exists():
        prior = json.loads(independent.read_text(encoding='utf-8'))
        assert prior == record, 'complete independent record differs'
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(json.dumps(dict(status='PASS', payloads=result['small_payloads_independently_hashed'],
        admission=result['original_admission_pass'],
        pairs={seed: {k: row[k] for k in ('delta', 'ci95', 'pass_seed')}
               for seed, row in result['pairs'].items()}), indent=2))
