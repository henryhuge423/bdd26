"""Standalone NumPy reproduction; no torch, model inference or original gate import.

Place next to P19_COLLECTION.json and P19_NUMBERS.json, then run this file.
It checks all small-file hashes, preservation, identities, aggregate endpoints and
the original 1000-draw tissue/image bootstrap against the published numbers.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def aggregate(rows):
    s={k:sum(r[k] for r in rows) for k in
       ('matched','total','candidates','dead_hit','dead_total','small_hit','small_total')}
    positive=[r for r in rows if r['dead_total']]
    return dict(**s,recall=s['matched']/s['total'],precision=s['matched']/s['candidates'],
        dead_recall=s['dead_hit']/s['dead_total'] if s['dead_total'] else None,
        dead_images=len(positive),nondeg=sum(r['candidates']>0 for r in positive)/len(positive) if positive else None,
        empty_images=sum(r['candidates']==0 for r in rows),median_candidates=float(np.median([r['candidates'] for r in rows])),
        max_candidates=max(r['candidates'] for r in rows),cap_images=sum(r.get('saturated_512',False) for r in rows))


def bootstrap(b,t):
    assert [(r['image'],r['dead_total']) for r in b]==[(r['image'],r['dead_total']) for r in t]
    groups={}
    for i,r in enumerate(b):
        if r['dead_total']:groups.setdefault(str(r['tissue']),[]).append(i)
    rng=np.random.default_rng(20261018);draws=[]
    for _ in range(1000):
        ids=np.concatenate([rng.choice(v,len(v),replace=True) for _,v in sorted(groups.items())])
        draws.append(sum(t[i]['dead_hit']-b[i]['dead_hit'] for i in ids)/sum(b[i]['dead_total'] for i in ids))
    ba,ta=aggregate(b),aggregate(t);ci=np.percentile(draws,[2.5,97.5]).tolist();delta=ta['dead_recall']-ba['dead_recall']
    checks=dict(support=ta['dead_images']>=10 and ta['dead_total']>=100,
        absolute_dead_recall=ta['dead_recall']>=.60,gain=delta>=.05,positive_bootstrap_lower=ci[0]>0,
        absolute_precision=ta['precision']>=.75,precision_noninferiority=ta['precision']>=ba['precision']-.02,
        overall_recall_noninferiority=ta['recall']>=ba['recall']-.02,nondegeneracy=ta['nondeg']>=.95)
    return dict(delta=delta,ci95=ci,checks=checks,pass_seed=all(checks.values()),
        net_dead_hits=ta['dead_hit']-ba['dead_hit'],recall_delta=ta['recall']-ba['recall'],precision_delta=ta['precision']-ba['precision'])


def main():
    root=Path(__file__).resolve().parent
    path=root/'P19_COLLECTION.json';c=json.loads(path.read_text(encoding='utf-8'))
    expected=json.loads((root/'P19_NUMBERS.json').read_text(encoding='utf-8'));files=c['files']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['collection_sha256']
    assert c['verified']==expected['small_file_hashes'] and len(c['verified'])==42
    for key,row in c['verified'].items():
        data=files[key].encode('utf-8');assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],key
    source=hashlib.sha256(files['source_hashes.json'].encode('utf-8')).hexdigest()
    assert source==expected['source_manifest_sha256']=='49217e27e937b7ae373d6abd5c43da162c3109ca99a7c08784c3751fa2777cd1'
    assert c['source_files_verified']==418
    load=lambda key:json.loads(files[key]);raw={};configurations={}
    for seed in (19,1):
        for arm in ('C','B'):
            tag=f'A_{arm}_s{seed}';key=f'{arm}_{seed}'
            m=load(tag+'/manifest.json');config=m['config'];configurations[key]=config
            assert {k:m[k] for k in ('weights_sha256','input_sha256','cohort_sha256')}==expected['identity']
            assert config==expected['runs'][key]['config']
            assert len(m['fit'])==2134 and len(m['audit'])==522 and not set(m['fit'])&set(m['audit'])
            for name,row in load(tag+'/preservation_manifest.json').items():
                if tag+'/'+name in files:
                    data=files[tag+'/'+name].encode('utf-8')
                    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            for name,ids,kind in [('evaluation.json',m['audit'],'audit'),('fit.json',m['fit'],'fit')]:
                rows=load(tag+'/'+name);assert [r['image'] for r in rows]==ids
                assert aggregate(rows)==expected['runs'][key][kind]
                if kind=='audit':raw[key]=rows
            logs=[json.loads(x) for x in files[tag+'/loss.jsonl'].splitlines()]
            assert [r['epoch'] for r in logs]==list(range(1,131))
            assert all(r['microbatch']==16 and r['effective_batch']==(16 if arm=='C' else 32) and r['optimizer_steps']==(134 if arm=='C' else 67) for r in logs)
            assert sum(r['training_seconds'] for r in logs)==expected['runs'][key]['training_seconds']
            assert sum(r['checkpoint_seconds'] for r in logs)==expected['runs'][key]['checkpoint_seconds']
            assert load(tag+'/complete.json')['epochs']==130 and load(tag+'/complete.json')['exit_code']==0
    pairs={}
    for seed in ('19','1'):
        value=bootstrap(raw[f'C_{seed}'],raw[f'B_{seed}'])
        assert all(expected['pairs'][seed][k]==v for k,v in value.items()),seed
        pairs[seed]=value
    assert expected['admission_pass']==all(p['pass_seed'] for p in pairs.values()) is False
    print(json.dumps(dict(status='PASS',numpy=np.__version__,small_files=42,source_files_previously_verified=418,
        model_inference=False,original_gate_imported=False,independent_pairs=pairs),indent=2,allow_nan=False))


if __name__=='__main__':main()
