"""Independently reconstruct P19 endpoints and original paired bootstrap from small payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_proposal_p18 import aggregate,bootstrap_and_gate

def analyze(path):
    collection=json.loads(Path(path).read_text());files=collection['files']
    for key,entry in collection['verified'].items():
        data=files[key].encode();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],key
    load=lambda key:json.loads(files[key])
    source_sha=hashlib.sha256(files['source_hashes.json'].encode()).hexdigest()
    assert source_sha=='49217e27e937b7ae373d6abd5c43da162c3109ca99a7c08784c3751fa2777cd1'
    assert collection['source_files_verified']==418
    runs={};raw={};identities=[]
    for seed in (19,1):
        for arm in ('C','B'):
            tag=f'A_{arm}_s{seed}';m=load(tag+'/manifest.json');pres=load(tag+'/preservation_manifest.json')
            for name,expected in pres.items():
                if tag+'/'+name not in files:continue
                data=files[tag+'/'+name].encode();assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],name
            config=m['config'];batch=16 if arm=='C' else 32
            assert config==dict(phase='A',arm=arm,seed=seed,target_sigma=1,target_radius=3,augmentation=dict(kind='original'),trainable_encoder_blocks=list(range(12,24)),architecture='original_UNI_last12',effective_batch=batch,microbatch=16)
            logs=[json.loads(line) for line in files[tag+'/loss.jsonl'].splitlines()]
            assert [r['epoch'] for r in logs]==list(range(1,131))
            assert all(r['optimizer_steps']==(134 if arm=='C' else 67) and r['effective_batch']==batch and r['microbatch']==16 for r in logs)
            assert all(np.isfinite(r['train_loss']) for r in logs)
            complete=load(tag+'/complete.json');assert complete['epochs']==130 and complete['exit_code']==0
            rows,fit=load(tag+'/evaluation.json'),load(tag+'/fit.json')
            assert len(rows)==522 and len(fit)==2134
            assert [r['image'] for r in rows]==m['audit'] and [r['image'] for r in fit]==m['fit']
            assert not set(m['fit'])&set(m['audit'])
            identity={k:m[k] for k in ('weights_sha256','input_sha256','cohort_sha256')};identities.append(identity)
            a,b=aggregate(rows),aggregate(fit);summary=load(tag+'/summary.json')
            assert all(a['dead_total' if k=='dead_objects' else k]==v for k,v in summary.items())
            training=sum(r['training_seconds'] for r in logs);saving=sum(r['checkpoint_seconds'] for r in logs)
            runs[f'{arm}_{seed}']=dict(audit=a,fit=b,config=config,optimizer_steps=130*(134 if arm=='C' else 67),expected_tail=2134%batch,
                training_seconds=training,checkpoint_seconds=saving,elapsed_seconds=complete['elapsed_seconds'],mean_training_epoch_seconds=training/130,
                other_seconds=complete['elapsed_seconds']-training-saving,peak_allocated_bytes=max(r['peak_memory'] for r in logs),
                tissues={t:aggregate([r for r in rows if r['tissue']==t]) for t in sorted({r['tissue'] for r in rows})})
            raw[f'{arm}_{seed}']=rows
    assert all(v==identities[0] for v in identities)
    from p5_contracts import validate_cohort,COHORT_SHA
    assert identities[0]['cohort_sha256']==COHORT_SHA
    assert validate_cohort(m['fit'],m['audit'])==COHORT_SHA
    pairs={str(seed):bootstrap_and_gate(raw[f'C_{seed}'],raw[f'B_{seed}']) for seed in (19,1)}
    from proposal_p3_utils import admission
    original=admission({str(s):dict(F1=raw[f'C_{s}'],T1=raw[f'B_{s}']) for s in (19,1)})
    assert original.get('reason')!='missing_seed'
    for seed,pair in pairs.items():
        for key in ('delta','ci95','pass_seed'):assert pair[key]==original['details'][seed][key],(seed,key)
        pair['dead_images']=[{('B_hits' if k=='H_hits' else k):v for k,v in r.items()} for r in pair['dead_images']]
        pair['dead_tissues']=[{('B_hits' if k=='H_hits' else k):v for k,v in r.items()} for r in pair['dead_tissues']]
    return dict(experiment='P19',collection_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),source_manifest_sha256=source_sha,source_files_verified=418,
        small_files_verified=len(collection['verified']),small_file_hashes=collection['verified'],identity=identities[0],runs=runs,pairs=pairs,
        original_gate=original,independent_gate_equal=True,admission_pass=all(p['pass_seed'] for p in pairs.values()),
        mean_dead_delta=float(np.mean([p['delta'] for p in pairs.values()])),accounting=collection['sacct'],queue=collection['queue'],
        bootstrap=dict(replicates=1000,seed=20261018,unit='Dead-positive tissue-stratified images',CI=[2.5,97.5]),
        checkpoints_independently_transferred_or_rehashed=False,patient_independence=False,test_performance=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.collection)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='PASS',admission=result['admission_pass'],runs={k:{x:r[x] for x in ('audit','fit','training_seconds','checkpoint_seconds','elapsed_seconds','peak_allocated_bytes')} for k,r in result['runs'].items()},pairs=result['pairs']),ensure_ascii=False,indent=2))
