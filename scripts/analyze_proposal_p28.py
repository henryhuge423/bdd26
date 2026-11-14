"""Verify collected P28 payloads, epoch/config identities and original paired decision.

python scripts/analyze_proposal_p28.py --root <collected directory> --out <new JSON>
The input directory contains p28_files/collection.json and its verified payloads.
"""
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=__import__('pathlib').Path,required=True)
parser.add_argument('--out',type=__import__('pathlib').Path,required=True)
import pathlib,json,hashlib,sys,numpy as np
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
from analyze_proposal_p18 import aggregate,bootstrap_and_gate
from proposal_p3_utils import admission
from train_proposal_p28 import run_config
from p5_contracts import COHORT_SHA,validate_cohort
args=parser.parse_args();o=args.root;c=json.loads((o/'p28_files/collection.json').read_text());files=c['files'];runs={};raw={};identities=[]
for k,e in c['verified'].items():
 d=(o/'p28_files'/k).read_bytes();assert len(d)==e['bytes'] and hashlib.sha256(d).hexdigest()==e['sha256'],k
load=lambda k:json.loads(files[k])
for seed in (19,1):
 for arm in ('C','I'):
  tag=f'A_{arm}_s{seed}';m=load(tag+'/manifest.json');logs=[json.loads(s) for s in files[tag+'/loss.jsonl'].splitlines()]
  assert m['config']==run_config(arm,seed)
  assert [r['epoch'] for r in logs]==list(range(1,131))
  assert all(r['optimizer_steps']==134 and r['effective_batch']==16 and r['microbatch']==4 and np.isfinite(r['train_loss']) for r in logs)
  done=load(tag+'/complete.json');assert done['epochs']==130 and done['exit_code']==0
  rows=load(tag+'/evaluation.json');fit=load(tag+'/fit.json');assert len(rows)==522 and len(fit)==2134
  assert [r['image'] for r in rows]==m['audit'] and [r['image'] for r in fit]==m['fit']
  assert validate_cohort(m['fit'],m['audit'])==COHORT_SHA
  identities.append({k:m[k] for k in ('weights_sha256','input_sha256','cohort_sha256')})
  a,b=aggregate(rows),aggregate(fit);s=load(tag+'/summary.json');assert all(a['dead_total' if k=='dead_objects' else k]==v for k,v in s.items())
  examples={scope:load(tag+f'/point_examples_{scope}/coordinates.json') for scope in ('fit','audit')}
  assert all([r['image'] for r in examples[scope]['records']]==m['fixed_examples'][scope] for scope in ('fit','audit'))
  runs[f'{arm}_{seed}']=dict(audit=a,fit=b,config=m['config'],optimizer_steps=17420,training_seconds=sum(r['training_seconds'] for r in logs),checkpoint_seconds=sum(r['checkpoint_seconds'] for r in logs),elapsed_seconds=done['elapsed_seconds'],peak_allocated_bytes=max(r['peak_memory'] for r in logs),loss_curve=[r['train_loss'] for r in logs],focal_curve=[r['focal_loss'] for r in logs],regularizer_curve=[r['regularizer_loss'] for r in logs],final_block_drift=logs[-1]['block_drift'],head_parameters=m['head_parameters'],trainable_encoder_parameters=m['trainable_encoder_parameters'],tissues={t:aggregate([r for r in rows if r['tissue']==t]) for t in sorted({r['tissue'] for r in rows})},fixed_examples=m['fixed_examples'],max_candidates=max(r['candidates'] for r in rows),saturated=sum(r['saturated_512'] for r in rows))
  raw[f'{arm}_{seed}']=rows
assert all(i==identities[0] for i in identities)
pairs={str(s):bootstrap_and_gate(raw[f'C_{s}'],raw[f'I_{s}']) for s in (19,1)}
gate=admission({str(s):dict(F1=raw[f'C_{s}'],T1=raw[f'I_{s}']) for s in (19,1)})
for seed,pair in pairs.items():
 for k in ('delta','ci95','pass_seed'):assert pair[k]==gate['details'][seed][k]
result=dict(experiment='P28',runs=runs,pairs=pairs,original_gate=gate,independent_gate_equal=True,identity=identities[0],source_files_verified=c['source_files_verified'],small_files_verified=len(c['verified']),png=48,checkpoint_sizes=c['checkpoint_sizes'],sacct=c['sacct'],repeated_fold1_development=True,test_performance=False,checkpoint_independently_rehashed=False)
with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(gate=gate,runs={k:{kk:vv for kk,vv in v.items() if kk not in ('loss_curve','focal_curve','regularizer_curve','tissues','fixed_examples','config')} for k,v in runs.items()}),indent=2))
