"""Reproduce source-bound P20 results and original admission from collected small payloads."""
import pathlib,json,hashlib,numpy as np
from analyze_proposal_p18 import aggregate,bootstrap_and_gate
from proposal_p3_utils import admission
from train_proposal_p20 import run_config
from p5_contracts import COHORT_SHA,validate_cohort
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--collection',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
p=args.collection;c=json.loads(p.read_text());files=c['files'];runs={};raw={};identities=[]
for k,e in c['verified'].items():
 d=files[k].encode();assert len(d)==e['bytes'] and hashlib.sha256(d).hexdigest()==e['sha256'],k
load=lambda k:json.loads(files[k])
assert c['source_files_verified']==426
assert hashlib.sha256(files['source_hashes.json'].encode()).hexdigest()=='1bd6e680e1b4c008f11b97ee50c165c21c4776209f718a915aca101333228e58'
for seed in (19,1):
 for arm in ('C','U'):
  tag=f'A_{arm}_s{seed}';m=load(tag+'/manifest.json');pres=load(tag+'/preservation_manifest.json')
  for n,e in pres.items():
   if tag+'/'+n not in files:continue
   d=files[tag+'/'+n].encode();assert len(d)==e['bytes'] and hashlib.sha256(d).hexdigest()==e['sha256'],n
  assert m['config']==run_config(arm,seed)
  logs=[json.loads(s) for s in files[tag+'/loss.jsonl'].splitlines()]
  assert [r['epoch'] for r in logs]==list(range(1,131))
  assert all(r['optimizer_steps']==134 and r['effective_batch']==16 and r['microbatch']==4 and np.isfinite(r['train_loss']) for r in logs)
  done=load(tag+'/complete.json');assert done['epochs']==130 and done['exit_code']==0
  rows=load(tag+'/evaluation.json');fit=load(tag+'/fit.json');assert len(rows)==522 and len(fit)==2134
  assert [r['image'] for r in rows]==m['audit'] and [r['image'] for r in fit]==m['fit']
  assert validate_cohort(m['fit'],m['audit'])==COHORT_SHA
  identity={k:m[k] for k in ('weights_sha256','input_sha256','cohort_sha256')};identities.append(identity)
  a,b=aggregate(rows),aggregate(fit);s=load(tag+'/summary.json')
  assert all(a['dead_total' if k=='dead_objects' else k]==v for k,v in s.items())
  training=sum(r['training_seconds'] for r in logs);saving=sum(r['checkpoint_seconds'] for r in logs)
  runs[f'{arm}_{seed}']=dict(audit=a,fit=b,config=m['config'],optimizer_steps=17420,expected_tail=6,training_seconds=training,checkpoint_seconds=saving,elapsed_seconds=done['elapsed_seconds'],other_seconds=done['elapsed_seconds']-training-saving,peak_allocated_bytes=max(r['peak_memory'] for r in logs),first_loss=logs[0]['train_loss'],last_loss=logs[-1]['train_loss'],tissues={t:aggregate([r for r in rows if r['tissue']==t]) for t in sorted({r['tissue'] for r in rows})})
  raw[f'{arm}_{seed}']=rows
assert all(x==identities[0] for x in identities)
pairs={str(s):bootstrap_and_gate(raw[f'C_{s}'],raw[f'U_{s}']) for s in (19,1)}
original=admission({str(s):dict(F1=raw[f'C_{s}'],T1=raw[f'U_{s}']) for s in (19,1)})
for seed,pair in pairs.items():
 for k in ('delta','ci95','pass_seed'):assert pair[k]==original['details'][seed][k],(seed,k)
 for key in ('dead_images','dead_tissues'):
  pair[key]=[{('U_hits' if k=='H_hits' else k):v for k,v in r.items()} for r in pair[key]]
result=dict(experiment='P20',collection_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(files['source_hashes.json'].encode()).hexdigest(),source_files_verified=426,small_files_verified=len(c['verified']),identity=identities[0],runs=runs,pairs=pairs,original_gate=original,independent_gate_equal=True,admission_pass=original['pass'],mean_dead_delta=float(np.mean([p['delta'] for p in pairs.values()])),accounting=c['sacct'],stderr=c['stderr'],queue=c['queue'],bootstrap=dict(replicates=1000,seed=20261018,unit='Dead-positive tissue-stratified images',CI=[2.5,97.5]),checkpoints_independently_transferred_or_rehashed=False,patient_independence=False,test_performance=False)
with args.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(status='PASS',admission=result['admission_pass'],dead_delta={k:p['delta'] for k,p in pairs.items()})))