"""Fail-closed source/identity/control/depth-witness/reload/visual/time readiness for P31."""
import hashlib,json,math,pathlib,sys
root,source=map(pathlib.Path,sys.argv[1:]);sys.path.insert(0,str(source/'scripts'))
from p5_contracts import verify_hashes,validate_cohort,COHORT_SHA,TISSUE_SHA
from train_proposal_p26 import run_budget_hours,WEIGHT_SHA,DATA_SHA
from train_proposal_p31 import run_config,DEPTH
control=json.loads((root/'control.json').read_text())
assert control['status']=='PASS' and control['fold1_only'] and set(control['seeds'])=={'19','1'}
for seed in control['seeds'].values():
 assert all(seed[k] for k in ('shared_initial_state_equal','initial_rng_equal','initial_prediction_equal','control_two_step_parity'))
 counts=seed['trainable_parameters'];assert set(counts)=={'P3','C','G'} and counts['P3']==counts['C'] and counts['G']==counts['C']+2816
 boundary=seed['normalization'];assert boundary['normalization_calls']==64 and boundary['max_formula_error']<.04
 assert boundary['frozen_tensors_unchanged'] and boundary['exact_optimizer_membership']
 for key in ('gradients','beyond_decay'):
  assert set(boundary[key])=={*(f'norm{i}' for i in range(8)),'block12','block23'} and all(math.isfinite(v) and v>0 for v in boundary[key].values())
reports={a:json.loads((root/a/'smoke.json').read_text()) for a in ('C','G')}
manifests={a:json.loads((root/a/'manifest.json').read_text()) for a in ('C','G')}
identity={k:manifests['C'][k] for k in ('weights_sha256','input_sha256','cohort_sha256')}
assert all({k:m[k] for k in identity}==identity for m in manifests.values())
assert identity['weights_sha256']==WEIGHT_SHA and identity['cohort_sha256']==COHORT_SHA
assert identity['input_sha256']=={**{k:v for k,v in DATA_SHA.items() if k.startswith('fold1/')},'fold1/tissue.npy':TISSUE_SHA}
for a,m in manifests.items():
 assert m['config']==run_config(a,19,4) and m['arguments']['microbatch']==4
 assert validate_cohort(m['fit'],m['audit'])==COHORT_SHA
 assert control['smoke_fit']==m['smoke_fit'] and len(set(m['smoke_fit']))==134
 assert set(m['smoke_fit'])<=set(m['fit']) and not set(m['smoke_fit'])&set(m['audit'])
 assert m['trained_blocks']==list(DEPTH[a])
 assert m['arguments']['smoke'] is True and m['arguments']['seed']==19
 fixed=m['fixed_examples'];ids={k:fixed[k] for k in ('fit','audit')}
 assert fixed['sha256']==hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert len(ids['fit'])==4 and len(ids['audit'])==8
 assert len(set(ids['fit']))==4 and len(set(ids['audit']))==8
 assert set(ids['fit'])<=set(m['fit']) and set(ids['audit'])<=set(m['audit'])
 assert not (root/a/'evaluation.json').exists()
assert manifests['C']['fit']==manifests['G']['fit'] and manifests['C']['audit']==manifests['G']['audit']
assert manifests['G']['trainable_encoder_parameters']==manifests['C']['trainable_encoder_parameters']>0
assert manifests['G']['head_parameters']==manifests['C']['head_parameters']+2816
assert manifests['G']['fixed_examples']==manifests['C']['fixed_examples']
for a,r in reports.items():
 assert r['status']=='PASS' and r['reload_equal'] and r['final_reload_equal']
 assert r['all_frozen_encoder_tensors_unchanged'] and r['exact_optimizer_membership'] and r['cross_arm_config_rejected']
 assert r['block12_changed'] is True and r['block0_changed'] is False and r['trained_blocks']==list(DEPTH[a]) and r['budget_hours']==run_budget_hours('C')
 assert r['observed_input_size']==256 and r['observed_token_grids']==[16]*4
 assert r['smoke_fit']==control['smoke_fit'] and r['training_images']==134
 row=json.loads((root/a/'status.json').read_text());assert row['optimizer_steps']==9 and row['effective_batch']==16 and row['microbatch']==4
 assert math.isclose(row['train_loss'],row['focal_loss']+row['regularizer_loss'],rel_tol=1e-12,abs_tol=1e-12)
 assert row['regularizer_loss']==0 and row['block_drift']=={}
 assert r['effective_batch_lengths']==[16]*8+[6]
 counts=r['effective_object_counts'];assert len(counts)==9 and all(type(v) is int and v>=0 for v in counts)
 names=r['trainable_encoder_names'];assert len(names)==len(set(names)) and all(n.startswith('blocks.') for n in names)
 assert sorted({int(n.split('.')[1]) for n in names})==list(DEPTH[a])
 estimate=130*r['training_seconds']/134*2134+130*r['checkpoint_seconds']+r['final_checkpoint_seconds']+1800
 assert math.isclose(r['safety_seconds'],1.5*estimate,rel_tol=1e-12) and 1.5*estimate<43200
 assert 0<r['training_peak_memory']<=24*1024**3
 assert math.isfinite(r['reload_seconds']) and r['reload_seconds']>0
 expected={'encoder','output',f'block{DEPTH[a][0]}'}
 lrs=[3e-4,1e-5]
 assert len(r['optimizer_initial_lrs'])==len(lrs) and all(math.isclose(x,y,rel_tol=1e-14) for x,y in zip(r['optimizer_initial_lrs'],lrs))
 assert set(r['changes'])==set(r['gradients'])==expected
 assert all(math.isfinite(v) and v>0 for k in ('changes','gradients') for v in r[k].values())
 assert r['precision_trace']==dict(enabled=True,amp_dtype='torch.bfloat16',layers=dict(encoder='torch.bfloat16',output='torch.bfloat16'),logits_dtype='torch.float32')
 visual=json.loads((root/a/'point_examples_smoke/coordinates.json').read_text())
 assert visual['scope']=='fit_1epoch_preliminary' and visual['main_map_only'] and len(visual['records'])==2
 assert all(row['image'] in control['smoke_fit'] and (root/a/'point_examples_smoke'/row['png']).stat().st_size>1000 for row in visual['records'])
 assert any(any(g[0]==4 and g[2] for g in row['gt']) for row in visual['records'])
provenance=json.loads((source.parent/'provenance.json').read_text());verify_hashes(source.parent,provenance['script_hashes'])
digest=hashlib.sha256((source.parent/'source_hashes.json').read_bytes()).hexdigest();assert digest==provenance['source_manifest_sha256']
for rel,v in json.loads((source.parent/'source_hashes.json').read_text()).items():assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==v,rel
with (root/'ready.json').open('x') as f:json.dump(dict(status='PASS',source_manifest_sha256=digest,script_hashes=provenance['script_hashes'],identity=identity,reports=reports,control=control,selected_microbatch=4,budgets={a:run_budget_hours('C') for a in ('C','G')},scheduler_seconds_per_formal_task=43200),f,indent=2)
print(json.dumps(dict(status='PASS',source_manifest_sha256=digest,safety_hours={a:r['safety_seconds']/3600 for a,r in reports.items()},peak_GiB={a:r['training_peak_memory']/1024**3 for a,r in reports.items()})))
