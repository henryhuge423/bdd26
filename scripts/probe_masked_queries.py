"""E10: bounded predicted-mask attention with hash-verified historical controls."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

HELPER_SHA='8432b771d2fe1b69b1633d748659a668ddad7095146f597ddd5a0e06d3df51f1'
CONTROL_SHA='cc2ff1eaeb5a5e407c2c6041a0137fe648e720114194d1d8efd0ddb0de27cd33'
CONTROL_DESIGN_SHA='83a5b3972438d1c80daa346daeac5a341130af8f698abc4663b21f05e4d52044'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def attention_mask(logits,heads):
    if logits.ndim!=3 or min(logits.shape)<1 or heads<1 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite nonempty BQK mask logits and positive head count required')
    blocked=(logits.detach()<0)
    fallback=blocked.all(-1)
    blocked[fallback]=False
    b,q,k=blocked.shape
    stats={'fallback_rows':int(fallback.sum()),'query_rows':b*q,'memory_positions':k,
           'blocked_fraction':float(blocked.float().mean())}
    return blocked[:,None].expand(b,heads,q,k).reshape(b*heads,q,k),stats


def query_forward(head,features,masked,diagnostics=None):
    if not masked:return head(features)
    from nucseg.setmask.model import spatial_position
    memory=head.memory_projection(F.avg_pool2d(features,8))
    mpos=spatial_position(*memory.shape[-2:],head.dim,memory.device,memory.dtype)
    mask_features=head.mask_projection(features)
    # Interpolation and dot product commute mathematically; freeze this exact float32 route.
    with torch.no_grad():
        low=F.interpolate(mask_features.float(),size=memory.shape[-2:],mode='bilinear',align_corners=False)
    memory=memory.flatten(2).transpose(1,2)
    q=head.query_content.weight[None].expand(features.shape[0],-1,-1)
    qpos=head.query_position.weight[None]
    for layer in head.layers:
        with torch.no_grad(),torch.autocast(features.device.type,enabled=False):
            embed=head.mask_head(q.float())
            logits=torch.einsum('bqc,bck->bqk',embed,low.flatten(2))/math.sqrt(embed.shape[-1])
            blocked,stats=attention_mask(logits,layer.cross.num_heads)
        if diagnostics is not None:diagnostics.append(stats)
        t=layer.norm1(q)
        q=q+layer.drop(layer.cross(t+qpos,memory+mpos,memory,attn_mask=blocked,need_weights=False)[0])
        t=layer.norm2(q)
        q=q+layer.drop(layer.self_attention(t+qpos,t+qpos,t,need_weights=False)[0])
        q=q+layer.drop(layer.ff(layer.norm3(q)))
    return {'class_logits':head.class_head(q),'mask_embed':head.mask_head(q),'mask_features':mask_features}


def validate_measurement(r,images):
    c=r['counts'];rows=r['per_image']
    if len(rows)!=len(images) or {x['image'] for x in rows}!=set(images):
        raise ValueError('Historical cohort identities differ')
    if any(c[k]!=sum(x[k] for x in rows) for k in c):
        raise ValueError('Historical per-image counts differ')
    if c['n']<=0 or c['dead_n']<=0:raise ValueError('Missing allocation support')
    want={'recall':c['tp']/c['n'],'precision':c['tp']/max(c['pred'],1),
          'dead_recall':c['dead_good']/c['dead_n'],
          'dead_typed_f1':2*c['dead_typed_tp']/max(c['dead_n']+c['dead_pred'],1),
          'raw_dead_recall':c['raw_dead_good']/c['dead_n'],
          'interior_dead_recall':c['interior_dead_good']/max(c['interior_dead_n'],1)}
    if any(not math.isclose(r[k],v,abs_tol=1e-12,rel_tol=0) for k,v in want.items()):
        raise ValueError('Historical count-derived rates differ')


def allocation(rows):
    details={}
    quality=('dead_recall','recall','precision','dead_typed_f1')
    for seed in ('19','1'):
        g,m,a0=(rows[seed][a] for a in ('G','M','A0'))
        changes={k:m[k]-g[k] for k in quality}
        original=m['fit_recall']>=.80 and changes['dead_recall']>=.05 and all(changes[k]>=0 for k in ('recall','precision','dead_typed_f1'))
        a0_changes={k:m[k]-a0[k] for k in quality}
        guard=all(v>=0 for v in a0_changes.values())
        details[seed]={'original_gate':bool(original),'a0_guard':bool(guard),
                       'fit_recall':m['fit_recall'],'m_minus_g':changes,'m_minus_a0':a0_changes}
    original=all(d['original_gate'] for d in details.values());guard=all(d['a0_guard'] for d in details.values())
    return {'original_gate':original,'a0_guard':guard,'promising':original and guard,'per_seed':details,
            'meaning':'Train-only allocation; separate development design only, no automatic large training'}


def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)
    if len(text.encode())>8*1024**2:raise ValueError('Output budget exceeded')
    with Path(path).open('x') as f:f.write(text)


@torch.no_grad()
def diagnose(head,row):
    stats=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        query_forward(head,row['feature'].to('cuda'),True,stats)
    return {'image':row['image'],'layers':stats}


def run(a):
    snapshot=a.snapshot.resolve();os.chdir(snapshot);sys.path.insert(0,str(snapshot/'src'))
    from torch.utils.data import default_collate
    from nucseg.setmask.engine import load_config,build_model
    from nucseg.setmask.data import PairedDataset
    from nucseg.setmask.matching import setmask_loss
    assert sha(a.helper)==HELPER_SHA and sha(a.control)==CONTROL_SHA
    assert sha(a.control_design)==CONTROL_DESIGN_SHA
    spec=importlib.util.spec_from_file_location('e9_frozen_measurement',a.helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # Explicit prediction dependency injection; frozen measurement/loss bytes stay unchanged.
    helper.query_forward=query_forward
    reference=json.loads(a.control.read_text())
    assert reference['manifest']['script_sha256']==HELPER_SHA
    assert reference['manifest']['design_sha256']==CONTROL_DESIGN_SHA
    assert reference['manifest']['steps']==300 and reference['manifest']['batch_size']==2
    assert torch.cuda.device_count()==1 and shutil.disk_usage(a.out.parent).free>20*1024**3
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    a.out.mkdir(exist_ok=False);start=time.monotonic()
    source=json.loads((snapshot/'MANIFEST.json').read_text())
    assert source['commit']==reference['manifest']['source_commit']
    for group in ('sources','vendor'):
        for name,digest in source[group].items():assert sha(snapshot/name)==digest,name
    ds=PairedDataset(1,False,19);present=ds.class_presence()[:,3]
    groups=helper.cohorts(np.flatnonzero(present).tolist(),np.flatnonzero(~present).tolist())
    assert groups==reference['manifest']['groups'] and len(groups['fit'])==64 and len(groups['audit'])==66
    for arms in reference['results'].values():
        for arm in ('A0','G'):
            for group,images in groups.items():validate_measurement(arms[arm][group],images)
            if arm=='G':assert [x['step'] for x in arms[arm]['optimization']]==[1,100,300]
    manifest={'fold':1,'source_commit':source['commit'],'groups':groups,'steps':300,'batch_size':2,
              'script_sha256':sha(__file__),'design_sha256':sha(a.design),'helper_sha256':HELPER_SHA,
              'control_sha256':CONTROL_SHA,'control_design_sha256':CONTROL_DESIGN_SHA,'checkpoints':{},
              'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'control_reuse':'E9 A0/G, conditional on exact A0 reproduction',
              'limitations':'Training images seen in E7; not validation/test; only query head optimized; no weights saved'}
    save(a.out/'manifest.json',manifest)
    results={};gate_rows={}
    for seed in (19,1):
        run_dir=a.run_root/f'Q_seed{seed}';cfg=load_config(run_dir)
        assert cfg.round=='E7' and cfg.arm=='Q' and cfg.seed==seed
        receipt=json.loads((run_dir/'completion.json').read_text())
        for name in ('config.json','provenance.json','log.jsonl','final.pth'):
            assert sha(run_dir/name)==receipt['hashes'][name],name
        assert receipt['hashes']['final.pth']==reference['manifest']['checkpoints'][str(seed)]
        state=torch.load(run_dir/'final.pth',map_location='cpu',weights_only=False,mmap=True)
        assert state['config']==asdict(cfg)
        prov=state['provenance'];assert torch.__version__==prov['torch'] and np.__version__==prov['numpy']
        for name in ('images','inst','type','tissue'):
            assert sha(ds.fold.dir/(name+'.npy'))==prov['inputs']['fold1/'+name+'.npy']
        assert sha(cfg.uni_ckpt)==prov['inputs']['UNI']
        model=build_model(cfg,pretrained=False).eval().to('cuda');model.load_state_dict(state['model']);del state
        caches={k:[] for k in groups};captured=[]
        hook=model.query_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        with torch.no_grad():
            for group,images in groups.items():
                for image in images:
                    batch=default_collate([ds[image]])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(batch['img'].to('cuda'));feature=captured.pop()
                        original=query_forward(model.query_head,feature,False);captured.clear()
                    assert all(torch.equal(pred[k],original[k]) for k in original)
                    caches[group].append({'image':image,'feature':feature.cpu(),'batch':batch,'tissue_logits':pred['tissue'].cpu()})
                    del pred,feature,original
                    helper.deadline(start)
        hook.remove();original_head=deepcopy(model.query_head).eval();del model;torch.cuda.empty_cache()
        a0={k:helper.measure(original_head,v,False,start) for k,v in caches.items()}
        assert a0==reference['results'][str(seed)]['A0'],'A0 reproduction differs; do not reuse controls'
        print(json.dumps({'seed':seed,'a0_exact_reproduction':True}),flush=True)
        before=helper.feature_hash(caches['fit']+caches['audit'])
        torch.manual_seed(seed);head=deepcopy(original_head).eval()
        assert all(torch.equal(v,original_head.state_dict()[k]) for k,v in head.state_dict().items())
        diagnostics={'before':{k:diagnose(head,v[0]) for k,v in caches.items()}}
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,betas=(.85,.95),weight_decay=1e-4)
        logs=[]
        for step in range(300):
            rows=[caches['fit'][(step*2+j)%64] for j in range(2)]
            feature=torch.cat([r['feature'] for r in rows]).to('cuda')
            batch={k:torch.cat([r['batch'][k] for r in rows]) for k in ('inst_map','tp_map','tissue')}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=query_forward(head,feature,True)
            pred['tissue']=torch.cat([r['tissue_logits'] for r in rows]).to('cuda')
            loss,terms=setmask_loss(pred,batch);assert bool(torch.isfinite(loss))
            loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
            optimizer.step()
            if step in (0,99,299):logs.append({'step':step+1,'loss':float(loss.detach()),'terms':terms})
            del pred,feature,loss
            helper.deadline(start)
        treatment={k:helper.measure(head,v,True,start) for k,v in caches.items()}
        diagnostics['after']={k:diagnose(head,v[0]) for k,v in caches.items()}
        treatment.update(optimization=logs,attention_diagnostics=diagnostics)
        assert helper.feature_hash(caches['fit']+caches['audit'])==before
        for group,images in groups.items():validate_measurement(treatment[group],images)
        arms={'A0':a0,'G':reference['results'][str(seed)]['G'],'M':treatment}
        results[str(seed)]=arms
        gate_rows[str(seed)]={arm:{**r['audit'],'fit_recall':r['fit']['recall']} for arm,r in arms.items()}
        manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth']
        save(a.out/f'seed{seed}.json',arms)
        print(json.dumps({'seed':seed,'M':{g:{k:v for k,v in treatment[g].items() if k!='per_image'} for g in ('fit','audit')}}),flush=True)
        del optimizer,head,original_head,caches;torch.cuda.empty_cache()
    payload={'manifest':manifest,'results':results,'allocation':allocation(gate_rows),
             'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    save(a.out/'results.json',payload);print(json.dumps(payload['allocation']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','helper','control','control-design','design','run-root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ('snapshot','helper','control','control_design','design','run_root','out'):
        setattr(args,name,getattr(args,name).resolve())
    run(args)
