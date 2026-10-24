"""E9 bounded train-only local-content experiment; no trained weights are saved."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def initial_queries(head,features,local):
    q=head.query_content.weight[None].expand(features.shape[0],-1,-1)
    if local:
        if q.shape[1]!=512:raise ValueError('E9 requires512 queries')
        content=head.memory_projection(F.adaptive_avg_pool2d(features,(16,32)))
        q=q+content.flatten(2).transpose(1,2)
    return q


def query_forward(head,features,local):
    if not local:return head(features)
    from nucseg.setmask.model import spatial_position
    memory=head.memory_projection(F.avg_pool2d(features,8))
    pos=spatial_position(*memory.shape[-2:],head.dim,memory.device,memory.dtype)
    memory=memory.flatten(2).transpose(1,2)
    q=initial_queries(head,features,True)
    for layer in head.layers:q=layer(q,head.query_position.weight[None],memory,pos)
    return {'class_logits':head.class_head(q),'mask_embed':head.mask_head(q),
            'mask_features':head.mask_projection(features)}


def cohorts(positive,negative):
    positive=list(map(int,positive));negative=list(map(int,negative))
    if not positive or len(negative)<len(positive) or set(positive)&set(negative):
        raise ValueError('Distinct positive and negative support required')
    key=lambda i:hashlib.sha256(f'e9:19:{i}'.encode()).hexdigest()
    p=sorted(positive,key=key);n=sorted(negative,key=key)[:len(p)]
    cut=len(p)//2
    return {'fit':sorted(p[:cut]+n[:cut],key=key),'audit':sorted(p[cut:]+n[cut:],key=key)}


def allocation(rows):
    details={}
    for seed in ('19','1'):
        g,l=rows[seed]['G'],rows[seed]['L']
        d={'dead_gain':l['dead_recall']-g['dead_recall'],
           'recall_change':l['recall']-g['recall'],
           'precision_change':l['precision']-g['precision'],
           'dead_typed_f1_change':l['dead_typed_f1']-g['dead_typed_f1'],
           'fit_recall':l['fit_recall']}
        d['pass']=d['dead_gain']>=.05 and d['recall_change']>=0 and d['precision_change']>=0 and d['dead_typed_f1_change']>=0 and d['fit_recall']>=.80
        details[seed]=d
    return {'promising':all(d['pass'] for d in details.values()),'per_seed':details,
            'meaning':'Train-only allocation; a pass permits a separate train/val design only'}


def feature_hash(cache):
    h=hashlib.sha256()
    for row in cache:
        h.update(row['feature'].contiguous().view(torch.uint8).numpy().tobytes())
        h.update(row['tissue_logits'].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def deadline(start):
    if time.monotonic()-start>1750:raise RuntimeError('E9 wall budget exhausted; no retry')


def interior_dead_mask(inst,ids,classes):
    from nucseg.visibility_probe import original_full
    interior=original_full(inst)
    return (classes==4) & np.asarray([interior[int(i)] for i in ids],dtype=bool)


@torch.no_grad()
def measure(head,cache,local,start):
    from nucseg.setmask.decode import decode_set
    from nucseg.setmask.matching import make_targets
    from nucseg.metrics.instance import overlap,pq_from_overlap
    from nucseg.metrics.pannuke_eval import instance_classes
    from nucseg.visibility_probe import original_full
    names=('n','pred','tp','dead_n','dead_good','dead_typed_tp','dead_pred','raw_dead_good','interior_dead_n','interior_dead_good')
    counts={k:0 for k in names};rows=[]
    for row in cache:
        feature=row['feature'].to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=query_forward(head,feature,local)
        inst,typ=decode_set(pred);pi,pt=inst[0],typ[0]
        gt=row['batch']['inst_map'][0].numpy();gt_type=row['batch']['tp_map'][0].numpy()
        target=make_targets(row['batch']['inst_map'],row['batch']['tp_map'])[0]
        gclasses=target['classes'].numpy();ov=overlap(gt,pi);s=pq_from_overlap(ov)
        _,pclasses=instance_classes(pi,pt)
        assert len(gclasses)==ov.nt and len(pclasses)==ov.np_
        dead=gclasses==4;good=np.zeros(ov.nt,dtype=bool);typed=np.zeros(ov.nt,dtype=bool)
        if len(s.paired):
            good[s.paired[:,0]]=True
            typed[s.paired[:,0]]=pclasses[s.paired[:,1]]==gclasses[s.paired[:,0]]
        idead=interior_dead_mask(gt,target['ids'],gclasses)
        # Exact native raw-mask coverage on Dead, without any GT-based selection at inference.
        raw_good=torch.zeros(int(dead.sum()),dtype=torch.bool,device='cuda')
        if dead.any():
            ids=target['ids'][torch.from_numpy(dead)]
            truth=(target['inst'][None]==ids[:,None,None]).to('cuda').flatten(1).float()
            area=truth.sum(1)
            for q in range(0,512,16):
                raw=(torch.einsum('qc,chw->qhw',pred['mask_embed'][0,q:q+16].float(),pred['mask_features'][0].float())/math.sqrt(128)>=0).flatten(1).float()
                inter=raw@truth.T
                iou=inter/(raw.sum(1)[:,None]+area[None]-inter).clamp_min(1)
                raw_good|=(iou>.5).any(0)
        c={'n':ov.nt,'pred':ov.np_,'tp':s.tp,'dead_n':int(dead.sum()),
           'dead_good':int((dead&good).sum()),'dead_typed_tp':int((dead&good&typed).sum()),
           'dead_pred':int((pclasses==4).sum()),'raw_dead_good':int(raw_good.sum()),
           'interior_dead_n':int(idead.sum()),'interior_dead_good':int((idead&good).sum())}
        for k in names:counts[k]+=c[k]
        rows.append({'image':row['image'],**c})
        del feature,pred
        deadline(start)
    assert counts['dead_n']>0 and counts['n']>0
    rates={'recall':counts['tp']/counts['n'],'precision':counts['tp']/max(counts['pred'],1),
           'dead_recall':counts['dead_good']/counts['dead_n'],
           'dead_typed_f1':2*counts['dead_typed_tp']/max(counts['dead_n']+counts['dead_pred'],1),
           'raw_dead_recall':counts['raw_dead_good']/counts['dead_n'],
           'interior_dead_recall':counts['interior_dead_good']/max(counts['interior_dead_n'],1)}
    return {**rates,'counts':counts,'per_image':rows}


def run(snapshot,run_root,out):
    sys.path.insert(0,str(snapshot/'src'));os.chdir(snapshot)
    from torch.utils.data import default_collate
    from nucseg.setmask.engine import load_config,build_model
    from nucseg.setmask.data import PairedDataset
    from nucseg.setmask.matching import setmask_loss
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    assert shutil.disk_usage(out.parent).free>=20*1024**3
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    source=json.loads((snapshot/'MANIFEST.json').read_text())
    for group in ('sources','vendor'):
        for name,digest in source[group].items():
            assert sha(snapshot/name)==digest,name
    ds=PairedDataset(1,False,19)
    present=ds.class_presence()[:,3]
    positive=np.flatnonzero(present).tolist();assert len(positive)==65
    groups=cohorts(positive,np.flatnonzero(~present).tolist())
    assert len(groups['fit'])==64 and len(groups['audit'])==66
    assert not set(groups['fit'])&set(groups['audit'])
    manifest={'fold':1,'source_commit':source['commit'],'script_sha256':sha(__file__),
              'design_sha256':sha(Path(__file__).with_name('DESIGN.md')),'groups':groups,
              'steps':300,'batch_size':2,'arms':['A0','G','L'],'checkpoints':{},
              'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'limitations':'GT-stratified training cohort; audit images seen by E7; not validation/test; no trained weights saved'}
    results={};gate_rows={}
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    for seed in (19,1):
        run_dir=run_root/f'Q_seed{seed}';cfg=load_config(run_dir)
        assert cfg.round=='E7' and cfg.arm=='Q' and cfg.seed==seed
        receipt=json.loads((run_dir/'completion.json').read_text())
        for name in ('config.json','provenance.json','log.jsonl','final.pth'):
            assert sha(run_dir/name)==receipt['hashes'][name],name
        state=torch.load(run_dir/'final.pth',map_location='cpu',weights_only=False,mmap=True)
        assert state['config']==asdict(cfg)
        provenance=state['provenance']
        assert torch.__version__==provenance['torch'] and np.__version__==provenance['numpy']
        for name in ('images','inst','type','tissue'):
            assert sha(ds.fold.dir/(name+'.npy'))==provenance['inputs']['fold1/'+name+'.npy']
        assert sha(cfg.uni_ckpt)==provenance['inputs']['UNI']
        model=build_model(cfg,pretrained=False).eval().to('cuda')
        model.load_state_dict(state['model']);del state
        cache={k:[] for k in groups};captured=[]
        hook=model.query_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        with torch.no_grad():
            for group,images in groups.items():
                for image in images:
                    batch=default_collate([ds[image]])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(batch['img'].to('cuda'))
                        feature=captured.pop()
                        original=query_forward(model.query_head,feature,False)
                        captured.clear()
                    assert torch.equal(pred['class_logits'],original['class_logits'])
                    cache[group].append({'image':image,'feature':feature.cpu(),'batch':batch,
                                         'tissue_logits':pred['tissue'].cpu()})
                    del pred,feature,original
                    deadline(start)
        hook.remove()
        original_head=deepcopy(model.query_head).eval();del model;torch.cuda.empty_cache()
        before=feature_hash(cache['fit']+cache['audit'])
        arms={'A0':{k:measure(original_head,v,False,start) for k,v in cache.items()}}
        for arm,local in (('G',False),('L',True)):
            torch.manual_seed(seed)
            head=deepcopy(original_head).eval()
            assert all(torch.equal(v,original_head.state_dict()[k]) for k,v in head.state_dict().items())
            optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,betas=(.85,.95),weight_decay=1e-4)
            logs=[]
            for step in range(300):
                rows=[cache['fit'][(step*2+j)%64] for j in range(2)]
                feature=torch.cat([r['feature'] for r in rows]).to('cuda')
                batch={k:torch.cat([r['batch'][k] for r in rows]) for k in ('inst_map','tp_map','tissue')}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=query_forward(head,feature,local)
                pred['tissue']=torch.cat([r['tissue_logits'] for r in rows]).to('cuda')
                loss,terms=setmask_loss(pred,batch)
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
                optimizer.step()
                if step in (0,99,299):logs.append({'step':step+1,'loss':float(loss.detach()),'terms':terms})
                del pred,feature,loss
                deadline(start)
            arms[arm]={k:measure(head,v,local,start) for k,v in cache.items()}
            arms[arm]['optimization']=logs
            assert feature_hash(cache['fit']+cache['audit'])==before
            print(json.dumps({'seed':seed,'arm':arm,'fit':{k:v for k,v in arms[arm]['fit'].items() if k!='per_image'},
                              'audit':{k:v for k,v in arms[arm]['audit'].items() if k!='per_image'}}),flush=True)
            del optimizer,head;torch.cuda.empty_cache()
        gate_rows[str(seed)]={arm:{**arms[arm]['audit'],'fit_recall':arms[arm]['fit']['recall']} for arm in ('G','L')}
        results[str(seed)]=arms;manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth']
        with (out/f'seed{seed}.json').open('x') as f:json.dump(arms,f,indent=2)
        del original_head,cache;torch.cuda.empty_cache()
    payload={'manifest':manifest,'results':results,'allocation':allocation(gate_rows),
             'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    with (out/'results.json').open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps(payload['allocation']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.snapshot.resolve(),args.run_root.resolve(),args.out.resolve())
