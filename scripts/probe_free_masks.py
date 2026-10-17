"""E12 GT-assisted fixed-feature fitting diagnostic; no deployable weights."""
import argparse
from dataclasses import asdict
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def mask_logits(embed,features):
    return torch.einsum('qc,chw->qhw',embed.float(),features.float())/math.sqrt(embed.shape[-1])

def mask_loss(logits,truth):
    bce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none').mean((1,2))
    p=logits.sigmoid().flatten(1);t=truth.flatten(1)
    dice=1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)
    return (5*bce+5*dice).mean()

@torch.no_grad()
def object_iou(logits,truth):
    p=logits>=0;t=truth.bool()
    return (p&t).sum((1,2))/(p|t).sum((1,2)).clamp_min(1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')

def summary(rows):
    objects=[o for r in rows for o in r['objects']]
    result={}
    for key,items in [('all',objects),('Dead',[o for o in objects if o['class']==4]),('interior_Dead',[o for o in objects if o['class']==4 and o['interior']])]:
        result[key]={'n':len(items),'initial_good':sum(o['initial_iou']>.5 for o in items),'final_good':sum(o['final_iou']>.5 for o in items),
                     'initial_mean_iou':sum(o['initial_iou'] for o in items)/max(1,len(items)),
                     'final_mean_iou':sum(o['final_iou'] for o in items)/max(1,len(items))}
        for stage in ('initial','final'):result[key][stage+'_coverage']=result[key][stage+'_good']/max(1,len(items))
    return result

def run(a):
    start=time.monotonic();os.chdir(a.snapshot);sys.path.insert(0,str(a.snapshot/'src'))
    from torch.utils.data import default_collate
    from nucseg.setmask.engine import load_config,build_model
    from nucseg.setmask.data import PairedDataset
    from nucseg.setmask.matching import make_targets,match_queries
    from nucseg.visibility_probe import original_full
    def deadline():
        if time.monotonic()-start>1750:raise RuntimeError('E12 wall budget exhausted; stop, no retry')
    assert torch.cuda.device_count()==1 and shutil.disk_usage(a.out.parent).free>20*1024**3
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    assert sha(a.control)=='8e9c81ba393f7fcec19a14692191edb4382a666eaf870cec6a24a502ea0d55ed'
    control=json.loads(a.control.read_text());source=json.loads((a.snapshot/'MANIFEST.json').read_text())
    assert source['commit']==control['manifest']['source_commit']
    for group in ('sources','vendor'):
        for name,digest in source[group].items():assert sha(a.snapshot/name)==digest,name
    ds=PairedDataset(1,False,19);present=ds.class_presence()[:,3]
    groups={}
    for group,ids in control['manifest']['groups'].items():
        pos=[i for i in ids if present[i]][:4];neg=[i for i in ids if not present[i]][:4]
        assert len(pos)==len(neg)==4
        groups[group]=[i for i in ids if i in pos+neg]
    a.out.mkdir(exist_ok=False)
    manifest={'source_commit':source['commit'],'script_sha256':sha(__file__),'design_sha256':sha(a.design),
              'control_sha256':sha(a.control),'groups':groups,'steps_per_image':300,'lr':.1,'betas':[.9,.999],
              'weight_decay':0,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'checkpoints':{},
              'scope':'Both inherited fit/audit groups fitted with GT identities; not validation/test or deployable method'}
    save(a.out/'manifest.json',manifest);results={}
    for seed in (19,1):
        torch.manual_seed(seed)
        run_dir=a.run_root/f'Q_seed{seed}';cfg=load_config(run_dir);receipt=json.loads((run_dir/'completion.json').read_text())
        assert cfg.round=='E7' and cfg.arm=='Q' and cfg.seed==seed
        for name in ('config.json','provenance.json','log.jsonl','final.pth'):assert sha(run_dir/name)==receipt['hashes'][name]
        assert receipt['hashes']['final.pth']==control['manifest']['checkpoints'][str(seed)]
        state=torch.load(run_dir/'final.pth',map_location='cpu',mmap=True,weights_only=False)
        assert state['config']==asdict(cfg)
        prov=state['provenance'];assert torch.__version__==prov['torch'] and np.__version__==prov['numpy']
        for name in ('images','inst','type','tissue'):assert sha(ds.fold.dir/(name+'.npy'))==prov['inputs']['fold1/'+name+'.npy']
        assert sha(cfg.uni_ckpt)==prov['inputs']['UNI']
        model=build_model(cfg,pretrained=False).eval().to('cuda');model.load_state_dict(state['model']);del state
        model.requires_grad_(False);rows=[]
        for group,images in groups.items():
            for image in images:
                batch=default_collate([ds[image]])
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(batch['img'].to('cuda'))
                target=make_targets(batch['inst_map'],batch['tp_map'])[0];count=len(target['ids'])
                assert count>0
                qi,ti=match_queries(pred['class_logits'],pred['mask_embed'],pred['mask_features'],[target])[0]
                assert len(qi)==count and sorted(ti.tolist())==list(range(count))
                assigned=torch.empty(count,dtype=torch.long,device='cuda');assigned[ti.to('cuda')]=qi.to('cuda')
                feat=pred['mask_features'][0].detach().float().clone();before=tensor_sha(feat)
                init=pred['mask_embed'][0,assigned].detach().float().clone();e=torch.nn.Parameter(init.clone())
                truth=(target['inst'][None]==target['ids'][:,None,None]).to('cuda').float()
                with torch.no_grad():
                    expected=mask_logits(pred['mask_embed'][0,assigned],pred['mask_features'][0])
                    assert torch.equal(mask_logits(e,feat),expected)
                    initial=object_iou(expected,truth).cpu().tolist()
                del pred,expected
                opt=torch.optim.AdamW([e],lr=.1,betas=(.9,.999),weight_decay=0);logs=[]
                for step in range(300):
                    opt.zero_grad(set_to_none=True);value=0.
                    for left in range(0,count,16):
                        right=min(count,left+16)
                        loss=mask_loss(mask_logits(e[left:right],feat),truth[left:right])*(right-left)/count
                        assert bool(torch.isfinite(loss));loss.backward();value+=float(loss.detach())
                    assert e.grad is not None and bool(torch.isfinite(e.grad).all())
                    opt.step()
                    if step in (0,99,299):logs.append({'step':step+1,'loss':value})
                    deadline()
                with torch.no_grad():final=object_iou(mask_logits(e,feat),truth).cpu().tolist()
                assert tensor_sha(feat)==before and all(p.grad is None for p in model.parameters())
                interior=original_full(target['inst'].numpy())
                objects=[{'id':int(oid),'class':int(cls),'interior':bool(interior[int(oid)]),'area':int((target['inst']==oid).sum()),'initial_iou':v0,'final_iou':v1}
                         for oid,cls,v0,v1 in zip(target['ids'],target['classes'],initial,final)]
                row={'group':group,'image':image,'objects':objects,'optimization':logs,'feature_sha256':before,'frozen_features_unchanged':True,'initial_logits_exact':True}
                rows.append(row);save(a.out/f'seed{seed}_partial.json',{'rows':rows})
                print(json.dumps({'seed':seed,'image':image,'summary':summary([row]),'elapsed':time.monotonic()-start}),flush=True)
                del e,opt,feat,truth,init,loss;torch.cuda.empty_cache();deadline()
        results[str(seed)]={'rows':rows,'summary':summary(rows),'by_inherited_group':{g:summary([r for r in rows if r['group']==g]) for g in groups}}
        manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth'];save(a.out/f'seed{seed}.json',results[str(seed)])
        del model;torch.cuda.empty_cache()
    witness=all(v['summary'][k]['final_coverage']>=.8 for v in results.values() for k in ('all','Dead'))
    payload={'manifest':manifest,'results':results,'fitting_witness':witness,'method_admission':False,'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    save(a.out/'results.json',payload);print(json.dumps({k:v for k,v in payload.items() if k not in ('results','manifest')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('snapshot','control','design','run-root','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    run(a)
