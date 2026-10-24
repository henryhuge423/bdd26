"""GT-point local-context conditional-mask diagnostic; no detector."""
import argparse,hashlib,importlib.util,json,math,os,shutil,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PointReadout(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(578,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,133))
        nn.init.normal_(self[-1].weight,std=.001);nn.init.zeros_(self[-1].bias)
        with torch.no_grad():
            self[-1].bias[128]=2;self[-1].bias[131:]=math.log(math.expm1(32))

def point_inputs(features,points,mean,std):
    h,w=features.shape[-2:];y,x=points[:,0],points[:,1]
    vectors=torch.cat([features[:,(y+dy).clamp(0,h-1),(x+dx).clamp(0,w-1)].T.float() for dy in (-4,0,4) for dx in (-4,0,4)],dim=1)
    xy=torch.stack((2*x.float()/max(1,w-1)-1,2*y.float()/max(1,h-1)-1),dim=1)
    return torch.cat(((vectors-mean)/std.clamp_min(1e-3),xy),dim=1)

def point_logits(params,basis,points):
    h,w=basis.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=basis.device),torch.arange(w,device=basis.device),indexing='ij')
    dx=(x[None]-points[:,1,None,None])/32;dy=(y[None]-points[:,0,None,None])/32
    z=torch.einsum('qc,chw->qhw',params[:,:128].float(),basis.float())/math.sqrt(128)
    c=params[:,128:,None,None]
    return z+c[:,0]+c[:,1]*dx+c[:,2]*dy-F.softplus(c[:,3])*dx.square()-F.softplus(c[:,4])*dy.square()

def native_mask_loss(logits,truth):
    bce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none').mean((1,2))
    p=logits.sigmoid().flatten(1);t=truth.flatten(1)
    return (5*bce+5*(1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1))).mean()

def run(a):
    start=time.monotonic();os.chdir(a.snapshot);sys.path.insert(0,str(a.snapshot/'src'))
    from torch.utils.data import default_collate
    from nucseg.setmask.engine import load_config,build_model
    from nucseg.setmask.data import PairedDataset
    from nucseg.setmask.matching import make_targets
    from nucseg.visibility_probe import original_full
    def deadline():
        if time.monotonic()-start>1750:raise RuntimeError('E15 budget exhausted; stop,no retry')
    def load(path):
        spec=importlib.util.spec_from_file_location('frozen_e12',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    assert torch.cuda.device_count()==1 and shutil.disk_usage(a.out.parent).free>20*1024**3
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    assert hashlib.sha256(a.free_base.read_bytes()).hexdigest()=='dc095f7cf06d1019504a9443aca2af54cc5e861b8470036f45452b8d776d8741'
    helper=load(a.free_base)
    assert helper.sha(a.control)=='94d0f68bf930328b155c4c4d2043f3bc6c570e724a5cbaf5719d28263f81c6d3'
    reference=json.loads(a.control.read_text());groups=reference['manifest']['groups']
    assert helper.sha(a.point_control)=='837d04404cc07465b494ee7b168bb3b37ab87243a37d2d47f93950aedcc9a06d'
    prior=json.loads(a.point_control.read_text())
    assert not prior['point_detector_probe_supported']
    assert prior['manifest']['groups']==groups and prior['manifest']['checkpoints']==reference['manifest']['checkpoints']
    assert len(groups['fit'])==len(groups['audit'])==8
    source=json.loads((a.snapshot/'MANIFEST.json').read_text())
    assert source['commit']==reference['manifest']['source_commit']
    for group in ('sources','vendor'):
        for name,digest in source[group].items():assert helper.sha(a.snapshot/name)==digest,name
    ds=PairedDataset(1,False,19);a.out.mkdir(exist_ok=False)
    manifest={'source_commit':source['commit'],'groups':groups,'script_sha256':helper.sha(__file__),'design_sha256':helper.sha(a.design),
              'control_sha256':helper.sha(a.control),'point_control_sha256':helper.sha(a.point_control),'context_offsets':[-4,0,4],'free_base_sha256':helper.sha(a.free_base),'steps':300,'fit_images_per_step':1,
              'lr':3e-4,'betas':[.9,.999],'weight_decay':0,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'checkpoints':{},
              'scope':'GT points/identities given;8fit/8audit originally seen by E7; no detector or validation/test claim'}
    helper.save(a.out/'manifest.json',manifest);results={}
    for seed in (19,1):
        run_dir=a.run_root/f'Q_seed{seed}';cfg=load_config(run_dir);receipt=json.loads((run_dir/'completion.json').read_text())
        assert cfg.round=='E7' and cfg.arm=='Q' and cfg.seed==seed
        for name in ('config.json','provenance.json','log.jsonl','final.pth'):assert helper.sha(run_dir/name)==receipt['hashes'][name]
        assert receipt['hashes']['final.pth']==reference['manifest']['checkpoints'][str(seed)]
        state=torch.load(run_dir/'final.pth',map_location='cpu',mmap=True,weights_only=False);assert state['config']==asdict(cfg)
        prov=state['provenance'];assert torch.__version__==prov['torch'] and np.__version__==prov['numpy']
        for name in ('images','inst','type','tissue'):assert helper.sha(ds.fold.dir/(name+'.npy'))==prov['inputs']['fold1/'+name+'.npy']
        assert helper.sha(cfg.uni_ckpt)==prov['inputs']['UNI']
        model=build_model(cfg,pretrained=False).eval().to('cuda');model.load_state_dict(state['model']);model.requires_grad_(False);del state
        captured=[];hook=model.query_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        cache=[]
        for group,images in groups.items():
            for image in images:
                batch=default_collate([ds[image]])
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(batch['img'].to('cuda'))
                feature=captured.pop()[0].float();target=make_targets(batch['inst_map'],batch['tp_map'])[0];points=target['points'].to('cuda')
                assert len(target['ids'])>0 and len(target['ids'])<=512
                assert torch.equal(target['inst'][target['points'][:,0],target['points'][:,1]],target['ids'])
                basis=pred['mask_features'][0].detach().float().cpu().clone()
                inputs=point_inputs(feature,points,torch.zeros(576,device='cuda'),torch.ones(576,device='cuda')).cpu()
                interior=original_full(target['inst'].numpy())
                cache.append({'image':image,'group':group,'basis':basis,'inputs':inputs,'points':points.cpu(),'target':target,'interior':interior,
                              'basis_sha256':helper.tensor_sha(basis),'input_sha256':helper.tensor_sha(inputs)})
                old=next(r for r in prior['results'][str(seed)]['rows'] if r['image']==image)
                assert helper.tensor_sha(basis)==old['basis_sha256']
                del pred,feature,points,basis,inputs;deadline()
        hook.remove();assert not captured and all(p.grad is None for p in model.parameters());del model;torch.cuda.empty_cache()
        vectors=torch.cat([r['inputs'][:,:576] for r in cache if r['group']=='fit']);mean=vectors.mean(0);std=vectors.std(0,unbiased=False).clamp_min(1e-3)
        for r in cache:
            r['normalized']=torch.cat(((r['inputs'][:,:576]-mean)/std,r['inputs'][:,576:]),dim=1)
        torch.manual_seed(seed);head=PointReadout().to('cuda');optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=0)
        @torch.no_grad()
        def measure():
            out=[]
            for r in cache:
                basis=r['basis'].to('cuda');inputs=r['normalized'].to('cuda');points=r['points'].to('cuda');target=r['target'];ious=[]
                for left in range(0,len(inputs),16):
                    right=min(len(inputs),left+16)
                    truth=(target['inst'][None]==target['ids'][left:right,None,None]).to('cuda').float()
                    logits=point_logits(head(inputs[left:right]),basis,points[left:right])
                    ious+=helper.object_iou(logits,truth).cpu().tolist()
                out.append(ious);deadline()
            return out
        initial=measure();fit=[r for r in cache if r['group']=='fit'];logs=[]
        for step in range(300):
            r=fit[step%8];basis=r['basis'].to('cuda');inputs=r['normalized'].to('cuda');points=r['points'].to('cuda');target=r['target'];count=len(inputs)
            optimizer.zero_grad(set_to_none=True);value=0.
            for left in range(0,count,16):
                right=min(count,left+16);truth=(target['inst'][None]==target['ids'][left:right,None,None]).to('cuda').float()
                loss=native_mask_loss(point_logits(head(inputs[left:right]),basis,points[left:right]),truth)*(right-left)/count
                assert bool(torch.isfinite(loss));loss.backward();value+=float(loss.detach())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters());optimizer.step()
            if step in (0,99,299):logs.append({'step':step+1,'image':r['image'],'loss':value});print(json.dumps({'seed':seed,**logs[-1]}),flush=True)
            deadline()
        final=measure();rows=[]
        for r,v0,v1 in zip(cache,initial,final):
            assert helper.tensor_sha(r['basis'])==r['basis_sha256'] and helper.tensor_sha(r['inputs'])==r['input_sha256']
            target=r['target'];objects=[{'id':int(oid),'class':int(cls),'area':int((target['inst']==oid).sum()),'point':p.tolist(),'interior':bool(r['interior'][int(oid)]),'initial_iou':x,'final_iou':y}
                                       for oid,cls,p,x,y in zip(target['ids'],target['classes'],r['points'],v0,v1)]
            rows.append({'image':r['image'],'group':r['group'],'objects':objects,'basis_sha256':r['basis_sha256'],'input_sha256':r['input_sha256'],'features_unchanged':True})
        results[str(seed)]={'rows':rows,'optimization':logs,'by_group':{g:helper.summary([r for r in rows if r['group']==g]) for g in groups},
                            'normalization_fit_points_only':True,'mean':mean.tolist(),'std':std.tolist(),'frozen_model_grad_none':True}
        manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth'];helper.save(a.out/f'seed{seed}.json',results[str(seed)])
        print(json.dumps({'seed':seed,'by_group':results[str(seed)]['by_group']}),flush=True)
        del head,optimizer,cache,basis,inputs,points,truth,loss;torch.cuda.empty_cache()
    continuation=all(r['by_group']['fit'][k]['final_coverage']>=.8 for r in results.values() for k in ('all','Dead'))
    payload={'manifest':manifest,'results':results,'point_detector_probe_supported':continuation,'method_admission':False,'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    helper.save(a.out/'results.json',payload);print(json.dumps({k:v for k,v in payload.items() if k not in ('manifest','results')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('snapshot','free-base','control','point-control','design','run-root','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    run(a)
