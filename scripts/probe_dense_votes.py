"""E18 dense foreground-to-instance-point voting and conditional-mask diagnostic."""
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

class DenseVoteHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,8,1))
        nn.init.zeros_(self[-1].bias)
        with torch.no_grad():self[-1].bias[0]=math.log(.1/.9)

def dense_targets(target):
    inst=target['inst'];h,w=inst.shape;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    offsets=torch.zeros(2,h,w);classes=torch.zeros(h,w,dtype=torch.long)
    for oid,point,cls in zip(target['ids'],target['points'],target['classes']):
        mask=inst==oid;offsets[0,mask]=(point[0]-y[mask])/32;offsets[1,mask]=(point[1]-x[mask])/32;classes[mask]=cls-1
    return {'inst':inst,'ids':target['ids'],'foreground':(inst>0).float(),'offset':offsets,'classes':classes}

def dense_losses(pred,target):
    device=pred.device;inst=target['inst'].to(device)
    foreground=F.binary_cross_entropy_with_logits(pred[0],target['foreground'].to(device))
    offsets=(pred[1:3]-target['offset'].to(device)).abs().mean(0)
    classes=F.cross_entropy(pred[3:][None],target['classes'].to(device)[None],reduction='none')[0]
    off=[];cls=[]
    for oid in target['ids']:
        mask=inst==int(oid);off.append(offsets[mask].mean());cls.append(classes[mask].mean())
    zero=pred.sum()*0
    return {'foreground':foreground,'offset':torch.stack(off).mean() if off else zero,'class':torch.stack(cls).mean() if cls else zero}

@torch.no_grad()
def vote_proposals(pred,capacity=512):
    if not bool(torch.isfinite(pred).all()):raise ValueError('Nonfinite dense voting predictions')
    h,w=pred.shape[-2:];prob=pred[0].sigmoid();yx=torch.nonzero(prob>=.5,as_tuple=False)
    if not len(yx):
        return (torch.empty(0,2,dtype=torch.long,device=pred.device),pred.new_empty(0),torch.empty(0,dtype=torch.long,device=pred.device),pred.new_zeros(h,w))
    y,x=yx[:,0],yx[:,1];cy=(y+32*pred[1,y,x]).round().clamp(0,h-1).long();cx=(x+32*pred[2,y,x]).round().clamp(0,w-1).long()
    index=(cy*w+cx).cpu().numpy();weights=prob[y,x].double().cpu().numpy();cp=pred[3:,y,x].T.softmax(-1).double().cpu().numpy()
    votes=np.zeros(h*w,dtype=np.float64);maximum=np.zeros(h*w,dtype=np.float64);class_votes=np.zeros((5,h*w),dtype=np.float64)
    np.add.at(votes,index,weights);np.maximum.at(maximum,index,weights)
    for c in range(5):np.add.at(class_votes[c],index,weights*cp[:,c])
    image=torch.from_numpy(votes.reshape(h,w));local=F.max_pool2d(image[None,None],3,1,1)[0,0].numpy()
    ids=np.flatnonzero((votes>=.5)&(votes==local.ravel()));ids=ids[np.lexsort((ids,-votes[ids]))[:capacity]]
    cls=class_votes[:,ids].argmax(0);confidence=maximum[ids]*class_votes[cls,ids]/class_votes[:,ids].sum(0).clip(1e-12)
    points=torch.as_tensor(np.stack((ids//w,ids%w),axis=1),device=pred.device,dtype=torch.long)
    return points,torch.as_tensor(confidence,device=pred.device,dtype=torch.float32),torch.as_tensor(cls+1,device=pred.device,dtype=torch.long),image.to(pred.device)

@torch.no_grad()
def decode_points(logits,readout,native,basis,mean,std):
    points,confidence,classes,_=vote_proposals(logits);h,w=native.shape[-2:]
    inst=torch.zeros(h,w,dtype=torch.int32,device=native.device);typ=torch.zeros(h,w,dtype=torch.uint8,device=native.device)
    if not len(points):return inst.cpu().numpy(),typ.cpu().numpy(),{'points':points,'params':native.new_empty(0,133),'confidence':confidence}
    params=readout(point_inputs(native,points,mean,std));best=torch.full((h,w),float('-inf'),device=native.device)
    for left in range(0,len(points),16):
        right=min(len(points),left+16);m=point_logits(params[left:right],basis,points[left:right]).sigmoid()
        weighted=(m*confidence[left:right,None,None]).masked_fill(m<.5,float('-inf'));top,which=weighted.max(0);replace=top>best;rank=which+left
        inst[replace]=(rank[replace]+1).to(torch.int32);typ[replace]=classes[rank[replace]].to(torch.uint8);best=torch.maximum(best,top)
    return inst.cpu().numpy(),typ.cpu().numpy(),{'points':points,'params':params,'confidence':confidence}



def point_allocation(rows):
    keys=('dead_recall','recall','precision','dead_typed_f1');details={}
    for seed in ('19','1'):
        a,s=rows[seed]['A0'],rows[seed]['V'];delta={k:s[k]-a[k] for k in keys}
        guard=all(v>=0 for v in delta.values())
        gate=s['fit_recall']>=.8 and delta['dead_recall']>=.05 and all(delta[k]>=0 for k in keys[1:])
        details[seed]={'original_gate':gate,'a0_guard':guard,'v_minus_a0':delta,'fit_recall':s['fit_recall']}
    return {'original_gate':all(d['original_gate'] for d in details.values()),'a0_guard':all(d['a0_guard'] for d in details.values()),'promising':all(d['original_gate'] and d['a0_guard'] for d in details.values()),'per_seed':details,
            'scope':'New16trainimage actual-inference small allocation; not historical64/66 gate, validation or test'}

def aggregate_rows(rows):
    names=('n','pred','tp','dead_n','dead_good','dead_typed_tp','dead_pred','raw_dead_good','interior_dead_n','interior_dead_good')
    c={k:sum(r[k] for r in rows) for k in names}
    return {'recall':c['tp']/max(1,c['n']),'precision':c['tp']/max(1,c['pred']),'dead_recall':c['dead_good']/max(1,c['dead_n']),
            'dead_typed_f1':2*c['dead_typed_tp']/max(1,c['dead_n']+c['dead_pred']),'raw_dead_recall':c['raw_dead_good']/max(1,c['dead_n']),
            'interior_dead_recall':c['interior_dead_good']/max(1,c['interior_dead_n']),'counts':c,'per_image':rows}

@torch.no_grad()
def measure_detector(readout,center,cache,mean,std,deadline):
    from nucseg.metrics.instance import overlap,pq_from_overlap
    from nucseg.metrics.pannuke_eval import instance_classes
    rows=[];mean=mean.to('cuda');std=std.to('cuda')
    for r in cache:
        native=r['feature'][0].to('cuda').float();basis=r['basis'].to('cuda')
        inp=(native-mean[256:320,None,None])/std[256:320,None,None]
        logits=center(inp[None])[0];pi,pt,aux=decode_points(logits,readout,native,basis,mean,std)
        target=r['target'];classes=target['classes'].numpy();ov=overlap(target['inst'].numpy(),pi);match=pq_from_overlap(ov)
        _,pc=instance_classes(pi,pt);good=np.zeros(ov.nt,dtype=bool);typed=good.copy()
        if len(match.paired):
            good[match.paired[:,0]]=True;typed[match.paired[:,0]]=classes[match.paired[:,0]]==pc[match.paired[:,1]]
        dead=classes==4;idead=np.asarray([bool(r['interior'][int(i)]) for i in target['ids']])&dead
        ids=target['ids'][torch.from_numpy(dead)];raw_good=torch.zeros(len(ids),dtype=torch.bool,device='cuda')
        if len(ids):
            truth=(target['inst'][None]==ids[:,None,None]).to('cuda').flatten(1).float();area=truth.sum(1)
            for left in range(0,len(aux['points']),16):
                right=min(len(aux['points']),left+16)
                masks=(point_logits(aux['params'][left:right],basis,aux['points'][left:right])>=0).flatten(1).float();inter=masks@truth.T
                iou=inter/(masks.sum(1)[:,None]+area[None]-inter).clamp_min(1);raw_good|=(iou>.5).any(0)
        rows.append({'image':r['image'],'candidate_points':len(aux['points']),'n':ov.nt,'pred':ov.np_,'tp':match.tp,'dead_n':int(dead.sum()),'dead_good':int((dead&good).sum()),
                     'dead_typed_tp':int((dead&good&typed).sum()),'dead_pred':int((pc==4).sum()),'raw_dead_good':int(raw_good.sum()),
                     'interior_dead_n':int(idead.sum()),'interior_dead_good':int((idead&good).sum())})
        deadline()
    return aggregate_rows(rows)

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
    assert helper.sha(a.detector_control)=='161962e950bdad55e2374c511980b493083afbfc32b228f2b954a5a5e466d6ef'
    detector_reference=json.loads(a.detector_control.read_text());assert not detector_reference['allocation']['promising']
    assert detector_reference['manifest']['groups']==groups
    assert helper.sha(a.point_control)=='837d04404cc07465b494ee7b168bb3b37ab87243a37d2d47f93950aedcc9a06d'
    prior=json.loads(a.point_control.read_text())
    assert helper.sha(a.context_base)=='250fd5bb14b929f864c758a1293064c9e488ff783457a75c2c19996f62622b5d'
    assert helper.sha(a.context_control)=='9b70fb76c535d8028c03f3f1bfa6e15164fdd4d814d48254befb654b8ae93c3a'
    context_reference=json.loads(a.context_control.read_text());assert context_reference['point_detector_probe_supported']
    assert context_reference['manifest']['groups']==groups
    assert helper.sha(a.measure)=='8432b771d2fe1b69b1633d748659a668ddad7095146f597ddd5a0e06d3df51f1'
    assert helper.sha(a.a0_control)=='8e9c81ba393f7fcec19a14692191edb4382a666eaf870cec6a24a502ea0d55ed'
    measurement=load(a.measure);a0_reference=json.loads(a.a0_control.read_text())
    assert not prior['point_detector_probe_supported']
    assert prior['manifest']['groups']==groups and prior['manifest']['checkpoints']==reference['manifest']['checkpoints']
    assert len(groups['fit'])==len(groups['audit'])==8
    source=json.loads((a.snapshot/'MANIFEST.json').read_text())
    assert source['commit']==reference['manifest']['source_commit']
    for group in ('sources','vendor'):
        for name,digest in source[group].items():assert helper.sha(a.snapshot/name)==digest,name
    ds=PairedDataset(1,False,19);a.out.mkdir(exist_ok=False)
    manifest={'source_commit':source['commit'],'groups':groups,'script_sha256':helper.sha(__file__),'design_sha256':helper.sha(a.design),
              'control_sha256':helper.sha(a.control),'point_control_sha256':helper.sha(a.point_control),'context_offsets':[-4,0,4],'free_base_sha256':helper.sha(a.free_base),'steps':300,'fit_images_per_step':1,'detector_control_sha256':helper.sha(a.detector_control),'proposal':'dense foreground-to-point votes, deterministic CPUfloat64 accumulation','context_source_sha256':helper.sha(a.context_base),'context_control_sha256':helper.sha(a.context_control),'a0_control_sha256':helper.sha(a.a0_control),'measure_sha256':helper.sha(a.measure),
              'lr':3e-4,'betas':[.9,.999],'weight_decay':0,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'checkpoints':{},
              'scope':'Actual predicted-point inference; GT only for training/evaluation;8fit/8audit seen byE7; no validation/test claim'}
    helper.save(a.out/'manifest.json',manifest);results={};gate_rows={}
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
                frozen_feature=captured.pop().cpu();feature=frozen_feature[0].to('cuda').float();target=make_targets(batch['inst_map'],batch['tp_map'])[0];points=target['points'].to('cuda')
                assert len(target['ids'])>0 and len(target['ids'])<=512
                assert torch.equal(target['inst'][target['points'][:,0],target['points'][:,1]],target['ids'])
                basis=pred['mask_features'][0].detach().float().cpu().clone()
                inputs=point_inputs(feature,points,torch.zeros(576,device='cuda'),torch.ones(576,device='cuda')).cpu()
                interior=original_full(target['inst'].numpy())
                cache.append({'feature':frozen_feature,'batch':batch,'tissue_logits':pred['tissue'].cpu(),'image':image,'group':group,'basis':basis,'inputs':inputs,'points':points.cpu(),'target':target,'interior':interior,
                              'basis_sha256':helper.tensor_sha(basis),'input_sha256':helper.tensor_sha(inputs)})
                old=next(r for r in prior['results'][str(seed)]['rows'] if r['image']==image)
                assert helper.tensor_sha(basis)==old['basis_sha256']
                del pred,feature,points,basis,inputs;deadline()
        hook.remove();assert not captured and all(p.grad is None for p in model.parameters())
        original={g:measurement.measure(model.query_head,[r for r in cache if r['group']==g],False,start) for g in groups}
        for g,ids in groups.items():
            expected=[next(row for row in a0_reference['results'][str(seed)]['A0'][g]['per_image'] if row['image']==image) for image in ids]
            assert original[g]['per_image']==expected,'A0 mismatch, do not proceed'
        print(json.dumps({'seed':seed,'A0_per_image_exact':True}),flush=True)
        del model;torch.cuda.empty_cache()
        vectors=torch.cat([r['inputs'][:,:576] for r in cache if r['group']=='fit']);mean=vectors.mean(0);std=vectors.std(0,unbiased=False).clamp_min(1e-3)
        for r in cache:
            r['normalized']=torch.cat(((r['inputs'][:,:576]-mean)/std,r['inputs'][:,576:]),dim=1)
            r['dense_target']=dense_targets(r['target'])
        torch.manual_seed(seed);head=PointReadout().to('cuda');torch.manual_seed(seed+1701);center=DenseVoteHead().to('cuda')
        trainable=list(head.parameters())+list(center.parameters());optimizer=torch.optim.AdamW(trainable,lr=3e-4,betas=(.9,.999),weight_decay=0)
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
            native=r['feature'][0].to('cuda').float();dense=(native-mean[256:320,None,None].to('cuda'))/std[256:320,None,None].to('cuda')
            prediction=center(dense[None])[0]
            dense_terms=dense_losses(prediction,r['dense_target'])
            ploss=2*dense_terms['foreground']+dense_terms['offset']
            closs=2*dense_terms['class']
            assert bool(torch.isfinite(ploss+closs));(ploss+closs).backward()
            for left in range(0,count,16):
                right=min(count,left+16);truth=(target['inst'][None]==target['ids'][left:right,None,None]).to('cuda').float()
                loss=native_mask_loss(point_logits(head(inputs[left:right]),basis,points[left:right]),truth)*(right-left)/count
                assert bool(torch.isfinite(loss));loss.backward();value+=float(loss.detach())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable);optimizer.step()
            if step in (0,99,299):logs.append({'step':step+1,'image':r['image'],'loss':value+float(ploss.detach())+float(closs.detach()),'mask_loss':value,'point_loss':float(ploss.detach()),'class_loss':float(closs.detach()),'foreground_loss':float(dense_terms['foreground'].detach()),'offset_loss':float(dense_terms['offset'].detach())});print(json.dumps({'seed':seed,**logs[-1]}),flush=True)
            deadline()
        final=measure();detector={g:measure_detector(head,center,[r for r in cache if r['group']==g],mean,std,deadline) for g in groups};rows=[]
        for r,v0,v1 in zip(cache,initial,final):
            assert helper.tensor_sha(r['basis'])==r['basis_sha256'] and helper.tensor_sha(r['inputs'])==r['input_sha256']
            target=r['target'];objects=[{'id':int(oid),'class':int(cls),'area':int((target['inst']==oid).sum()),'point':p.tolist(),'interior':bool(r['interior'][int(oid)]),'initial_iou':x,'final_iou':y}
                                       for oid,cls,p,x,y in zip(target['ids'],target['classes'],r['points'],v0,v1)]
            rows.append({'image':r['image'],'group':r['group'],'objects':objects,'basis_sha256':r['basis_sha256'],'input_sha256':r['input_sha256'],'features_unchanged':True})
        results[str(seed)]={'rows':rows,'optimization':logs,'by_group':{g:helper.summary([r for r in rows if r['group']==g]) for g in groups},
                            'A0':original,'V':detector,'normalization_fit_points_only':True,'mean':mean.tolist(),'std':std.tolist(),'frozen_model_grad_none':True}
        gate_rows[str(seed)]={'A0':original['audit'],'V':{**detector['audit'],'fit_recall':detector['fit']['recall']}}
        manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth'];helper.save(a.out/f'seed{seed}.json',results[str(seed)])
        print(json.dumps({'seed':seed,'GT_point_by_group':results[str(seed)]['by_group'],'V':{g:{k:v for k,v in detector[g].items() if k!='per_image'} for g in groups}}),flush=True)
        del head,center,optimizer,cache,basis,inputs,points,truth,loss,native,dense,prediction,ploss,closs,trainable;torch.cuda.empty_cache()
    continuation=all(r['by_group']['fit'][k]['final_coverage']>=.8 for r in results.values() for k in ('all','Dead'))
    payload={'manifest':manifest,'results':results,'gt_point_fitting_witness':continuation,'allocation':point_allocation(gate_rows),'large_training_started':False,'method_admission':False,'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    helper.save(a.out/'results.json',payload);print(json.dumps({k:v for k,v in payload.items() if k not in ('manifest','results')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('snapshot','free-base','control','point-control','context-control','context-base','measure','a0-control','detector-control','design','run-root','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    run(a)
