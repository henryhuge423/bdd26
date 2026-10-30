"""P21: joint semantic auxiliary center supervision; original main-map inference."""
import argparse
import json
import math
from numbers import Integral
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from proposal_p3_utils import (split_indices,gt_objects,gaussian_target,augment_pair,
    focal_numerator,decode_peaks,score_points,aggregate,admission)
from train_proposal_p3 import (Proposal as OriginalProposal,Fold as OriginalFold,WEIGHT_SHA,DATA_SHA,
    sha,dump,checkpoint,evaluate,load_adapted)
from p5_contracts import validate_cohort,TISSUE_SHA,verify_hashes


import torch.nn.functional as F
from train_proposal_p19 import effective_batches,frozen_hashes


from torch import nn
from train_proposal_p3 import PointHead as OriginalPointHead

def typed_targets(gt):
    targets=[gaussian_target(gt)]
    targets.extend(gaussian_target([g for g in gt if g[0]==c]) for c in range(1,6))
    return np.stack(targets)

def smoke_ids(fit,gt):
    first=list(fit[:118]);later=[i for i in fit[118:] if any(g[0]==4 for g in gt[i])][:16]
    if len(first)!=118 or len(later)!=16 or len(set(first+later))!=134:
        raise ValueError('P21 needs distinct fixed134 fit smoke images')
    return first+later

class SemanticPointHead(OriginalPointHead):
    def __init__(self,original=None):
        if original is None:super().__init__()
        else:
            nn.Module.__init__(self)
            self.proj,self.stem,self.up,self.out=original.proj,original.stem,original.up,original.out
        with torch.random.fork_rng(devices=[]):
            self.auxiliary=nn.Conv2d(64,5,1)
            nn.init.zeros_(self.auxiliary.weight)
            nn.init.constant_(self.auxiliary.bias,math.log(.01/.99))

    def forward_with_aux(self,feats):
        x=torch.cat([p(f) for p,f in zip(self.proj,feats)],1);x=F.relu(self.stem(x))
        for layer in self.up:x=F.relu(layer(F.interpolate(x,scale_factor=2,mode='nearest')))
        main=F.interpolate(self.out(x),size=(256,256),mode='bilinear',align_corners=False)[:,0]
        aux=F.interpolate(self.auxiliary(x),size=(256,256),mode='bilinear',align_corners=False)
        return main,aux

class Proposal(OriginalProposal):
    def __init__(self,weights,trainable=True,arm='C'):
        if trainable is not True or arm not in ('C','A'):raise ValueError('P21 requires C/A')
        super().__init__(weights,True);self.arm=arm
        if arm=='A':self.head=SemanticPointHead(self.head)
        self.observed_input_size=256;self.observed_token_grids=[16]*4

    def features(self,x):
        if x.ndim!=4 or x.shape[0]<1 or tuple(x.shape[1:])!=(3,256,256):
            raise ValueError('P21 requires nonempty nativeRGB256 input')
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        _,tokens=self.encoder((x-mean)/std)
        if len(tokens)!=4 or any(tuple(t.shape[1:])!=(257,1024) for t in tokens):
            raise ValueError('P21 requires four16x16 native token taps')
        return [F.layer_norm(t[:,1:],(1024,)).transpose(1,2).reshape(len(t),1024,16,16) for t in tokens]

    def forward(self,x):
        if self.arm=='C':return super().forward(x)
        return self.head(self.features(x))

    def forward_with_aux(self,x):
        if self.arm=='C':return self.forward(x),None
        return self.head.forward_with_aux(self.features(x))

class Fold(OriginalFold):
    def __init__(self,root,fold,arm):
        if fold!=1 or arm not in ('C','A'):raise ValueError('P21 is fold1 C/A only')
        super().__init__(root,fold);self.arm=arm

    def batch(self,ids,seed,epoch,aug):
        if self.arm=='C':return super().batch(ids,seed,epoch,aug)
        images=[];targets=[];count=0
        for i in ids:
            im=torch.from_numpy(self.images[i].copy()).permute(2,0,1).float()/255
            target=torch.from_numpy(typed_targets(self.gt[i]));count+=len(self.gt[i])
            if aug:im,target=augment_pair(im,target,np.random.default_rng(np.random.SeedSequence([seed,epoch,int(i),987])))
            images.append(im);targets.append(target)
        return torch.stack(images),torch.stack(targets),count

def loss_for_microbatch(logits,target,total_count,arm,auxiliary=None):
    if (arm not in ('C','A') or logits.ndim!=3 or not len(logits)
        or not isinstance(total_count,Integral) or isinstance(total_count,(bool,np.bool_)) or total_count<0):
        raise ValueError('invalid P21 loss metadata')
    if arm=='C':
        if target.shape!=logits.shape or auxiliary is not None:raise ValueError('invalid C loss shapes')
        return focal_numerator(logits,target)/max(1,total_count)
    if (target.shape!=(len(logits),6,*logits.shape[1:]) or auxiliary is None
        or auxiliary.shape!=(len(logits),5,*logits.shape[1:])):raise ValueError('invalid A loss shapes')
    return (focal_numerator(logits,target[:,0])+focal_numerator(auxiliary,target[:,1:]))/max(1,total_count)

def run_config(arm,seed,microbatch=4):
    if arm not in ('C','A') or type(seed) is not int or seed not in (19,1) or type(microbatch) is not int or microbatch!=4:
        raise ValueError('invalid P21 configuration')
    return dict(phase='A',arm=arm,seed=seed,target_sigma=1,target_radius=3,
        augmentation=dict(kind='original'),trainable_encoder_blocks=list(range(12,24)),
        architecture='original_UNI_last12' if arm=='C' else 'UNI_last12_semantic_auxiliary',
        effective_batch=16,microbatch=4,input_size=256,token_grid=16,output_size=256,
        auxiliary=None if arm=='C' else dict(classes=[1,2,3,4,5],coefficient=1,channels=64,parameters=325,decoder_lr=3e-4))

def run_budget_hours(arm):
    if arm not in ('C','A'):raise ValueError('invalid P21 resource arm')
    return 12

def match_points(points,gt):
    from scipy.optimize import linear_sum_assignment
    if not points or not gt:return []
    pos=np.asarray([g[1] for g in gt],float)
    dist=((pos[:,None,:]-np.asarray(points,float)[None,:,:])**2).sum(-1)
    cost=np.full((len(gt),len(points)+len(gt)),1e6)
    cost[:,:len(points)]=np.where(dist<=9,dist,2e6)
    rows,cols=linear_sum_assignment(cost)
    return [(int(r),int(c)) for r,c in zip(rows,cols) if c<len(points) and dist[r,c]<=9]

def select_examples(ids,gt,limit=8):
    ids=sorted(ids)
    dead=[i for i in ids if any(g[0]==4 and g[2] for g in gt[i])]
    empty=[i for i in ids if not gt[i]]
    other=[i for i in ids if gt[i] and i not in dead]
    selected=dead[:limit//2]+empty[:limit//4]+other[:limit-len(dead[:limit//2])-len(empty[:limit//4])]
    return (selected+[i for i in ids if i not in selected])[:limit]

def render_example(image,gt,points,out,title):
    from PIL import Image,ImageDraw
    pairs=match_points(points,gt);hits={r for r,c in pairs};matched={c for r,c in pairs}
    canvas=Image.new('RGB',(1024,310),'white');draw=ImageDraw.Draw(canvas)
    for col,label in enumerate(('RGB','GT: magenta=interiorDead','Predicted main-map points','Green=match red=miss orange=extra')):
        canvas.paste(Image.fromarray(np.asarray(image,dtype=np.uint8)),(col*256,42))
        draw.text((col*256+3,26),label,fill='black')
    draw.text((4,4),title,fill='black')
    def circle(col,pos,color,r=3):
        y,x=pos;x+=col*256;y+=42
        draw.ellipse((x-r,y-r,x+r,y+r),outline=color,width=1)
    for i,g in enumerate(gt):
        circle(1,g[1],'magenta' if g[0]==4 and g[2] else 'cyan')
        if i not in hits:circle(3,g[1],'red',4)
    for j,point in enumerate(points):
        circle(2,point,'lime')
        circle(3,point,'lime' if j in matched else 'orange')
    draw.text((4,299),f'GT={len(gt)} predictions={len(points)} matched={len(pairs)} missed={len(gt)-len(pairs)} extra={len(points)-len(pairs)}; radius3 native pixels',fill='black')
    canvas.save(out)

@torch.no_grad()
def write_examples(model,fold,ids,out,scope,limit):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);model.eval();records=[]
    chosen=select_examples(ids,fold.gt,limit)
    for i in chosen:
        x=torch.from_numpy(fold.images[i].copy()).permute(2,0,1).float()[None].cuda()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):prob=model(x).float().sigmoid()[0].cpu().numpy()
        points=decode_peaks(prob);gt=fold.gt[i];pairs=match_points(points,gt)
        row=score_points(points,gt);assert row['matched']==len(pairs)
        name=f'{scope}_{i:04d}.png'
        render_example(fold.images[i],gt,points,out/name,f'P21 {model.arm} {scope} image{i} tissue{fold.tissue[i]}')
        records.append(dict(image=int(i),scope=scope,tissue=str(fold.tissue[i]),gt=gt,points=points,unique_pairs=pairs,score=row,png=name))
    dump(out/'coordinates.json',dict(selection='GT-only deterministic strata',scope=scope,records=records,auxiliary_used_for_predictions=False))
    return records

def validate_config(saved,expected):
    if saved!=expected:raise ValueError('checkpoint P21 configuration mismatch')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['A'],default='A')
    p.add_argument('--arm',choices=['C','A'],required=True)
    p.add_argument('--seed',type=int,choices=[19,1],default=19)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--microbatch',type=int,choices=[4],default=4)
    return p.parse_args(argv)


def main():
    args=parse_args();config=run_config(args.arm,args.seed,args.microbatch);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic()
    manifest=dict(config=config,arguments=vars(args),weights_sha256=sha(args.weights),input_sha256={})
    assert manifest['weights_sha256']==WEIGHT_SHA
    for name,expected in DATA_SHA.items():
        if not name.startswith('fold1/'):continue
        actual=sha(Path(args.data)/name);assert actual==expected,(name,actual)
        manifest['input_sha256'][name]=actual
    manifest['input_sha256']['fold1/tissue.npy']=sha(Path(args.data)/'fold1/tissue.npy')
    assert manifest['input_sha256']['fold1/tissue.npy']==TISSUE_SHA
    fold=Fold(args.data,1,args.arm)
    fit,audit=split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    manifest.update(fit=fit,audit=audit,cohort_sha256=validate_cohort(fit,audit))
    assert len(fit)==2134 and len(audit)==522
    counts=np.asarray([len(fold.gt[i]) for i in fit])
    manifest['fit_count_summary']=dict(images=len(fit),objects=int(counts.sum()),empty_images=int((counts==0).sum()),count_percentiles=np.percentile(counts,[0,25,50,75,100]).tolist())
    manifest['fit_dead_summary']=dict(objects=sum(sum(g[0]==4 for g in fold.gt[i]) for i in fit),images=sum(any(g[0]==4 for g in fold.gt[i]) for i in fit),unique_centers=sum(len({g[1] for g in fold.gt[i] if g[0]==4}) for i in fit),all_centers=sum(len({g[1] for g in fold.gt[i]}) for i in fit))
    if args.smoke:manifest['smoke_fit']=smoke_ids(fit,fold.gt)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.85)
    model=Proposal(args.weights,True,args.arm).cuda()
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src'),source.__file__
    manifest.update(source=str(Path(source.__file__).resolve()),torch=torch.__version__,numpy=np.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in model.head.parameters()))
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.23.5'
    dump(out/'manifest.json',manifest)
    encoder_names=[n for n,p in model.encoder.named_parameters() if p.requires_grad]
    manifest['trainable_encoder_names']=encoder_names
    dump(out/'manifest.json',manifest)
    groups=[dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)]
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    optimizer_ids={id(p) for g in opt.param_groups for p in g['params']}
    assert optimizer_ids=={id(p) for p in model.parameters() if p.requires_grad}
    watched=dict(encoder=model.encoder.blocks[23].attn.qkv.weight,output=model.head.out.weight)
    if args.arm=='A':watched['auxiliary']=model.head.auxiliary.weight
    gradients={k:0. for k in watched};precision={};handles=[]
    if args.smoke:
        fit=manifest['smoke_fit'];epochs=1
        before={k:v.detach().clone() for k,v in watched.items()}
        frozen_before=frozen_hashes(model)
        block12_before=model.encoder.blocks[12].attn.qkv.weight.detach().clone()
        for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out)]:
            handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    else:epochs=130
    for epoch in range(epochs):
        if time.monotonic()-started>run_budget_hours(args.arm)*3600:raise TimeoutError('P21 run budget exhausted')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(fit).tolist()
        batch_lengths=[];object_counts=[];loss_sum=0.;num=0;torch.cuda.synchronize();epoch_start=time.monotonic()
        for step_index,selected in enumerate(effective_batches(ids,config['effective_batch'])):
            start=step_index*config['effective_batch'];total_count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
            step_loss=0.
            batch_lengths.append(len(selected));object_counts.append(total_count)
            for begin in range(0,len(selected),args.microbatch):
                batch=selected[begin:begin+args.microbatch];x,y,_=fold.batch(batch,args.seed,epoch,True);x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,auxiliary=model.forward_with_aux(x)
                    if args.smoke:
                        assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
                        assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
                        assert logits.dtype==torch.float32
                loss=loss_for_microbatch(logits,y,total_count,args.arm,auxiliary)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();step_loss+=float(loss.detach())
            if args.smoke and start>=16:
                for key,p in watched.items():
                    value=float(p.grad.norm());assert math.isfinite(value) and value>0,(key,value)
                    gradients[key]=max(gradients[key],value)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
            opt.step();loss_sum+=step_loss*len(selected);num+=len(selected)
            if start==0:print(json.dumps(dict(event='optimizer_step',epoch=epoch+1,step=1,loss=step_loss,gradnorm=float(norm))),flush=True)
        torch.cuda.synchronize();training_seconds=time.monotonic()-epoch_start
        payload=dict(model=model.adapted_state(),optimizer=opt.state_dict(),completed_epoch=epoch+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config)
        save_start=time.monotonic();saved=checkpoint(out/'last.pth',payload);save_seconds=time.monotonic()-save_start
        row=dict(epoch=epoch+1,train_loss=loss_sum/max(1,num),training_seconds=training_seconds,checkpoint_seconds=save_seconds,epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,optimizer_steps=math.ceil(len(ids)/config['effective_batch']),effective_batch=config['effective_batch'],microbatch=args.microbatch,peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        dump(out/'status.json',row);print(json.dumps(row),flush=True)
    save_start=time.monotonic()
    checkpoint(out/'final.pth',dict(model=model.adapted_state(),weights_sha256=WEIGHT_SHA,config=config))
    final_seconds=time.monotonic()-save_start
    if args.smoke:
        training_peak=row['peak_memory']
        assert frozen_hashes(model)==frozen_before,'a frozen encoder tensor changed'
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
        block12_changed=not torch.equal(model.encoder.blocks[12].attn.qkv.weight,block12_before)
        assert block12_changed
        changes={k:float((p-before[k]).abs().max()) for k,p in watched.items()}
        assert all(v>0 and math.isfinite(v) for v in changes.values())
        for h in handles:h.remove()
        model.eval();x,_,_=fold.batch(fit[:2],args.seed,0,False);x=x.cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x).float().cpu()
        reload_start=time.monotonic()
        fresh=Proposal(args.weights,True,args.arm).cuda().eval()
        for name in ('last.pth','final.pth'):
            loaded=torch.load(out/name,map_location='cpu',weights_only=False);validate_config(loaded['config'],config)
            load_adapted(fresh,loaded['model'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=fresh(x).float().cpu()
            assert torch.equal(prediction,actual),name
            del loaded
        torch.cuda.synchronize();reload_seconds=time.monotonic()-reload_start
        estimate=130*training_seconds/len(fit)*2134+130*save_seconds+final_seconds+1800
        report=dict(status='MEASURED',estimated_seconds_130=estimate,safety_seconds=1.5*estimate,training_images=len(fit),training_seconds=training_seconds,checkpoint_seconds=save_seconds,final_checkpoint_seconds=final_seconds,changes=changes,gradients=gradients,precision_trace=dict(enabled=True,amp_dtype='torch.bfloat16',layers=precision,logits_dtype='torch.float32'),reload_equal=True,final_reload_equal=True,reload_seconds=reload_seconds,training_peak_memory=training_peak,peak_memory=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
        report.update(observed_input_size=model.observed_input_size,observed_token_grids=model.observed_token_grids,effective_batch_lengths=batch_lengths,effective_object_counts=object_counts,smoke_fit=fit,all_frozen_encoder_tensors_unchanged=True,frozen_encoder_tensors=len(frozen_before),exact_optimizer_membership=True,trainable_encoder_names=encoder_names,block12_changed=block12_changed)
        try:validate_config(config,run_config('A' if args.arm=='C' else 'C',args.seed,args.microbatch))
        except ValueError:report['cross_arm_config_rejected']=True
        else:raise AssertionError('wrong arm config accepted')
        dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<run_budget_hours(args.arm)*3600,estimate
        write_examples(model,fold,fit,out/'point_examples_smoke','fit_1epoch_preliminary',2)
        report['status']='PASS';dump(out/'smoke.json',report);return
    dump(out/'fit.json',evaluate(model,fold,fit))
    rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    write_examples(model,fold,audit,out/'point_examples_audit','audit_final130',8)
    write_examples(model,fold,fit,out/'point_examples_fit','fit_final130',4)
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()
