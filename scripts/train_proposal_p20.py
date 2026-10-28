"""P20: encoder input256/512 with fixed native256 center targets and logits."""
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

class Proposal(OriginalProposal):
    def __init__(self,weights,trainable=True,arm='C'):
        if trainable is not True or arm not in ('C','U'):raise ValueError('P20 requires C/U')
        super().__init__(weights,True)
        self.arm=arm
        self.observed_input_size=None
        self.observed_token_grids=None

    def forward(self,x):
        if x.ndim!=4 or x.shape[0]<1 or tuple(x.shape[1:])!=(3,256,256):
            raise ValueError('P20 requires nonempty native RGB256 input')
        if self.arm=='C':
            self.observed_input_size=256;self.observed_token_grids=[16]*4
            return super().forward(x)
        if self.arm!='U':raise ValueError('invalid P20 arm')
        x=F.interpolate(x,size=(512,512),mode='bilinear',align_corners=False)
        self.observed_input_size=x.shape[-1]
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        _,tokens=self.encoder((x-mean)/std)
        if len(tokens)!=4 or any(tuple(t.shape[1:])!=(1025,1024) for t in tokens):
            raise ValueError('P20 U requires four native32 token grids plus CLS')
        self.observed_token_grids=[32]*4
        feats=[]
        for t in tokens:
            t=F.layer_norm(t[:,1:],(1024,))
            feats.append(t.transpose(1,2).reshape(len(t),1024,32,32))
        return self.head(feats)


class Fold(OriginalFold):
    def __init__(self,root,fold,arm):
        if fold!=1 or arm not in ('C','U'):raise ValueError('P20 is fold1 C/U only')
        super().__init__(root,fold)


def loss_for_microbatch(logits,target,total_count,arm):
    if (arm not in ('C','U') or logits.ndim!=3 or logits.shape!=target.shape
        or not len(logits) or not isinstance(total_count,Integral)
        or isinstance(total_count,(bool,np.bool_)) or total_count<0):
        raise ValueError('invalid P20 loss metadata or shapes')
    return focal_numerator(logits,target)/max(1,total_count)


def run_config(arm,seed,microbatch=4):
    if (arm not in ('C','U') or type(seed) is not int or seed not in (19,1)
        or type(microbatch) is not int or microbatch!=4):
        raise ValueError('invalid P20 configuration')
    return dict(phase='A',arm=arm,seed=seed,target_sigma=1,target_radius=3,
        augmentation=dict(kind='original'),trainable_encoder_blocks=list(range(12,24)),
        architecture='original_UNI_last12' if arm=='C' else 'UNI_last12_input512_native256',
        effective_batch=16,microbatch=4,input_size=256 if arm=='C' else 512,
        token_grid=16 if arm=='C' else 32,output_size=256,
        input_interpolation='none' if arm=='C' else 'bilinear_align_corners_false')


def validate_config(saved,expected):
    if saved!=expected:raise ValueError('checkpoint P20 configuration mismatch')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['A'],default='A')
    p.add_argument('--arm',choices=['C','U'],required=True)
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
    if args.smoke:manifest['smoke_fit']=list(fit[:134])
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
        if time.monotonic()-started>24*3600:raise TimeoutError('P20 run budget exhausted')
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
                    logits=model(x)
                    if args.smoke:
                        assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
                        assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
                        assert logits.dtype==torch.float32
                loss=loss_for_microbatch(logits,y,total_count,args.arm)
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
        try:validate_config(config,run_config('U' if args.arm=='C' else 'C',args.seed,args.microbatch))
        except ValueError:report['cross_arm_config_rejected']=True
        else:raise AssertionError('wrong arm config accepted')
        dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<86400,estimate
        report['status']='PASS';dump(out/'smoke.json',report);return
    dump(out/'fit.json',evaluate(model,fold,fit))
    rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()
