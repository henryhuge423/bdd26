"""P17: convolutional multiscale representation, fold1 only."""
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
from train_proposal_p3 import (Proposal as OriginalProposal,PointHead as OriginalPointHead,Fold as OriginalFold,WEIGHT_SHA,DATA_SHA,
    sha,dump,checkpoint,evaluate,load_adapted)
from p5_contracts import validate_cohort,TISSUE_SHA,verify_hashes


class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return torch.nn.Sequential(torch.nn.Conv2d(a,b,3,padding=1),torch.nn.GroupNorm(8,b),torch.nn.ReLU(),torch.nn.Conv2d(b,b,3,padding=1),torch.nn.GroupNorm(8,b),torch.nn.ReLU())
        self.blocks=torch.nn.ModuleList([block(3,32),block(32,64),block(64,128)])
    def forward(self,x):
        a=self.blocks[0](x)
        b=self.blocks[1](torch.nn.functional.avg_pool2d(a,2))
        c=self.blocks[2](torch.nn.functional.avg_pool2d(b,2))
        resize=lambda t:torch.nn.functional.interpolate(t,size=c.shape[-2:],mode='bilinear',align_corners=False)
        return torch.cat([c,resize(b),resize(a)],1)

class ConvHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=torch.nn.Sequential(torch.nn.Conv2d(224,64,3,padding=1),torch.nn.GroupNorm(8,64),torch.nn.ReLU(),torch.nn.Conv2d(64,32,3,padding=1),torch.nn.GroupNorm(8,32),torch.nn.ReLU())
        self.out=torch.nn.Conv2d(32,1,1)
        torch.nn.init.zeros_(self.out.weight);torch.nn.init.constant_(self.out.bias,math.log(.01/.99))
    def forward(self,x):
        return torch.nn.functional.interpolate(self.out(self.fusion(x)).float(),size=(256,256),mode='bilinear',align_corners=False)[:,0]

class Proposal(OriginalProposal):
    def __init__(self,weights,trainable=True,arm='C'):
        if trainable is not True or arm not in ('C','K'):raise ValueError('P17 requires C/K adaptation')
        if arm=='C':super().__init__(weights,True)
        else:
            torch.nn.Module.__init__(self)
            self.encoder=ConvEncoder();self.head=ConvHead()
        self.arm=arm
    def forward(self,x):
        if self.arm=='C':return super().forward(x)
        if x.ndim!=4 or x.shape[1:]!=(3,256,256):raise ValueError('K requires native RGB256')
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        return self.head(self.encoder((x-mean)/std))
    def train(self,mode=True):
        if self.arm=='C':return super().train(mode)
        torch.nn.Module.train(self,mode);return self

class Fold(OriginalFold):
    def __init__(self,root,fold,arm):
        if fold!=1 or arm not in ('C','K'):raise ValueError('P17 is fold1 C/K only')
        super().__init__(root,fold)

def loss_for_microbatch(logits,target,total_count,arm):
    if (arm not in ('C','K') or logits.ndim!=3 or logits.shape!=target.shape
        or not len(logits) or not isinstance(total_count,Integral)
        or isinstance(total_count,(bool,np.bool_)) or total_count<0):
        raise ValueError('invalid P17 loss metadata or shapes')
    if not torch.isfinite(target).all() or (target<0).any() or (target>1).any():
        raise ValueError('P17 requires finite targets in[0,1]')
    return focal_numerator(logits,target)/max(1,total_count)


def encoder_lr(arm):
    if arm not in ('C','K'):raise ValueError('invalid P17 arm')
    return 1e-5 if arm=='C' else 3e-4

def make_optimizer(model,arm):
    lr=encoder_lr(arm)
    groups=[dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=lr,initial_lr=lr)]
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('optimizer parameter membership mismatch')
    return opt

def schedule_epoch(opt,epoch):
    if type(epoch) is not int or not 0<=epoch<130:raise ValueError('invalid P17 epoch')
    factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
    for group in opt.param_groups:group['lr']=group['initial_lr']*factor

def run_config(arm,seed):
    if arm not in ('C','K') or type(seed) is not int or seed not in (19,1):raise ValueError('invalid P17 configuration')
    return dict(phase='A',arm=arm,seed=seed,target_sigma=1,target_radius=3,
        objective=dict(kind='original_center_focal',denominator='effective_batch_objects'),
        architecture=('original_UNI_last12' if arm=='C' else 'scratch_conv_GN32_64_128_fusion224_64_32'),
        augmentation=dict(kind='original'),trainable_encoder_blocks=(list(range(12,24)) if arm=='C' else [0,1,2]),
        optimizer=dict(kind='AdamW',head_lr=3e-4,encoder_lr=encoder_lr(arm),betas=[.9,.999],weight_decay=.01,
            schedule='original_cosine_epoch0_to129_factor1_to0.1',clip_norm=1,effective_batch=16,microbatch=4))

def frozen_hashes(model):
    import hashlib
    # Every frozen parameter and buffer is included, not only sampled blocks.
    trainable={n for n,p in model.encoder.named_parameters() if p.requires_grad}
    return {n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for n,t in model.encoder.state_dict().items() if n not in trainable}


def validate_config(saved,expected):
    if saved!=expected:raise ValueError('checkpoint P17 configuration mismatch')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['A'],default='A')
    p.add_argument('--arm',choices=['C','K'],required=True)
    p.add_argument('--seed',type=int,choices=[19,1],default=19)
    p.add_argument('--smoke',action='store_true')
    return p.parse_args(argv)


def main():
    args=parse_args();config=run_config(args.arm,args.seed);out=Path(args.out)
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
    if args.smoke:manifest['smoke_fit']=list(fit[:256])
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
    opt=make_optimizer(model,args.arm)
    optimizer_ids={id(p) for g in opt.param_groups for p in g['params']}
    assert optimizer_ids=={id(p) for p in model.parameters() if p.requires_grad}
    encoder_watch=(model.encoder.blocks[23].attn.qkv if args.arm=='C' else model.encoder.blocks[2][0])
    watched=dict(encoder=encoder_watch.weight,output=model.head.out.weight)
    gradients={k:0. for k in watched};precision={};handles=[]
    if args.smoke:
        fit=manifest['smoke_fit'];epochs=1
        before={k:v.detach().clone() for k,v in watched.items()}
        frozen_before=frozen_hashes(model)
        block12_parameter=(model.encoder.blocks[12].attn.qkv.weight if args.arm=='C' else model.encoder.blocks[0][0].weight)
        block12_before=block12_parameter.detach().clone()
        for key,module in [('encoder',encoder_watch),('output',model.head.out)]:
            handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    else:epochs=130
    for epoch in range(epochs):
        if time.monotonic()-started>12*3600:raise TimeoutError('P17 run budget exhausted')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();schedule_epoch(opt,epoch)
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(fit).tolist()
        loss_sum=0.;num=0;torch.cuda.synchronize();epoch_start=time.monotonic()
        for start in range(0,len(ids),16):
            selected=ids[start:start+16];total_count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
            step_loss=0.
            for begin in range(0,len(selected),4):
                batch=selected[begin:begin+4];x,y,_=fold.batch(batch,args.seed,epoch,True);x=x.cuda();y=y.cuda()
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
        row=dict(epoch=epoch+1,train_loss=loss_sum/max(1,num),training_seconds=training_seconds,checkpoint_seconds=save_seconds,epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        dump(out/'status.json',row);print(json.dumps(row),flush=True)
    save_start=time.monotonic()
    checkpoint(out/'final.pth',dict(model=model.adapted_state(),weights_sha256=WEIGHT_SHA,config=config))
    final_seconds=time.monotonic()-save_start
    if args.smoke:
        training_peak=row['peak_memory']
        assert frozen_hashes(model)==frozen_before,'a frozen encoder tensor changed'
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
        block12_changed=not torch.equal(block12_parameter,block12_before)
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
        report.update(optimizer_initial_lrs=[g['initial_lr'] for g in opt.param_groups],smoke_fit=fit,all_frozen_encoder_tensors_unchanged=True,frozen_encoder_tensors=len(frozen_before),exact_optimizer_membership=True,trainable_encoder_names=encoder_names,block12_changed=block12_changed)
        try:validate_config(config,run_config('K' if args.arm=='C' else 'C',args.seed))
        except ValueError:report['cross_arm_config_rejected']=True
        else:raise AssertionError('wrong arm config accepted')
        dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<43200,estimate
        report['status']='PASS';dump(out/'smoke.json',report);return
    dump(out/'fit.json',evaluate(model,fold,fit))
    rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()
