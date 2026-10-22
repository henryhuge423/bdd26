"""P9: fixed Dead-center positive focal weighting, fold1 only."""
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
from train_proposal_p3 import (Proposal,Fold as OriginalFold,WEIGHT_SHA,DATA_SHA,
    sha,dump,checkpoint,evaluate,load_adapted)
from p5_contracts import validate_cohort,TISSUE_SHA,verify_hashes


def positive_weights(gt,height=256,width=256):
    weights=torch.ones((height,width),dtype=torch.float32)
    for typ,(y,x),_,_ in gt:
        if not (0<=y<height and 0<=x<width):raise ValueError('center outside weight map')
        if typ==4:weights[y,x]=4
    return weights


def smoke_indices(fit,gt):
    base=list(fit[:240])
    dead=[int(i) for i in fit[240:] if any(g[0]==4 for g in gt[i])][:16]
    result=base+dead
    if len(base)!=240 or len(dead)!=16 or len(set(result))!=256:
        raise ValueError('smoke requires240fit images and16distinct subsequent Dead-positive fit images')
    return result


class Fold(OriginalFold):
    def __init__(self,root,fold,arm):
        if fold!=1 or arm not in ('C','D'):raise ValueError('P9 is fold1 C/D only')
        super().__init__(root,fold)
    def positive_weights(self,ids,seed,epoch,aug):
        result=[]
        for i in ids:
            weights=positive_weights(self.gt[i])
            if aug:
                rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,int(i),987]))
                _,weights=augment_pair(torch.zeros(1,1,1),weights,rng)
            result.append(weights)
        return torch.stack(result)


def loss_for_microbatch(logits,target,weights,total_count,arm):
    if (arm not in ('C','D') or logits.ndim!=3 or logits.shape!=target.shape
        or weights.shape!=logits.shape or not len(logits)
        or not isinstance(total_count,Integral) or isinstance(total_count,(bool,np.bool_))
        or total_count<0 or not torch.all((weights==1)|(weights==4))):
        raise ValueError('invalid P9 loss metadata or shapes')
    original=focal_numerator(logits,target)
    if arm=='C':return original/max(1,total_count)
    x=logits.float();p=x.sigmoid()
    extra=-(torch.nn.functional.logsigmoid(x)*(1-p).square()*(target==1)*(weights.float()-1)).sum()
    return (original+extra)/max(1,total_count)


def run_config(arm,seed):
    if arm not in ('C','D') or seed not in (19,1):raise ValueError('invalid P9 configuration')
    policy=dict(dead=1 if arm=='C' else 4,others=1,negative=1,denominator='original_effective_batch_instance_count',centers='original_gt_objects',collision='max',boundary_dead=True)
    return dict(phase='A',arm=arm,seed=seed,target_sigma=1,target_radius=3,augmentation=dict(kind='original'),positive_weight=policy)


def validate_config(saved,expected):
    if saved!=expected:raise ValueError('checkpoint P9 configuration mismatch')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['A'],default='A')
    p.add_argument('--arm',choices=['C','D'],required=True)
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
    if args.smoke:manifest['smoke_fit']=smoke_indices(fit,fold.gt)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.85)
    model=Proposal(args.weights,True).cuda()
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src'),source.__file__
    manifest.update(source=str(Path(source.__file__).resolve()),torch=torch.__version__,numpy=np.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in model.head.parameters()))
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.23.5'
    dump(out/'manifest.json',manifest)
    groups=[dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)]
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    watched=dict(encoder=model.encoder.blocks[12].attn.qkv.weight,output=model.head.out.weight)
    gradients={k:0. for k in watched};precision={};handles=[];weighted_positive_pixels=0
    if args.smoke:
        fit=manifest['smoke_fit'];epochs=1
        before={k:v.detach().clone() for k,v in watched.items()}
        for key,module in [('encoder',model.encoder.blocks[12].attn.qkv),('output',model.head.out)]:
            handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    else:epochs=130
    for epoch in range(epochs):
        if time.monotonic()-started>12*3600:raise TimeoutError('P9 run budget exhausted')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
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
                weights=(fold.positive_weights(batch,args.seed,epoch,True).cuda() if args.arm=='D' else torch.ones_like(y))
                if args.smoke and args.arm=='D':
                    assert torch.all(y[weights==4]==1),'Dead weight not on augmented true centers'
                    weighted_positive_pixels+=int(((weights==4)&(y==1)).sum())
                loss=loss_for_microbatch(logits,y,weights,total_count,args.arm)
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
        changes={k:float((p-before[k]).abs().max()) for k,p in watched.items()}
        assert all(v>0 and math.isfinite(v) for v in changes.values())
        for h in handles:h.remove()
        model.eval();x,_,_=fold.batch(fit[:2],args.seed,0,False);x=x.cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x).float().cpu()
        reload_start=time.monotonic()
        fresh=Proposal(args.weights,True).cuda().eval()
        for name in ('last.pth','final.pth'):
            loaded=torch.load(out/name,map_location='cpu',weights_only=False);validate_config(loaded['config'],config)
            load_adapted(fresh,loaded['model'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=fresh(x).float().cpu()
            assert torch.equal(prediction,actual),name
            del loaded
        torch.cuda.synchronize();reload_seconds=time.monotonic()-reload_start
        estimate=130*training_seconds/len(fit)*2134+130*save_seconds+final_seconds+1800
        report=dict(status='MEASURED',estimated_seconds_130=estimate,safety_seconds=1.5*estimate,training_images=len(fit),training_seconds=training_seconds,checkpoint_seconds=save_seconds,final_checkpoint_seconds=final_seconds,changes=changes,gradients=gradients,precision_trace=dict(enabled=True,amp_dtype='torch.bfloat16',layers=precision,logits_dtype='torch.float32'),reload_equal=True,final_reload_equal=True,reload_seconds=reload_seconds,training_peak_memory=training_peak,peak_memory=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
        report.update(smoke_fit=fit,weighted_positive_pixels=weighted_positive_pixels)
        if args.arm=='D':assert weighted_positive_pixels>0,'real weighted path was not exercised'
        dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<43200,estimate
        report['status']='PASS';dump(out/'smoke.json',report);return
    dump(out/'fit.json',evaluate(model,fold,fit))
    rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()
