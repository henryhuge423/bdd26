"""P5: matched-capacity native-grid refinement, frozen fold1 development study."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from proposal_p3_utils import (split_indices,gt_objects,gaussian_target,augment_pair,
    focal_numerator,decode_peaks,score_points,aggregate,admission)
from train_proposal_p3 import (WEIGHT_SHA,DATA_SHA,sha,dump,configure_encoder,
    load_adapted,Fold,checkpoint,evaluate,Proposal as BaseProposal)
from train_proposal_p4 import PointHead as BaseHead
from p5_contracts import (TISSUE_SHA,COHORT_SHA,validate_cohort,
    require_positive_gradients,verify_hashes)


class PointHead(BaseHead):
    def __init__(self,native):
        super().__init__()
        self.native=bool(native)
        self.refine=nn.Conv2d(64,64,3,padding=1)

    def forward(self,feats):
        x=self.forward_features(feats)
        if self.native:
            x=F.interpolate(x,size=(256,256),mode='bilinear',align_corners=False)
        x=self.out(F.relu(self.refine(x)))
        if not self.native:
            x=F.interpolate(x,size=(256,256),mode='bilinear',align_corners=False)
        return x[:,0]


class Proposal(BaseProposal):
    def __init__(self,weights,native):
        nn.Module.__init__(self)
        from nucseg.cellvit.model import UNIEncoder
        self.encoder=UNIEncoder(drop_path_rate=0,attn_drop_rate=0)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        configure_encoder(self.encoder,True)
        self.head=PointHead(native)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--arm',choices=['L','R'],required=True)
    parser.add_argument('--seed',type=int,choices=[19,1],required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=Path(args.out)
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic();config=dict(arm=args.arm,seed=args.seed,phase='A',study='P5')
    manifest=dict(config=vars(args),weights_sha256=sha(args.weights),input_sha256={})
    assert manifest['weights_sha256']==WEIGHT_SHA
    for name,expected in DATA_SHA.items():
        if not name.startswith('fold1/'):continue
        actual=sha(Path(args.data)/name);assert actual==expected,(name,actual)
        manifest['input_sha256'][name]=actual
    manifest['input_sha256']['fold1/tissue.npy']=sha(Path(args.data)/'fold1/tissue.npy')
    assert manifest['input_sha256']['fold1/tissue.npy']==TISSUE_SHA
    fold=Fold(args.data,1)
    fit,audit=split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    assert len(fit)==2134 and len(audit)==522
    manifest['cohort_sha256']=validate_cohort(fit,audit)
    manifest.update(fit=fit,audit=audit,audit_dead_images=sum(any(g[0]==4 and g[2] for g in fold.gt[i]) for i in audit),audit_dead_objects=sum(sum(g[0]==4 and g[2] for g in fold.gt[i]) for i in audit))
    assert manifest['audit_dead_images']==11 and manifest['audit_dead_objects']==145
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(.85)
    model=Proposal(args.weights,args.arm=='R').cuda()
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src')
    manifest.update(source=str(Path(source.__file__).resolve()),trainer_sha256=sha(__file__),torch=torch.__version__,numpy=np.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu_name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),precision='BF16 autocast forward / FP32 parameters and loss',trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in model.head.parameters()))
    if args.resume:
        old=json.loads((out/'manifest.json').read_text())
        assert old['trainer_sha256']==manifest['trainer_sha256'] and old['input_sha256']==manifest['input_sha256']
        assert old['fit']==fit and old['audit']==audit
    else:dump(out/'manifest.json',manifest)
    groups=[dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)]
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    first=0
    if args.resume:
        state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        assert state['config']==config
        load_adapted(model,state['model']);opt.load_state_dict(state['optimizer'])
        first=state['completed_epoch'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    if args.smoke:
        assert not args.resume
        fit=fit[:36];epochs=1
        watched=model.encoder.blocks[12].attn.qkv.weight
        before=watched.detach().clone();head_before=model.head.out.weight.detach().clone();refine_before=model.head.refine.weight.detach().clone()
    else:epochs=130
    for epoch in range(first,epochs):
        if time.monotonic()-started>12*3600:raise TimeoutError('run budget exhausted')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(fit).tolist()
        loss_sum=0.;num=0;torch.cuda.synchronize();epoch_start=time.monotonic()
        for start in range(0,len(ids),16):
            selected=ids[start:start+16];den=max(1,sum(len(fold.gt[i]) for i in selected));opt.zero_grad(set_to_none=True);step_loss=0.
            for begin in range(0,len(selected),4):
                batch=selected[begin:begin+4];x,y,_=fold.batch(batch,args.seed,epoch,True);x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
                assert logits.dtype==torch.bfloat16,logits.dtype
                loss=focal_numerator(logits,y)/den
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();step_loss+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
            opt.step();loss_sum+=step_loss*len(selected);num+=len(selected)
            if start==0:print(json.dumps(dict(event='optimizer_step',epoch=epoch+1,loss=step_loss,gradnorm=float(norm),logits_dtype=str(logits.dtype))),flush=True)
        torch.cuda.synchronize();training_seconds=time.monotonic()-epoch_start
        payload=dict(model=model.adapted_state(),optimizer=opt.state_dict(),completed_epoch=epoch+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config)
        save_start=time.monotonic();saved=checkpoint(out/'last.pth',payload);save_seconds=time.monotonic()-save_start
        row=dict(epoch=epoch+1,train_loss=loss_sum/max(1,num),training_seconds=training_seconds,checkpoint_seconds=save_seconds,epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        dump(out/'status.json',row);print(json.dumps(row),flush=True)
    final_start=time.monotonic()
    checkpoint(out/'final.pth',dict(model=model.adapted_state(),weights_sha256=WEIGHT_SHA,config=config))
    final_save_seconds=time.monotonic()-final_start
    if args.smoke:
        gradients=dict(encoder=float(watched.grad.float().norm()) if watched.grad is not None else 0.,refinement=float(model.head.refine.weight.grad.float().norm()) if model.head.refine.weight.grad is not None else 0.,output=float(model.head.out.weight.grad.float().norm()) if model.head.out.weight.grad is not None else 0.)
        require_positive_gradients(gradients)
        changes=dict(encoder=float((watched-before).abs().max()),output=float((model.head.out.weight-head_before).abs().max()),refinement=float((model.head.refine.weight-refine_before).abs().max()))
        assert all(v>0 for v in changes.values()),changes
        model.eval();x,_,_=fold.batch(fit[:2],args.seed,0,False);x=x.cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before_reload=model(x).float().cpu()
        state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        fresh=Proposal(args.weights,args.arm=='R').cuda().eval();load_adapted(fresh,state['model'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after_reload=fresh(x).float().cpu()
        assert torch.equal(before_reload,after_reload)
        fresh_final=Proposal(args.weights,args.arm=='R').cuda().eval()
        load_adapted(fresh_final,torch.load(out/'final.pth',map_location='cpu',weights_only=False)['model'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after_final=fresh_final(x).float().cpu()
        assert torch.equal(before_reload,after_final)
        estimate=130*training_seconds/len(fit)*2134+130*save_seconds+final_save_seconds+1800
        assert estimate*1.5<12*3600,estimate
        dump(out/'smoke.json',dict(changes=changes,gradients=gradients,reload_equal=True,final_reload_equal=True,estimated_seconds_130=estimate,safety_factor=1.5,logits_dtype=str(logits.dtype),training_seconds=training_seconds,measured_images=len(fit),last_checkpoint_seconds=save_seconds,final_checkpoint_seconds=final_save_seconds,peak_memory=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started))
        return
    dump(out/'fit.json',evaluate(model,fold,fit));rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()
