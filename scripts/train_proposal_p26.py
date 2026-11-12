"""P26: original last12 control versus fixed pretrained-weight anchoring."""
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
    sha,dump,checkpoint,evaluate,load_adapted,configure_encoder)
from p5_contracts import validate_cohort,TISSUE_SHA,verify_hashes
from train_proposal_p19 import effective_batches,frozen_hashes
from train_proposal_p21 import match_points,select_examples,render_example

ARMS=('C','R')
DEPTH={'C':list(range(12,24)),'R':list(range(12,24))}

def configure_encoder_depth(enc, arm):
    if arm not in ARMS: raise ValueError('invalid P26 arm')
    configure_encoder(enc, True)

def smoke_ids(fit,gt):
    first=list(fit[:118]);later=[i for i in fit[118:] if any(g[0]==4 for g in gt[i])][:16]
    if len(first)!=118 or len(later)!=16 or len(set(first+later))!=134:
        raise ValueError('P26 needs distinct fixed134 fit smoke images')
    return first+later

class Proposal(OriginalProposal):
    def __init__(self,weights,trainable=True,arm='C'):
        if trainable is not True or arm not in ARMS:raise ValueError('P26 requires C/R')
        super().__init__(weights,True);self.arm=arm
        configure_encoder_depth(self.encoder,arm)
        self.observed_input_size=256;self.observed_token_grids=[16]*4

    def forward(self,x):
        if x.ndim!=4 or x.shape[0]<1 or tuple(x.shape[1:])!=(3,256,256):
            raise ValueError('P26 requires nonempty nativeRGB256 input')
        return super().forward(x)

class Fold(OriginalFold):
    def __init__(self,root,fold,arm):
        if fold!=1 or arm not in ARMS:raise ValueError('P26 is fold1 C/R only')
        super().__init__(root,fold);self.arm=arm

def loss_for_microbatch(logits,target,total_count,arm):
    if (arm not in ARMS or not torch.is_tensor(logits) or logits.ndim!=3 or not len(logits) or tuple(target.shape)!=tuple(logits.shape)
        or not isinstance(total_count,Integral) or isinstance(total_count,(bool,np.bool_)) or total_count<0):
        raise ValueError('invalid P26 loss metadata')
    return focal_numerator(logits,target)/max(1,total_count)

def make_optimizer(model, arm):
    if arm not in ARMS: raise ValueError('invalid P26 arm')
    import train_proposal_p25 as p25
    return p25.make_optimizer(model, 'C')

def schedule_epoch(opt, epoch):
    import train_proposal_p25 as p25
    return p25.schedule_epoch(opt, epoch)

def run_config(arm, seed, microbatch=4):
    if arm not in ARMS: raise ValueError('invalid P26 arm')
    import train_proposal_p25 as p25
    config = p25.run_config('C', seed, microbatch)
    config.update(arm=arm, anchor_lambda=.01 if arm == 'R' else 0.,
        anchor_scope='trainable_encoder' if arm == 'R' else 'none', anchor_weights_sha256=WEIGHT_SHA)
    return config

def snapshot_anchor(model):
    anchor = {n: p.detach().clone() for n,p in model.encoder.named_parameters() if p.requires_grad}
    if not anchor: raise ValueError('empty encoder anchor')
    return anchor

def anchor_penalty(model, anchor):
    params = {n:p for n,p in model.encoder.named_parameters() if p.requires_grad}
    if not params or params.keys() != anchor.keys(): raise ValueError('encoder anchor membership mismatch')
    return .005 * sum((p - anchor[n]).square().sum() for n,p in params.items())

def backward_effective_batch(model, anchor, losses):
    focal = 0.
    for loss in losses:
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite focal loss')
        loss.backward()
        focal += float(loss.detach())
    reg = 0.
    if anchor is not None:
        penalty = anchor_penalty(model, anchor)
        if not torch.isfinite(penalty): raise FloatingPointError('nonfinite anchor loss')
        penalty.backward()
        reg = float(penalty.detach())
    return focal, reg

def accumulate_epoch_loss(totals, focal, reg, batch_size):
    totals['focal'] += focal * batch_size
    totals['reg'] += reg * batch_size
    totals['total'] += (focal + reg) * batch_size
    totals['images'] += batch_size

def prepare_output(out):
    Path(out).mkdir(parents=True, exist_ok=False)

def final_payload(model, config):
    return dict(model=model.adapted_state(), weights_sha256=WEIGHT_SHA, config=config)

@torch.no_grad()
def block_drift(model, anchor):
    if anchor is None: return {}
    sums = {}
    for name, param in model.encoder.named_parameters():
        if name in anchor:
            block = name.split('.')[1]
            sums[block] = sums.get(block, 0.) + float((param-anchor[name]).square().sum())
    return {k: math.sqrt(v) for k,v in sums.items()}

def run_budget_hours(arm):
    if arm not in ARMS:raise ValueError('invalid P26 resource arm')
    return 12

def validate_smoke_memory(peak_bytes):
    if not math.isfinite(peak_bytes) or not 0<peak_bytes<=24*1024**3:
        raise ValueError('P26 training peak exceeds24GiB or is invalid')
    return True

def validate_config(saved,expected):
    if saved!=expected:raise ValueError('checkpoint P26 configuration mismatch')

@torch.no_grad()
def write_examples(model,fold,ids,out,scope,limit,experiment='P26'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);model.eval();records=[]
    chosen=select_examples(ids,fold.gt,limit)
    for i in chosen:
        x=torch.from_numpy(fold.images[i].copy()).permute(2,0,1).float()[None].cuda()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):prob=model(x).float().sigmoid()[0].cpu().numpy()
        points=decode_peaks(prob);gt=fold.gt[i];pairs=match_points(points,gt)
        row=score_points(points,gt);assert row['matched']==len(pairs)
        name=f'{scope}_{i:04d}.png'
        render_example(fold.images[i],gt,points,out/name,f'{experiment} {model.arm} {scope} image{i} tissue{fold.tissue[i]}')
        records.append(dict(image=int(i),scope=scope,tissue=str(fold.tissue[i]),gt=gt,points=points,unique_pairs=pairs,score=row,png=name))
    dump(out/'coordinates.json',dict(selection='GT-only deterministic strata',scope=scope,records=records,main_map_only=True))
    return records

def example_manifest(fit,audit,gt):
    import hashlib
    ids=dict(fit=select_examples(fit,gt,4),audit=select_examples(audit,gt,8))
    encoded=json.dumps(ids,sort_keys=True,separators=(',',':')).encode()
    return dict(**ids,sha256=hashlib.sha256(encoded).hexdigest(),selection='GT-only deterministic strata')

def parse_args(argv=None, *, arms=ARMS):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['A'],default='A')
    p.add_argument('--arm',choices=list(arms),required=True)
    p.add_argument('--seed',type=int,choices=[19,1],default=19)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--microbatch',type=int,choices=[4],default=4)
    return p.parse_args(argv)

def main():
    args = parse_args()
    train(args, run_config(args.arm, args.seed, args.microbatch), Proposal, args.arm)


def train(args, config, proposal_class, objective_arm, *, experiment='P26'):
    out = Path(args.out)
    prepare_output(out)
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
    fold=Fold(args.data,1,objective_arm)
    fit,audit=split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    manifest.update(fit=fit,audit=audit,cohort_sha256=validate_cohort(fit,audit))
    assert len(fit)==2134 and len(audit)==522
    manifest['fixed_examples']=example_manifest(fit,audit,fold.gt)
    counts=np.asarray([len(fold.gt[i]) for i in fit])
    manifest['fit_count_summary']=dict(images=len(fit),objects=int(counts.sum()),empty_images=int((counts==0).sum()),count_percentiles=np.percentile(counts,[0,25,50,75,100]).tolist())
    if args.smoke:manifest['smoke_fit']=smoke_ids(fit,fold.gt)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.85)
    model=proposal_class(args.weights,True,args.arm).cuda()
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src'),source.__file__
    encoder_names=[n for n,p in model.encoder.named_parameters() if p.requires_grad]
    trained_blocks=sorted({int(n.split('.')[1]) for n in encoder_names if n.startswith('blocks.')})
    assert trained_blocks==config['trainable_encoder_blocks'] and all(n.startswith('blocks.') for n in encoder_names)
    manifest.update(source=str(Path(source.__file__).resolve()),torch=torch.__version__,numpy=np.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in model.head.parameters()),trainable_encoder_names=encoder_names,trained_blocks=trained_blocks)
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.23.5'
    dump(out/'manifest.json',manifest)
    anchor=snapshot_anchor(model) if objective_arm=='R' else None
    opt=make_optimizer(model,objective_arm)
    optimizer_ids={id(p) for g in opt.param_groups for p in g['params']}
    assert optimizer_ids=={id(p) for p in model.parameters() if p.requires_grad}
    first_block, last_block = trained_blocks[0], trained_blocks[-1]
    watched = dict(encoder=model.encoder.blocks[last_block].attn.qkv.weight,
                   output=model.head.out.weight)
    watched[f'block{first_block}'] = model.encoder.blocks[first_block].attn.qkv.weight
    gradients={k:0. for k in watched};precision={};handles=[]
    if args.smoke:
        fit=manifest['smoke_fit'];epochs=1
        before={k:v.detach().clone() for k,v in watched.items()}
        frozen_before=frozen_hashes(model)
        block12_before=model.encoder.blocks[12].attn.qkv.weight.detach().clone()
        block0_before=model.encoder.blocks[0].attn.qkv.weight.detach().clone()
        for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out)]:
            handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    else:epochs=130
    for epoch in range(epochs):
        if time.monotonic()-started>run_budget_hours(objective_arm)*3600:raise TimeoutError('P26 run budget exhausted')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();schedule_epoch(opt,epoch)
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(fit).tolist()
        batch_lengths=[];object_counts=[];totals=dict(focal=0.,reg=0.,total=0.,images=0);torch.cuda.synchronize();epoch_start=time.monotonic()
        for step_index,selected in enumerate(effective_batches(ids,config['effective_batch'])):
            start=step_index*config['effective_batch'];total_count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
            batch_lengths.append(len(selected));object_counts.append(total_count)
            def losses():
                for begin in range(0,len(selected),args.microbatch):
                    batch=selected[begin:begin+args.microbatch];x,y,_=fold.batch(batch,args.seed,epoch,True);x=x.cuda();y=y.cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(x)
                        if args.smoke:
                            assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
                            assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
                            assert logits.dtype==torch.float32
                    yield loss_for_microbatch(logits,y,total_count,objective_arm)
            focal,reg=backward_effective_batch(model,anchor,losses())
            step_loss=focal+reg

            if args.smoke and start>=16:
                for key,p in watched.items():
                    value=float(p.grad.norm());assert math.isfinite(value) and value>0,(key,value)
                    gradients[key]=max(gradients[key],value)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
            opt.step();accumulate_epoch_loss(totals,focal,reg,len(selected))
            if start==0:print(json.dumps(dict(event='optimizer_step',epoch=epoch+1,step=1,loss=step_loss,gradnorm=float(norm))),flush=True)
        drift=block_drift(model,anchor)
        torch.cuda.synchronize();training_seconds=time.monotonic()-epoch_start
        payload=dict(model=model.adapted_state(),optimizer=opt.state_dict(),completed_epoch=epoch+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config)
        save_start=time.monotonic();saved=checkpoint(out/'last.pth',payload);save_seconds=time.monotonic()-save_start
        row=dict(epoch=epoch+1,train_loss=totals['total']/max(1,totals['images']),focal_loss=totals['focal']/max(1,totals['images']),regularizer_loss=totals['reg']/max(1,totals['images']),block_drift=drift,training_seconds=training_seconds,checkpoint_seconds=save_seconds,epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,optimizer_steps=math.ceil(len(ids)/config['effective_batch']),effective_batch=config['effective_batch'],microbatch=args.microbatch,peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        dump(out/'status.json',row);print(json.dumps(row),flush=True)
    save_start=time.monotonic()
    checkpoint(out/'final.pth',final_payload(model,config))
    final_seconds=time.monotonic()-save_start
    if args.smoke:
        training_peak=row['peak_memory']
        validate_smoke_memory(training_peak)
        assert frozen_hashes(model)==frozen_before,'a frozen encoder tensor changed'
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
        block12_changed=not torch.equal(model.encoder.blocks[12].attn.qkv.weight,block12_before);assert block12_changed == (12 in trained_blocks)
        block0_changed=not torch.equal(model.encoder.blocks[0].attn.qkv.weight,block0_before)
        assert block0_changed == (0 in trained_blocks), (args.arm, block0_changed)
        changes={k:float((p-before[k]).abs().max()) for k,p in watched.items()}
        assert all(v>0 and math.isfinite(v) for v in changes.values())
        for h in handles:h.remove()
        model.eval();x,_,_=fold.batch(fit[:2],args.seed,0,False);x=x.cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x).float().cpu()
        reload_start=time.monotonic()
        fresh=proposal_class(args.weights,True,args.arm).cuda().eval()
        for name in ('last.pth','final.pth'):
            loaded=torch.load(out/name,map_location='cpu',weights_only=False);validate_config(loaded['config'],config)
            load_adapted(fresh,loaded['model'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=fresh(x).float().cpu()
            assert torch.equal(prediction,actual),name
            del loaded
        torch.cuda.synchronize();reload_seconds=time.monotonic()-reload_start
        estimate=130*training_seconds/len(fit)*2134+130*save_seconds+final_seconds+1800
        report=dict(status='MEASURED',estimated_seconds_130=estimate,safety_seconds=1.5*estimate,budget_hours=run_budget_hours(objective_arm),training_images=len(fit),training_seconds=training_seconds,checkpoint_seconds=save_seconds,final_checkpoint_seconds=final_seconds,checkpoint_bytes=saved,changes=changes,gradients=gradients,precision_trace=dict(enabled=True,amp_dtype='torch.bfloat16',layers=precision,logits_dtype='torch.float32'),reload_equal=True,final_reload_equal=True,reload_seconds=reload_seconds,training_peak_memory=training_peak,peak_memory=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
        report.update(optimizer_initial_lrs=[g['initial_lr'] for g in opt.param_groups],observed_input_size=model.observed_input_size,observed_token_grids=model.observed_token_grids,effective_batch_lengths=batch_lengths,effective_object_counts=object_counts,smoke_fit=fit,all_frozen_encoder_tensors_unchanged=True,frozen_encoder_tensors=len(frozen_before),exact_optimizer_membership=True,trainable_encoder_names=encoder_names,trained_blocks=trained_blocks,block12_changed=block12_changed,block0_changed=block0_changed)
        try:validate_config(dict(config, arm='different'), config)
        except ValueError:report['cross_arm_config_rejected']=True
        else:raise AssertionError('wrong arm config accepted')
        dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<run_budget_hours(objective_arm)*3600,estimate
        write_examples(model,fold,fit,out/'point_examples_smoke','fit_1epoch_preliminary',2,experiment=experiment)
        report['status']='PASS';dump(out/'smoke.json',report);return
    dump(out/'fit.json',evaluate(model,fold,fit))
    rows=evaluate(model,fold,audit)
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    write_examples(model,fold,audit,out/'point_examples_audit','audit_final130',8,experiment=experiment)
    write_examples(model,fold,fit,out/'point_examples_fit','fit_final130',4,experiment=experiment)
    for scope in ('fit','audit'):
        saved_examples=json.loads((out/f'point_examples_{scope}'/'coordinates.json').read_text())
        assert [r['image'] for r in saved_examples['records']]==manifest['fixed_examples'][scope]
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)


if __name__=='__main__':main()

