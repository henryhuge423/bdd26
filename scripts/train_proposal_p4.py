"""P4: fixed auxiliary-foreground supervision study, fold1 only."""
import argparse
import hashlib
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

WEIGHT_SHA='56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40'
DATA_SHA={
 'fold1/images.npy':'b4f24c4d117a44fa930120d33b52175dc8189afdceb89a2cdf91d2682293105b',
 'fold1/inst.npy':'59fb7dc60354e54e642f41072595ee3a4bec700c0b36bd2c94075ae030e5de7e',
 'fold1/type.npy':'cda88107c0860fd3be6c58d9dafde7623b9709ee639649246e67c4844ae593c0',
 'fold2/images.npy':'d7c93f1734d218e1313d276ab2c3c02130289e3076108f724324d9f71860aa39',
 'fold2/inst.npy':'145420bb21d05d5047ecc6d0d043ef05d6a2789c608fe987488480d16f7b7c21',
 'fold2/type.npy':'82d09f44f79575fd8629b02495c25dca2ddce12b49c25191e3d084f473f57077'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)

def configure_encoder(enc,trainable):
    enc.requires_grad_(False)
    if trainable:
        for b in enc.blocks[12:]: b.requires_grad_(True)
    enc.eval()

def load_adapted(model,state):
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if set(state)!=expected: raise ValueError('checkpoint trainable key mismatch')
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or set(result.missing_keys)!=set(model.state_dict())-expected:
        raise ValueError('checkpoint frozen key mismatch')

def balanced_foreground_bce(logits, target):
    """Per-image equally weighted foreground/background BCE; absent groups give zero."""
    logits=logits.float(); target=target.float()
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    axes=tuple(range(1,loss.ndim))
    fg=(target>0).float();bg=1-fg
    return .5*((loss*fg).sum(axes)/fg.sum(axes).clamp_min(1)+(loss*bg).sum(axes)/bg.sum(axes).clamp_min(1))

def augment_triplet(image,center,foreground,rng,photometric=True):
    image,targets=augment_pair(image,torch.stack([center,foreground]),rng,photometric)
    return image,targets[0],targets[1]

def clip_separate(main_parameters,aux_parameters):
    norm=torch.nn.utils.clip_grad_norm_(main_parameters,1,error_if_nonfinite=True)
    if aux_parameters: torch.nn.utils.clip_grad_norm_(aux_parameters,1,error_if_nonfinite=True)
    return norm

class PointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.ModuleList([nn.Conv2d(1024,192,1) for _ in range(4)])
        self.stem=nn.Conv2d(768,256,3,padding=1)
        self.up=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in ((256,192),(192,128),(128,64))])
        self.out=nn.Conv2d(64,1,1)
        nn.init.zeros_(self.out.weight);nn.init.constant_(self.out.bias,math.log(.01/.99))
    def forward_features(self,feats):
        x=torch.cat([p(f) for p,f in zip(self.proj,feats)],1);x=F.relu(self.stem(x))
        for layer in self.up: x=F.relu(layer(F.interpolate(x,scale_factor=2,mode='nearest')))
        return x
    def forward(self,feats):
        return F.interpolate(self.out(self.forward_features(feats)),size=(256,256),mode='bilinear',align_corners=False)[:,0]

class Proposal(nn.Module):
    def __init__(self,weights,trainable):
        super().__init__()
        from nucseg.cellvit.model import UNIEncoder
        self.encoder=UNIEncoder(drop_path_rate=0,attn_drop_rate=0)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        configure_encoder(self.encoder,trainable)
        self.head=PointHead()
        self.aux=nn.Conv2d(64,1,1)
        nn.init.zeros_(self.aux.weight);nn.init.zeros_(self.aux.bias)
        self.aux.requires_grad_(trainable)
    def forward(self,x,auxiliary=False):
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        _,tokens=self.encoder((x-mean)/std)
        feats=[]
        for t in tokens:
            t=F.layer_norm(t[:,1:],(1024,))
            feats.append(t.transpose(1,2).reshape(len(t),1024,16,16))
        feature=self.head.forward_features(feats)
        center=F.interpolate(self.head.out(feature),size=(256,256),mode='bilinear',align_corners=False)[:,0]
        if not auxiliary: return center
        foreground=F.interpolate(self.aux(feature),size=(256,256),mode='bilinear',align_corners=False)[:,0]
        return center,foreground
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def adapted_state(self):
        train_names={n for n,p in self.named_parameters() if p.requires_grad}
        return {n:t for n,t in self.state_dict().items() if n in train_names}

class Fold:
    def __init__(self,root,fold):
        p=Path(root)/f'fold{fold}'
        self.images=np.load(p/'images.npy',mmap_mode='r')
        self.inst=np.load(p/'inst.npy',mmap_mode='r');typ=np.load(p/'type.npy',mmap_mode='r')
        self.tissue=np.load(p/'tissue.npy')
        self.gt=[gt_objects(a,b) for a,b in zip(self.inst,typ)]
    def batch(self,ids,seed,epoch,aug):
        images=[];targets=[];foregrounds=[];count=0
        for i in ids:
            im=torch.from_numpy(self.images[i].copy()).permute(2,0,1).float()/255
            target=torch.from_numpy(gaussian_target(self.gt[i]));count+=len(self.gt[i])
            foreground=torch.from_numpy((self.inst[i]>0).astype(np.float32))
            if aug: im,target,foreground=augment_triplet(im,target,foreground,np.random.default_rng(np.random.SeedSequence([seed,epoch,int(i),987])))
            images.append(im);targets.append(target);foregrounds.append(foreground)
        return torch.stack(images),torch.stack(targets),count,torch.stack(foregrounds)

def tensor_bytes(obj):
    if torch.is_tensor(obj): return obj.numel()*obj.element_size()
    if isinstance(obj,dict): return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj,(list,tuple)): return sum(tensor_bytes(v) for v in obj)
    return 0

def checkpoint(path,payload):
    path=Path(path)
    required=int(tensor_bytes(payload)*1.1)+(20<<30)
    if shutil.disk_usage(path.parent).free<required: raise RuntimeError('insufficient atomic checkpoint headroom')
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
    return path.stat().st_size

@torch.no_grad()
def evaluate(model,fold,ids):
    model.eval();rows=[]
    for start in range(0,len(ids),4):
        selected=ids[start:start+4]
        x=torch.stack([torch.from_numpy(fold.images[i].copy()).permute(2,0,1).float()/255 for i in selected]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): p=model(x).float().sigmoid().cpu().numpy()
        for i,prob in zip(selected,p):
            points=decode_peaks(prob);row=score_points(points,fold.gt[i]);row.update(image=int(i),tissue=str(fold.tissue[i]),inside=sum(fold.inst[i][y,x]>0 for y,x in points))
            row['inside']=int(row['inside']);row['saturated_512']=len(points)==512;rows.append(row)
    return rows

def main():
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--weights',required=True);a.add_argument('--out',required=True)
    a.add_argument('--phase',choices=['A'],default='A');a.add_argument('--arm',choices=['C','N'],default='N');a.add_argument('--seed',type=int,choices=[19,1],default=19)
    a.add_argument('--prepare',action='store_true');a.add_argument('--smoke',action='store_true');a.add_argument('--resume',action='store_true')
    args=a.parse_args();out=Path(args.out)
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic();manifest=dict(config=vars(args),weights_sha256=sha(args.weights),input_sha256={})
    assert manifest['weights_sha256']==WEIGHT_SHA
    for name,expected in DATA_SHA.items():
        if name.startswith('fold2') and args.phase=='A': continue
        actual=sha(Path(args.data)/name);assert actual==expected,(name,actual)
        manifest['input_sha256'][name]=actual
    for fold in (1,2) if args.phase=='B' else (1,):
        manifest['input_sha256'][f'fold{fold}/tissue.npy']=sha(Path(args.data)/f'fold{fold}/tissue.npy')
    fold=Fold(args.data,1)
    fit,audit=split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    if args.phase=='B': fit=list(range(len(fold.images)))
    manifest.update(fit=fit,audit=audit,audit_dead_images=sum(any(g[0]==4 and g[2] for g in fold.gt[i]) for i in audit),audit_dead_objects=sum(sum(g[0]==4 and g[2] for g in fold.gt[i]) for i in audit))
    assert manifest['audit_dead_images']>=10 and manifest['audit_dead_objects']>=100
    if args.prepare:
        dump(out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ('fit','audit')}));return
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.85)
    trainable=True;aug=True
    model=Proposal(args.weights,trainable).cuda()
    model.aux.requires_grad_(args.arm=='N')
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src'), source.__file__
    manifest.update(source=str(Path(source.__file__).resolve()),torch=torch.__version__,numpy=np.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in model.head.parameters()))
    dump(out/'manifest.json',manifest)
    groups=[dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4)]
    if trainable: groups.append(dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5))
    if args.arm=='N': groups.append(dict(params=list(model.aux.parameters()),lr=3e-4,initial_lr=3e-4))
    main_parameters=[p for p in model.encoder.parameters() if p.requires_grad]+list(model.head.parameters())
    aux_parameters=list(model.aux.parameters()) if args.arm=='N' else []
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    first=0
    if args.resume:
        c=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        assert c['config']==dict(phase=args.phase,arm=args.arm,seed=args.seed)
        load_adapted(model,c['model'])
        opt.load_state_dict(c['optimizer']);first=c['completed_epoch'];torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    if args.smoke:
        fit=fit[:36];epochs=1
        watched=model.encoder.blocks[12].attn.qkv.weight
        before=watched.detach().clone();head_before=model.head.out.weight.detach().clone()
        aux_before=model.aux.weight.detach().clone()
    else: epochs=130
    deadline=(96 if args.phase=='A' else 48)*3600  # controller also enforces phase/global budget
    for epoch in range(first,epochs):
        if time.monotonic()-started>deadline: raise TimeoutError('run budget exhausted')
        if shutil.disk_usage(out).free<20<<30: raise RuntimeError('disk floor')
        model.train()
        factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
        for g in opt.param_groups: g['lr']=g['initial_lr']*factor
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(fit).tolist()
        loss_sum=0.;num=0;epoch_start=time.monotonic()
        for start in range(0,len(ids),16):
            selected=ids[start:start+16];den=max(1,sum(len(fold.gt[i]) for i in selected));opt.zero_grad(set_to_none=True)
            step_loss=0.
            for begin in range(0,len(selected),4):
                batch=selected[begin:begin+4];x,y,_,foreground=fold.batch(batch,args.seed,epoch,aug);x=x.cuda();y=y.cuda();foreground=foreground.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if args.arm=='N': logits,aux_logits=model(x,auxiliary=True)
                    else: logits=model(x)
                loss=focal_numerator(logits,y)/den
                if args.arm=='N': loss=loss+balanced_foreground_bce(aux_logits,foreground).sum()/len(selected)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward();step_loss+=float(loss.detach())
            norm=clip_separate(main_parameters,aux_parameters)
            opt.step();loss_sum+=step_loss*len(selected);num+=len(selected)
            if start==0: print(json.dumps(dict(event='optimizer_step',epoch=epoch+1,step=1,loss=step_loss,gradnorm=float(norm))),flush=True)
        training_seconds=time.monotonic()-epoch_start
        payload=dict(model=model.adapted_state(),optimizer=opt.state_dict(),completed_epoch=epoch+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=dict(phase=args.phase,arm=args.arm,seed=args.seed))
        saved=checkpoint(out/'last.pth',payload)
        row=dict(training_seconds=training_seconds,epoch=epoch+1,train_loss=loss_sum/max(1,num),epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        dump(out/'status.json',row);print(json.dumps(row),flush=True)
    if args.smoke:
        encoder_change=float((watched-before).abs().max());head_change=float((model.head.out.weight-head_before).abs().max())
        assert head_change>0 and (encoder_change>0 if trainable else encoder_change==0)
        aux_change=float((model.aux.weight-aux_before).abs().max())
        assert aux_change>0 if args.arm=='N' else aux_change==0
        model.train();x,_,_,foreground=fold.batch(fit[:2],args.seed,0,False);x=x.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): _,aux_logits=model(x,auxiliary=True)
        extra=balanced_foreground_bce(aux_logits,foreground.cuda()).mean()
        aux_shared_grad=float(torch.autograd.grad(extra,model.head.stem.weight)[0].norm())
        assert aux_shared_grad>0 if args.arm=='N' else aux_shared_grad==0
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before_reload,before_aux=model(x,auxiliary=True)
            before_reload=before_reload.float().cpu();before_aux=before_aux.float().cpu()
        reload=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        fresh=Proposal(args.weights,trainable).cuda().eval()
        fresh.aux.requires_grad_(args.arm=='N')
        load_adapted(fresh,reload['model'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            after_reload,after_aux=fresh(x,auxiliary=True)
            after_reload=after_reload.float().cpu();after_aux=after_aux.float().cpu()
        assert torch.equal(before_reload,after_reload) and torch.equal(before_aux,after_aux)
        assert torch.equal(model.aux.weight,fresh.aux.weight) and torch.equal(model.aux.bias,fresh.aux.bias)
        estimate_seconds=training_seconds*(2134/len(fit))*130+max(0,row['epoch_seconds']-training_seconds)*130+1800
        assert estimate_seconds*1.5<12*3600,estimate_seconds
        dump(out/'smoke.json',dict(estimated_seconds_130=estimate_seconds,aux_change=aux_change,aux_shared_grad=aux_shared_grad,aux_reload_equal=True,encoder_change=encoder_change,head_change=head_change,reload_equal=True,seconds=time.monotonic()-started,checkpoint_bytes=saved,peak_memory=torch.cuda.max_memory_allocated()))
        return
    checkpoint(out/'final.pth',dict(model=model.adapted_state(),weights_sha256=WEIGHT_SHA,config=dict(phase=args.phase,arm=args.arm,seed=args.seed)))
    dump(out/'fit.json',evaluate(model,fold,fit))
    if args.phase=='A': rows=evaluate(model,fold,audit)
    else:
        val=Fold(args.data,2);rows=evaluate(model,val,list(range(len(val.images))))
    dump(out/'evaluation.json',rows);dump(out/'summary.json',aggregate(rows))
    dump(out/'complete.json',dict(exit_code=0,epochs=130,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',summary=aggregate(rows))),flush=True)

if __name__=='__main__': main()
