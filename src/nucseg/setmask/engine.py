"""Guarded, explicitly resumed E6-only training and final-checkpoint inference."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cellvit.data import cell_tissue_weights
from ..cellvit.engine import TrainConfig, cellvit_loss, to_nhwc, predict_fold, _forward_probs, _post
from ..cellvit.model import CellViTUNI, UNI_MEAN, UNI_STD
from ..data.pannuke import PanNukeFold
from .data import PairedDataset, sample_keys, stable_seed
from .decode import decode_set
from .matching import setmask_loss
from .model import SetMaskUNI
from .state import atomic_save, capture_rng, restore_rng

ROOT=Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SetMaskConfig:
    arm:str
    seed:int
    out_dir:str
    uni_ckpt:str='weights/MahmoodLab/UNI/pytorch_model.bin'
    workers:int=8
    round:str=field(default='E6',init=False)
    split:int=field(default=1,init=False)
    upscale:int=field(default=1,init=False)
    epochs:int=field(default=130,init=False)
    unfreeze_epoch:int=field(default=25,init=False)
    batch_size:int=field(default=16,init=False)
    lr:float=field(default=3e-4,init=False)
    weight_decay:float=field(default=1e-4,init=False)
    gamma:float=field(default=.85,init=False)
    sampling_gamma:float=field(default=.85,init=False)
    hv_min_size:int=field(default=30,init=False)

    def __post_init__(self):
        if self.arm not in ('B','Q') or self.seed not in (19,1) or self.workers<0:
            raise ValueError('Frozen E6 arms B/Q, seeds19/1 and nonnegative worker count required')
        object.__setattr__(self,'out_dir',str(self.out_dir))
        object.__setattr__(self,'uni_ckpt',str(self.uni_ckpt))


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def make_provenance(cfg):
    sources={str(p.relative_to(ROOT)):sha_file(p) for p in sorted((ROOT/'src/nucseg').rglob('*.py'))}
    sources.update({str(p.relative_to(ROOT)):sha_file(p) for p in sorted((ROOT/'scripts').glob('*setmask*.py'))})
    for name in ('analyze_fusion_arms.py','probe_visibility.py'):
        path=ROOT/'scripts'/name
        sources['scripts/'+name]=sha_file(path)
    inputs={}
    for fold in (1,2):
        ds=PanNukeFold(fold)
        for name in ('images','inst','type','tissue'):
            inputs[f'fold{fold}/{name}.npy']=sha_file(ds.dir/(name+'.npy'))
    inputs['UNI']=sha_file(cfg.uni_ckpt)
    return {'sources':sources,'inputs':inputs,'torch':torch.__version__,'numpy':np.__version__}


def prepare_run(cfg,provenance,resume=False):
    out=Path(cfg.out_dir);current=asdict(cfg)
    if resume:
        if not all((out/n).is_file() for n in ('config.json','provenance.json','last.pth')) or (out/'final.pth').exists():
            raise ValueError('Explicit resume needs an incomplete, owned epoch checkpoint')
        if json.loads((out/'config.json').read_text())!=current or json.loads((out/'provenance.json').read_text())!=provenance:
            raise ValueError('Configuration/source/input provenance mismatch')
        try:state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        except Exception as exc:raise ValueError('Unreadable checkpoint') from exc
        required={'model','opt','sched','epoch','global_step','rng','config','provenance'}
        if not required.issubset(state) or state['config']!=current or state['provenance']!=provenance:
            raise ValueError('Incomplete or mismatched checkpoint state')
        if not isinstance(state['epoch'],int) or not 0<=state['epoch']<cfg.epochs:
            raise ValueError('Invalid completed epoch')
        return state
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Refusing existing output; use a new directory or explicit verified resume')
    out.mkdir(parents=True,exist_ok=True)
    for name,payload in (('config.json',current),('provenance.json',provenance)):
        with (out/name).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    return None


def epoch_loader(dataset,weights,seed,epoch,batch_size=16,workers=8):
    return DataLoader(dataset,batch_size=batch_size,sampler=sample_keys(weights,seed,epoch),
                      drop_last=True,num_workers=workers,pin_memory=True,persistent_workers=False,
                      generator=torch.Generator().manual_seed(stable_seed(seed,epoch,1234,bits=63)))


def move_batch(batch,device,arm):
    names={'img','np_map','hv_map','tp_map','small_map','tissue'}
    if arm=='Q':names.discard('tp_map')
    return {k:(v.to(device,non_blocking=True) if k in names and torch.is_tensor(v) else v) for k,v in batch.items()}


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else nullcontext()


def run_epoch(model,loader,opt,loss_fn,device,arm):
    model.train();aggregate={};digest=hashlib.sha256();steps=0;t0=time.monotonic()
    for batch in loader:
        for d in batch['digest']:digest.update(d.encode('ascii'));digest.update(b'\n')
        batch=move_batch(batch,device,arm)
        opt.zero_grad(set_to_none=True)
        with amp(device):pred=model(batch['img'])
        loss,terms=loss_fn(pred,batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; fail-stop')
        loss.backward();opt.step()
        for k,v in {'loss':float(loss.detach()),**terms}.items():aggregate[k]=aggregate.get(k,0.)+float(v)
        steps+=1
    if not steps:raise ValueError('Empty training epoch')
    return {**{f'train/{k}':v/steps for k,v in aggregate.items()},'data_digest':digest.hexdigest(),
            'steps':steps,'train_seconds':time.monotonic()-t0}


@torch.no_grad()
def validate_loss(model,loader,loss_fn,device,arm):
    model.eval();total=0.;steps=0
    for batch in loader:
        batch=move_batch(batch,device,arm)
        with amp(device):pred=model(batch['img'])
        value,_=loss_fn(pred,batch)
        if not torch.isfinite(value):raise ValueError('Nonfinite validation loss')
        total+=float(value);steps+=1
    return {'val/loss':total/max(steps,1)}


def build_model(cfg,pretrained=True):
    return (CellViTUNI if cfg.arm=='B' else SetMaskUNI)(uni_ckpt=cfg.uni_ckpt if pretrained else None)


def loss_for(cfg):
    if cfg.arm=='Q':return setmask_loss
    base=TrainConfig(split=1,out_dir=cfg.out_dir)
    return lambda pred,batch:cellvit_loss(to_nhwc(pred),batch,base)


def train_dev(cfg,device='cuda',resume=False):
    if cfg.workers!=8 or torch.device(device).type!='cuda' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Formal E6 requires workers8 and exactly one visible admitted CUDA device')
    provenance=make_provenance(cfg)
    state=prepare_run(cfg,provenance,resume=resume)
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    model=build_model(cfg,pretrained=state is None).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,betas=(.85,.95),weight_decay=cfg.weight_decay)
    sched=torch.optim.lr_scheduler.ExponentialLR(opt,cfg.gamma)
    train_ds=PairedDataset(1,True,cfg.seed);val_ds=PairedDataset(2,False,cfg.seed)
    weights=cell_tissue_weights(train_ds,cfg.sampling_gamma)
    val_loader=DataLoader(val_ds,batch_size=16,num_workers=cfg.workers,pin_memory=True,
                          generator=torch.Generator().manual_seed(stable_seed(cfg.seed,212)))
    ep0=global_step=0;previous_elapsed=0.
    if state is not None:
        model.load_state_dict(state['model']);opt.load_state_dict(state['opt']);sched.load_state_dict(state['sched'])
        ep0=state['epoch']+1;global_step=state['global_step'];previous_elapsed=state.get('elapsed_seconds',0.)
        restore_rng(state['rng'])
        del state
    loss_fn=loss_for(cfg);out=Path(cfg.out_dir);t0=time.monotonic()
    with (out/'log.jsonl').open('a',buffering=1) as log:
        for epoch in range(ep0,cfg.epochs):
            if previous_elapsed+time.monotonic()-t0>4*3600:raise RuntimeError('E6 per-run4h budget exhausted')
            model.freeze_encoder(epoch<cfg.unfreeze_epoch)
            loader=epoch_loader(train_ds,weights,cfg.seed,epoch,cfg.batch_size,cfg.workers)
            rec=run_epoch(model,loader,opt,loss_fn,device,cfg.arm);global_step+=rec['steps'];sched.step()
            rec.update(epoch=epoch,lr=opt.param_groups[0]['lr'],global_step=global_step)
            if (epoch+1)%5==0 or epoch+1==cfg.epochs:
                rec.update(validate_loss(model,val_loader,loss_fn,device,cfg.arm))
            elapsed=previous_elapsed+time.monotonic()-t0
            atomic_save(out/'last.pth',model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),
                        epoch=epoch,global_step=global_step,rng=capture_rng(),config=asdict(cfg),
                        provenance=provenance,elapsed_seconds=elapsed)
            log.write(json.dumps(rec)+'\n');print(json.dumps(rec),flush=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite final weights')
    atomic_save(out/'final.pth',model=model.state_dict(),config=asdict(cfg),provenance=provenance)
    return model


def load_config(run):
    payload=json.loads((Path(run)/'config.json').read_text())
    cfg=SetMaskConfig(**{k:payload[k] for k in ('arm','seed','out_dir','uni_ckpt','workers')})
    if asdict(cfg)!=payload:raise ValueError('Unsupported E6 configuration')
    return cfg


def load_final(run,device='cuda'):
    run=Path(run);cfg=load_config(run)
    p=json.loads((run/'provenance.json').read_text())
    if make_provenance(cfg)!=p:raise ValueError('Current inference source/input identity differs from training')
    state=torch.load(run/'final.pth',map_location='cpu',weights_only=False)
    if state.get('config')!=asdict(cfg) or state.get('provenance')!=p:raise ValueError('Final checkpoint provenance mismatch')
    model=build_model(cfg,pretrained=False);model.load_state_dict(state['model']);del state
    return cfg,model.eval().to(device)


@torch.no_grad()
def infer_batch(model,images,arm,device='cuda'):
    raw=torch.from_numpy(np.array(images,copy=True)).to(device)
    if arm=='B':
        p=_forward_probs(model,raw)
        tp=p['tp'].argmax(-1,keepdim=True).float()
        maps=torch.cat((tp,p['np'][...,1:],p['hv']),-1).cpu().numpy()
        rows=[_post((row,1.,1.)) for row in maps]
        return np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows]),p['tissue'].float().cpu().numpy()
    if arm!='Q':raise ValueError('Unknown E6 arm')
    mean=raw.new_tensor(UNI_MEAN,dtype=torch.float32);std=raw.new_tensor(UNI_STD,dtype=torch.float32)
    x=((raw.float()/255-mean)/std).permute(0,3,1,2).contiguous()
    with amp(device):pred=model(x)
    inst,typ=decode_set(pred)
    return inst,typ,pred['tissue'].float().softmax(-1).cpu().numpy()


def predict_dev(run,fold=2,device='cuda',batch_size=16):
    if fold!=2:raise ValueError('E6 prediction is validation-fold2 only')
    run=Path(run);path=run/'pred_fold2.npz'
    if path.exists():raise ValueError('Prediction already exists; no overwrite/replay')
    cfg,model=load_final(run,device)
    ds=PanNukeFold(2)
    if cfg.arm=='B':
        inst,typ,tissue=predict_fold(model,ds,device=device,batch_size=batch_size,workers=8,tta=False,inst_probs=False)
    else:
        n=len(ds);inst=np.empty((n,256,256),np.int32);typ=np.empty((n,256,256),np.uint8);tissue=np.empty((n,19),np.float32)
        for start in range(0,n,batch_size):
            stop=min(start+batch_size,n)
            inst[start:stop],typ[start:stop],tissue[start:stop]=infer_batch(model,ds.images[start:stop],cfg.arm,device)
    fd,name=tempfile.mkstemp(prefix='.prediction-',dir=run)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,inst=inst,type=typ,tissue_prob=tissue);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
    return path
