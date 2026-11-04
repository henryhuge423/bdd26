"""P23: original online trajectory and fixed epoch-final parameter EMA; fold1 only."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import train_proposal_p22 as base

DECAY = .999


class ParameterEMA:
    """Trainable-only independent FP32 state. This is not a training-resume API."""
    def __init__(self, model):
        if list(model.named_buffers()):
            raise ValueError('P23 does not permit model buffers')
        self.shadow = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
        if not self.shadow:
            raise ValueError('empty trainable state')
        self.updates = 0

    def _parameters(self, model):
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(params) != set(self.shadow) or list(model.named_buffers()):
            raise ValueError('EMA model membership changed')
        if any(params[n].shape != s.shape for n, s in self.shadow.items()):
            raise ValueError('EMA model shape changed')
        return params

    @torch.no_grad()
    def update(self, model, successful_step=True):
        params = self._parameters(model)
        if type(successful_step) is not bool:
            raise ValueError('successful_step must be boolean')
        if not successful_step:
            return
        for name, shadow in self.shadow.items():
            shadow.mul_(DECAY).add_(params[name].detach().float(), alpha=1-DECAY)
        self.updates += 1

    def state_dict(self):
        return dict(decay=DECAY, updates=self.updates,
                    shadow={n: s.detach().cpu().clone() for n, s in self.shadow.items()})

    def load_state_dict(self, state, expected_updates):
        if (not isinstance(state, dict) or set(state) != {'decay','updates','shadow'}
            or state['decay'] != DECAY or type(state['updates']) is not int
            or type(expected_updates) is not int or expected_updates < 0
            or state['updates'] != expected_updates or not isinstance(state['shadow'], dict)
            or set(state['shadow']) != set(self.shadow)):
            raise ValueError('invalid EMA checkpoint metadata')
        for n, s in state['shadow'].items():
            if (not torch.is_tensor(s) or s.shape != self.shadow[n].shape
                or s.dtype != torch.float32 or not torch.isfinite(s).all()):
                raise ValueError('invalid EMA checkpoint tensor')
        new = {n:s.detach().to(self.shadow[n].device).clone() for n,s in state['shadow'].items()}
        self.shadow = new
        self.updates = expected_updates

    @torch.no_grad()
    def apply(self, model):
        params = self._parameters(model)
        if not all(torch.isfinite(s).all() for s in self.shadow.values()):
            raise ValueError('nonfinite EMA')
        for n, p in params.items():
            p.copy_(self.shadow[n])


def optimizer_step(model, optimizer, ema):
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        1, error_if_nonfinite=True)
    optimizer.step()
    ema.update(model)
    return norm


def run_config(seed):
    if type(seed) is not int or seed not in (19,1):
        raise ValueError('P23 seeds must be 19 or 1')
    config = base.run_config('C', seed)
    config.update(experiment='P23', ema_decay=DECAY, outputs=['C','E'],
                  ema_initialization='initial_trainable_clone', ema_update='after_each_successful_optimizer_step',
                  checkpoint_rule='epoch130_online_and_epoch130_EMA', resume_supported=False)
    return config


def create_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,choices=[19,1],default=19)
    p.add_argument('--smoke',action='store_true')
    return p.parse_args(argv)


@torch.no_grad()
def write_examples(model,fold,ids,out,scope,limit,arm):
    out=create_output(out);records=[];model.eval()
    for i in base.select_examples(ids,fold.gt,limit):
        x=torch.from_numpy(fold.images[i].copy()).permute(2,0,1).float()[None].cuda()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):prob=model(x).float().sigmoid()[0].cpu().numpy()
        points=base.decode_peaks(prob);gt=fold.gt[i];pairs=base.match_points(points,gt)
        row=base.score_points(points,gt);assert row['matched']==len(pairs)
        name=f'{scope}_{i:04d}.png'
        base.render_example(fold.images[i],gt,points,out/name,f'P23 {arm} {scope} image{i} {fold.tissue[i]}')
        records.append(dict(image=int(i),scope=scope,tissue=str(fold.tissue[i]),gt=gt,points=points,unique_pairs=pairs,score=row,png=name))
    base.dump(out/'coordinates.json',dict(selection='GT-only deterministic strata',arm=arm,scope=scope,records=records,main_map_only=True))


def main():
    args=parse_args();config=run_config(args.seed);out=create_output(args.out)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic();assert base.sha(args.weights)==base.WEIGHT_SHA
    inputs={}
    for name,digest in base.DATA_SHA.items():
        if name.startswith('fold1/'):
            inputs[name]=base.sha(Path(args.data)/name);assert inputs[name]==digest
    inputs['fold1/tissue.npy']=base.sha(Path(args.data)/'fold1/tissue.npy')
    assert inputs['fold1/tissue.npy']==base.TISSUE_SHA
    fold=base.Fold(args.data,1,'C')
    fit,audit=base.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    cohort=base.validate_cohort(fit,audit);assert len(fit)==2134 and len(audit)==522
    smoke_fit=base.smoke_ids(fit,fold.gt)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    model=base.Proposal(args.weights,True,'C').cuda();ema=ParameterEMA(model)
    from nucseg.cellvit import model as source
    assert Path(source.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent/'src')
    names=[n for n,p in model.encoder.named_parameters() if p.requires_grad]
    assert sorted({int(n.split('.')[1]) for n in names})==list(range(12,24))
    assert all(n.startswith('blocks.') for n in names)
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.23.5'
    manifest=dict(config=config,arguments=vars(args),weights_sha256=base.WEIGHT_SHA,input_sha256=inputs,
        cohort_sha256=cohort,fit=fit,audit=audit,smoke_fit=smoke_fit if args.smoke else None,
        source=str(Path(source.__file__).resolve()),torch=torch.__version__,numpy=np.__version__,
        trainable_encoder_names=names,buffer_names=[],trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        head_parameters=sum(p.numel() for p in model.head.parameters()),gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    base.dump(out/'manifest.json',manifest)
    opt=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)],betas=(.9,.999),weight_decay=.01)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    frozen_before=base.frozen_hashes(model)
    training_ids=smoke_fit if args.smoke else fit;epochs=1 if args.smoke else 130
    for epoch in range(epochs):
        if time.monotonic()-started>12*3600:raise TimeoutError('P23 12h budget')
        if shutil.disk_usage(out).free<20<<30:raise RuntimeError('disk floor')
        model.train();factor=.1+.9*(1+math.cos(math.pi*epoch/129))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        ids=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,123])).permutation(training_ids).tolist()
        batch_lengths=[];loss_sum=0.;num=0;torch.cuda.synchronize();epoch_start=time.monotonic()
        for selected in base.effective_batches(ids,16):
            count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True);step_loss=0.
            batch_lengths.append(len(selected))
            for begin in range(0,len(selected),4):
                batch=selected[begin:begin+4];x,y,_=fold.batch(batch,args.seed,epoch,True);x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
                assert logits.dtype==torch.float32
                loss=base.loss_for_microbatch(logits,y,count,'C')
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();step_loss+=float(loss.detach())
            norm=optimizer_step(model,opt,ema);loss_sum+=step_loss*len(selected);num+=len(selected)
        torch.cuda.synchronize();training_seconds=time.monotonic()-epoch_start
        expected=(epoch+1)*math.ceil(len(training_ids)/16);assert ema.updates==expected
        save_start=time.monotonic()
        payload=dict(model=model.adapted_state(),optimizer=opt.state_dict(),completed_epoch=epoch+1,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config,ema=ema.state_dict())
        saved=base.checkpoint(out/'last.pth',payload);del payload
        save_seconds=time.monotonic()-save_start
        row=dict(epoch=epoch+1,train_loss=loss_sum/num,training_seconds=training_seconds,checkpoint_seconds=save_seconds,
            elapsed_seconds=time.monotonic()-started,checkpoint_bytes=saved,optimizer_steps=len(batch_lengths),ema_updates=ema.updates,
            effective_batch=16,microbatch=4,tail=batch_lengths[-1],peak_memory=torch.cuda.max_memory_allocated())
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        base.dump(out/'status.json',row);print(json.dumps(row),flush=True)
    assert ema.updates==(9 if args.smoke else 17420)
    assert base.frozen_hashes(model)==frozen_before
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    # Online output is immutable before switching this model to E.
    final_save_seconds=0.;reload_evidence={};summaries={};training_peak=row['peak_memory']
    x,_,_=fold.batch(training_ids[:2],args.seed,0,False);x=x.cuda()
    for arm in ('C','E'):
        armout=create_output(out/arm)
        if arm=='E':ema.apply(model)
        model.eval();save_start=time.monotonic()
        base.checkpoint(armout/'final.pth',dict(model=model.adapted_state(),weights_sha256=base.WEIGHT_SHA,
            config=config,output_arm=arm,ema_updates=ema.updates,completed_epoch=epochs))
        final_save_seconds+=time.monotonic()-save_start
        if arm=='C':online_digest=base.sha(armout/'final.pth')
        if args.smoke:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x).float().cpu()
            fresh=base.Proposal(args.weights,True,'C').cuda().eval()
            loaded=torch.load(armout/'final.pth',map_location='cpu',weights_only=False)
            base.validate_config(loaded['config'],config);assert loaded['output_arm']==arm
            base.load_adapted(fresh,loaded['model']);del loaded
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=fresh(x).float().cpu()
            assert torch.equal(prediction,actual);reload_evidence[arm]=True
            loaded=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
            base.validate_config(loaded['config'],config)
            base.load_adapted(fresh,loaded['model'])
            if arm=='E':
                restored=ParameterEMA(fresh);restored.load_state_dict(loaded['ema'],expected_updates=9);restored.apply(fresh);del restored
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=fresh(x).float().cpu()
            assert torch.equal(prediction,actual);reload_evidence['last_'+arm]=True
            del loaded,fresh;gc.collect();torch.cuda.empty_cache()
            write_examples(model,fold,training_ids,armout/'point_examples_smoke','fit_1epoch_preliminary',6,arm)
        else:
            base.dump(armout/'fit.json',base.evaluate(model,fold,fit))
            rows=base.evaluate(model,fold,audit);base.dump(armout/'evaluation.json',rows)
            summaries[arm]=base.aggregate(rows);base.dump(armout/'summary.json',summaries[arm])
            write_examples(model,fold,audit,armout/'point_examples_audit','audit_final130',8,arm)
            write_examples(model,fold,fit,armout/'point_examples_fit','fit_final130',4,arm)
    assert base.sha(out/'C/final.pth')==online_digest
    assert base.frozen_hashes(model)==frozen_before
    if args.smoke:
        estimate=130*training_seconds/134*2134+130*save_seconds+final_save_seconds+3600
        report=dict(status='MEASURED',training_images=134,training_seconds=training_seconds,checkpoint_seconds=save_seconds,
            final_checkpoint_seconds=final_save_seconds,estimated_seconds_130=estimate,safety_seconds=1.5*estimate,budget_hours=12,
            training_peak_memory=training_peak,ema_updates=ema.updates,effective_batch_lengths=batch_lengths,
            reload_equal=reload_evidence,online_checkpoint_unchanged=True,all_frozen_unchanged=True,smoke_fit=smoke_fit,
            no_audit_predictions=True,buffer_names=[],seconds=time.monotonic()-started)
        base.dump(out/'smoke_measurements.json',report)
        assert estimate*1.5<12*3600 and training_peak<=24*1024**3
        report['status']='PASS';base.dump(out/'smoke.json',report)
    else:
        base.dump(out/'complete.json',dict(exit_code=0,epochs=130,ema_updates=ema.updates,outputs=['C','E'],elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(event='complete',summary=summaries)),flush=True)


if __name__=='__main__':main()
