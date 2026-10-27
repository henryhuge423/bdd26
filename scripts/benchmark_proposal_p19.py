"""Fitting-only physical-batch timing and numerical witness, on an allocated GPU."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p19 as p19
from verify_p18_control import reset, reference_optimizer, hashes, optimizer_hash, rng_hashes


def create(weights,seed=19):
    reset(seed)
    model=p19.Proposal(weights,True,'C').cuda().train()
    return model,reference_optimizer(model)


def step(model,opt,fold,ids,micro,seed=19,arm='C',collect=False):
    opt.zero_grad(set_to_none=True)
    count=sum(len(fold.gt[i]) for i in ids)
    total=0.
    for start in range(0,len(ids),micro):
        x,y,n=fold.batch(ids[start:start+micro],seed,0,True)
        assert n==sum(len(fold.gt[i]) for i in ids[start:start+micro])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x.cuda())
            assert torch.is_autocast_enabled('cuda') and logits.dtype==torch.float32
        loss=p19.loss_for_microbatch(logits,y.cuda(),count,arm)
        assert torch.isfinite(loss)
        loss.backward();total+=float(loss.detach())
    params=[p for p in model.parameters() if p.requires_grad]
    if collect:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    gradient=torch.cat([p.grad.detach().cpu().flatten() for p in params]) if collect else None
    norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
    if collect:assert math.isfinite(float(norm)) and float(norm)>0
    opt.step()
    return total,gradient


def numeric_trials(weights,fold,ids,seed):
    refs=[];reference_updates=None;reports={}
    for micro in (4,8,16):
        model=opt=initial=gradient=updates=None
        try:
            model,opt=create(weights,seed);frozen=p19.frozen_hashes(model);trial=[]
            params=[p for p in model.parameters() if p.requires_grad]
            initial=torch.cat([p.detach().cpu().flatten() for p in params])
            encoder_size=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
            for start in (0,16):
                loss,gradient=step(model,opt,fold,ids[start:start+16],micro,seed=seed,collect=True)
                if micro==4:refs.append((loss,gradient));gradient=None
                else:
                    result=p19.numeric_comparison(refs[start//16][0],loss,refs[start//16][1],gradient)
                    for name,part in [('encoder',slice(0,encoder_size)),('head',slice(encoder_size,None))]:
                        result[name]=p19.numeric_comparison(refs[start//16][0],loss,refs[start//16][1][part],gradient[part])
                    trial.append(result);gradient=None
            updates=torch.cat([p.detach().cpu().flatten() for p in params])-initial
            if micro==4:reference_updates=updates
            differences={}
            for name,part in [('all',slice(None)),('encoder',slice(0,encoder_size)),('head',slice(encoder_size,None))]:
                difference=updates[part]-reference_updates[part]
                differences[name]=dict(relative_l2=float(difference.norm()/reference_updates[part].norm().clamp_min(1e-20)),max_absolute=float(difference.abs().max()))
            assert p19.frozen_hashes(model)==frozen
            reports[micro]=dict(passed=all(x['passed'] and x['encoder']['passed'] and x['head']['passed'] for x in trial),steps=trial,
                update_difference_after_two_steps=differences,frozen_unchanged=True)
        except torch.cuda.OutOfMemoryError:
            if micro==4:raise
            reports[micro]=dict(passed=False,failure='CUDA_OOM',frozen_unchanged=False)
        finally:
            model=opt=initial=gradient=updates=None
            params=None;gc.collect();torch.cuda.empty_cache()
    return reports


def control_parity(weights,fold,fit):
    from verify_p18_control import run
    import verify_p18_control as verifier
    old=verifier.p18;verifier.p18=p19
    try:
        for seed in (19,1):
            ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(fit).tolist()[:32]
            records=[]
            for kind in ('P3','C'):
                reset(seed);model=(original.Proposal(weights,True) if kind=='P3' else p19.Proposal(weights,True,'C')).cuda()
                initial=hashes(model.state_dict().items());rng=rng_hashes()
                records.append(dict(initial=initial,rng=rng,steps=run(model,fold,ids,seed,None if kind=='P3' else 'C')))
                del model;gc.collect();torch.cuda.empty_cache()
            assert records[0]==records[1],seed
    finally:verifier.p18=old
    return dict(seeds=[19,1],full_initial_gradient_parameter_AdamW_RNG_equal=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);assert not out.exists();torch.set_num_threads(4)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(a.weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(Path(a.data)/name)==digest
    assert original.sha(Path(a.data)/'fold1/tissue.npy')==p19.TISSUE_SHA
    fold=p19.Fold(a.data,1,'C');fit,audit=p19.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt]);cohort=p19.validate_cohort(fit,audit)
    selected=fit[:256];assert len(selected)==256 and not set(selected)&set(audit)
    ids=np.random.default_rng(np.random.SeedSequence([19,0,123])).permutation(selected).tolist()
    parity=control_parity(a.weights,fold,selected)
    by_seed={}
    for seed in (19,1):
        numeric_ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(selected).tolist()
        by_seed[str(seed)]=numeric_trials(a.weights,fold,numeric_ids,seed)
    numeric={k:dict(passed=all(by_seed[s][k]['passed'] for s in by_seed),frozen_unchanged=all(by_seed[s][k]['frozen_unchanged'] for s in by_seed),by_seed={s:by_seed[s][k] for s in by_seed}) for k in (4,8,16)}
    rows={k:dict(seconds=[],peak=0,passed=numeric[k]['passed']) for k in (4,8,16)}
    for order in ((4,8,16),(16,8,4),(4,8,16)):
        for micro in order:
            if not rows[micro]['passed']:continue
            model=opt=None
            try:
                model,opt=create(a.weights)
                # Identical fitting-only warmup then reset; include target generation and transfers in timing.
                step(model,opt,fold,ids[:16],micro)
                del model,opt;gc.collect();torch.cuda.empty_cache()
                model,opt=create(a.weights);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.monotonic()
                for block in p19.effective_batches(ids,16):step(model,opt,fold,block,micro)
                torch.cuda.synchronize();seconds=time.monotonic()-started
                rows[micro]['seconds'].append(seconds)
                rows[micro]['peak']=max(rows[micro]['peak'],torch.cuda.max_memory_allocated())
                print(json.dumps(dict(micro=micro,seconds=seconds,peak=rows[micro]['peak'])),flush=True)
            except torch.cuda.OutOfMemoryError:
                if micro==4:raise
                rows[micro].update(passed=False,failure='CUDA_OOM')
            finally:
                model=opt=None;gc.collect();torch.cuda.empty_cache()
    chosen=p19.select_microbatch(rows)
    result=dict(status='PASS',selected_microbatch=chosen,effective_batch=16,selection='fastest median >=5% over micro4, numeric pass, <=12GiB',
        trials=rows,numerical=numeric,original_control=parity,fit=selected,cohort_sha256=cohort,
        fold1_only=True,no_audit_predictions=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__)
    p19.dump(out,result)


if __name__=='__main__':main()
