"""Fit-only original-P3 parity with EMA bookkeeping and analytical EMA witnesses."""
import gc
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p22 as base
import train_proposal_p23 as p23
import verify_p18_control as v


def validate_ema_witness(row):
    flags={'analytic_equal','online_unchanged','rng_unchanged','keys_exact','finite','shadow_changed','shadow_differs_online'}
    if (not isinstance(row,dict) or set(row)!=flags|{'updates'}
        or type(row['updates']) is not int or row['updates']!=2
        or any(row[k] is not True for k in flags)):
        raise ValueError('incomplete or failed P23 EMA witness')
    return True


def run_with_ema(model,fold,ids,seed):
    opt=v.reference_optimizer(model);ema=p23.ParameterEMA(model)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    frozen=base.frozen_hashes(model);traces=[];model.train()
    watch=['encoder.blocks.12.attn.qkv.weight','encoder.blocks.23.attn.qkv.weight','head.out.weight']
    params=dict(model.named_parameters())
    independent={n:params[n].detach().clone() for n in watch}
    initial={n:s.clone() for n,s in independent.items()}
    precision={};handles=[]
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    for start in (0,16):
        selected=ids[start:start+16];count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            batch=selected[begin:begin+4];x,y,n=fold.batch(batch,seed,0,True)
            rx,ry,rn=original.Fold.batch(fold,batch,seed,0,True)
            assert torch.equal(x,rx) and torch.equal(y,ry) and n==rn
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
            assert logits.dtype==torch.float32
            y=y.cuda();loss=base.loss_for_microbatch(logits,y,count,'C')
            assert torch.equal(loss,original.focal_numerator(logits,y)/max(1,count))
            loss.backward()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in trainable)
        if start==16:
            assert all(float(params[n].grad.norm())>0 for n in watch)
        gradients=v.hashes((n,p.grad) for n,p in trainable)
        # The production step includes clipping, AdamW and exactly one shadow update.
        rng=v.rng_hashes();p23.optimizer_step(model,opt,ema);assert v.rng_hashes()==rng
        for n in watch:
            independent[n]=independent[n]*.999+params[n].detach()*.001
            torch.testing.assert_close(ema.shadow[n],independent[n],rtol=2e-6,atol=1e-7)
        traces.append(dict(gradients=gradients,parameters=v.hashes(model.adapted_state().items()),
                           optimizer_state=v.optimizer_hash(opt.state_dict()),rng=v.rng_hashes()))
    for h in handles:h.remove()
    assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
    assert base.frozen_hashes(model)==frozen
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    saved=v.hashes(model.adapted_state().items());rng=v.rng_hashes()
    state=ema.state_dict();restored=p23.ParameterEMA(model);restored.load_state_dict(state,expected_updates=2)
    assert all(torch.equal(ema.shadow[n],restored.shadow[n]) for n in ema.shadow)
    evidence=dict(updates=ema.updates,analytic_equal=True,online_unchanged=saved==v.hashes(model.adapted_state().items()),
        rng_unchanged=rng==v.rng_hashes(),keys_exact=set(ema.shadow)==set(model.adapted_state()),
        finite=all(bool(torch.isfinite(s).all()) for s in ema.shadow.values()),
        shadow_changed=all(not torch.equal(ema.shadow[n],initial[n]) for n in watch),
        shadow_differs_online=all(not torch.equal(ema.shadow[n],params[n]) for n in watch))
    validate_ema_witness(evidence)
    return traces,evidence


def main():
    weights,data,out=sys.argv[1:];torch.set_num_threads(4)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(base.Path(data)/name)==digest
    assert original.sha(base.Path(data)/'fold1/tissue.npy')==base.TISSUE_SHA
    fold=base.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=base.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    base.validate_cohort(fit,audit);smoke=base.smoke_ids(fit,fold.gt)
    initial={};parity={};witnesses={}
    for seed in (19,1):
        signatures=[]
        for kind in ('P3','EMA'):
            v.reset(seed);model=(original.Proposal(weights,True) if kind=='P3' else base.Proposal(weights,True,'C'))
            if kind=='EMA':ema=p23.ParameterEMA(model)
            signatures.append(dict(state=v.hashes(model.state_dict().items()),rng=v.rng_hashes(),
                                   trainable=sorted(n for n,p in model.named_parameters() if p.requires_grad)))
            if kind=='EMA':del ema
            del model;gc.collect()
        assert signatures[0]==signatures[1];initial[str(seed)]=True
        ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke).tolist()
        assert not set(ids)&set(audit)
        v.reset(seed);model=original.Proposal(weights,True).cuda()
        reference_trace=v.run(model,reference,ids[:32],seed)
        del model;gc.collect();torch.cuda.empty_cache()
        v.reset(seed);model=base.Proposal(weights,True,'C').cuda()
        actual,witnesses[str(seed)]=run_with_ema(model,fold,ids[:32],seed)
        assert reference_trace==actual;parity[str(seed)]=True
        del model;gc.collect();torch.cuda.empty_cache()
    base.dump(out,dict(status='PASS',seeds=[19,1],initial_full_state_RNG_equal=initial,
        two_step_parity_by_seed=parity,C_full_gradient_parameter_AdamW_RNG_equal=True,ema_by_seed=witnesses,
        smoke_fit=smoke,fold1_only=True,no_audit_predictions=True))


if __name__=='__main__':main()
