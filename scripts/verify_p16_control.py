"""Same-GPU real UNI control parity and fit-only objective witnesses for P16."""
import gc
import hashlib
import json
import math
import random
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p16 as p16


def hashes(items):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in items}


def optimizer_hash(state):
    def convert(value):
        if torch.is_tensor(value):
            return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        if isinstance(value,dict):return {str(k):convert(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [convert(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(convert(state),sort_keys=True).encode()).hexdigest()


def rng_hashes():
    return dict(cpu=hashes([('cpu',torch.get_rng_state())]),
        cuda=hashes([(str(i),v) for i,v in enumerate(torch.cuda.get_rng_state_all())]))


def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def measure_updates(before,after):
    assert set(before)==set(after)=={'encoder','output'}
    changes={}
    for name,old in before.items():
        new=after[name]
        assert old.shape==new.shape and old.numel()>0
        assert torch.isfinite(old).all() and torch.isfinite(new).all(),name
        change=float((new.float()-old.float()).abs().max())
        assert math.isfinite(change) and change>0,(name,change)
        changes[name]=change
    return changes


def reference_optimizer(model):
    # Exact original runner metadata as well as numeric rates; no P16 helper used.
    return torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)],betas=(.9,.999),weight_decay=.01)


def run(model,fold,ids,seed,arm=None):
    opt=reference_optimizer(model)
    if arm is not None:opt=p16.make_optimizer(model,arm)
    assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-5]
    if arm is not None:
        p16.schedule_epoch(opt,129)
        assert all(abs(g['lr']-v)<1e-18 for g,v in zip(opt.param_groups,[3e-5,1e-6]))
        p16.schedule_epoch(opt,0)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    frozen=p16.frozen_hashes(model);traces=[];model.train()
    watched={'encoder':model.encoder.blocks[23].attn.qkv.weight,'output':model.head.out.weight}
    before={k:p.detach().cpu().clone() for k,p in watched.items()} if arm=='R' else None
    precision={};handles=[]
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    for start in (0,16):
        selected=ids[start:start+16];assert len(selected)==16
        count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        for begin in range(0,len(selected),4):
            x,y,_=fold.batch(selected[begin:begin+4],seed,0,True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x.cuda())
                assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
            assert logits.dtype==torch.float32
            loss=(original.focal_numerator(logits,y.cuda())/max(1,count) if arm is None
                else p16.loss_for_microbatch(logits,y.cuda(),count,arm))
            assert torch.isfinite(loss)
            if arm=='R':
                # Independent expectation: direct exp expression and analytic logit derivative.
                target=y.cuda().float();prob=1/(1+torch.exp(-logits.detach().float()))
                expected=(prob-target).pow(2).sum()/max(1,count)
                torch.testing.assert_close(loss,expected,rtol=3e-6,atol=1e-7)
                analytic=2*(prob-target)*prob*(1-prob)/max(1,count)
                actual=torch.autograd.grad(loss,logits,retain_graph=True)[0]
                torch.testing.assert_close(actual,analytic,rtol=5e-6,atol=1e-9)
                halo=(target>0)&(target<1);assert int(halo.sum())>0
                assert torch.all(actual[halo & (prob<target)]<=0)
            loss.backward()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in trainable)
        if start==16:
            for p in (model.encoder.blocks[23].attn.qkv.weight,model.head.out.weight):
                assert math.isfinite(float(p.grad.norm())) and float(p.grad.norm())>0
        gradients=hashes((n,p.grad) for n,p in trainable)
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1,error_if_nonfinite=True);opt.step()
        traces.append(dict(gradients=gradients,parameters=hashes(model.adapted_state().items()),
            optimizer_state=optimizer_hash(opt.state_dict()),rng=rng_hashes()))
    for h in handles:h.remove()
    assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
    assert p16.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    if arm=='R':
        traces[-1]['R_two_step_changes']=measure_updates(before,{k:p.detach().cpu() for k,p in watched.items()})
    return traces


def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p16.Path(data)/name)==digest
    assert original.sha(p16.Path(data)/'fold1/tissue.npy')==p16.TISSUE_SHA
    fold=p16.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p16.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    p16.validate_cohort(fit,audit);smoke_fit=list(fit[:256]);updates={};initial={};measured={}
    for seed in (19,1):
        states=[]
        for arm in ('P3','C','R'):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p16.Proposal(weights,True,arm))
            states.append(dict(state=hashes(model.state_dict().items()),rng=rng_hashes()))
            del model;gc.collect()
        assert states[0]==states[1]==states[2];initial[str(seed)]=True
        ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke_fit).tolist()
        assert len(ids)==len(set(ids))==256 and set(ids)==set(smoke_fit) and not set(ids)&set(audit)
        results=[]
        for arm,f in (('P3',reference),('C',fold),('R',fold)):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p16.Proposal(weights,True,arm)).cuda()
            results.append(run(model,f,ids[:32],seed,None if arm=='P3' else arm))
            del model;gc.collect();torch.cuda.empty_cache()
        assert results[0]==results[1]
        # Real encoder tensors after two effective steps differ under the new objective.
        assert results[1][-1]['parameters']['encoder.blocks.23.attn.qkv.weight']!=results[2][-1]['parameters']['encoder.blocks.23.attn.qkv.weight']
        measured[str(seed)]=results[2][-1]['R_two_step_changes']
        updates[str(seed)]=True
    p16.dump(out,dict(status='PASS',seeds=[19,1],exact_initial_state_equal=initial,
        initial_P3_C_R_equal=True,actual_two_optimizer_steps_equal=True,two_step_parity_by_seed=updates,
        all_trainable_gradients_equal=True,all_adapted_parameters_equal=True,all_optimizer_states_equal=True,
        all_update_rng_equal=True,all_frozen_encoder_tensors_unchanged=True,R_independent_actual_loss_gradient=True,R_halo_witness=True,R_numeric_rate_checked=True,
        R_encoder_updates_differ=True,R_two_updates_finite_nonzero=True,R_two_step_changes=measured,original_order_equal=True,
        exact_optimizer_membership=True,AMP_verified=True,smoke_fit=smoke_fit,fold1_only=True,no_audit_predictions=True,
        independent_schedule_endpoints=dict(head=[3e-4,3e-5],C=[1e-5,1e-6],R=[1e-5,1e-6])))

if __name__=='__main__':main()
