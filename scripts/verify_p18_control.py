"""Same-GPU real UNI control parity and fit-only RGB residual learning witnesses for P18."""
import gc
import hashlib
import json
import math
import random
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p18 as p18


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
    # Exact original runner metadata as well as numeric rates; no P18 helper used.
    return torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)],betas=(.9,.999),weight_decay=.01)


def run(model,fold,ids,seed,arm=None):
    opt=reference_optimizer(model)
    if arm is not None:opt=reference_optimizer(model)
    assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-5]
    if arm is not None:
        schedule_epoch(opt,129)
        assert all(abs(g['lr']-v)<1e-18 for g,v in zip(opt.param_groups,[3e-5,1e-6]))
        schedule_epoch(opt,0)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    frozen=p18.frozen_hashes(model);traces=[];model.train()
    watched={'encoder':model.encoder.blocks[23].attn.qkv.weight,'output':model.head.out.weight}
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
                else p18.loss_for_microbatch(logits,y.cuda(),count,arm))
            assert torch.isfinite(loss)
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
    assert p18.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return traces


def measure_k_step(before,after,lrs,weight_decay):
    if set(before)!=set(after) or set(before)!=set(lrs):raise ValueError('K update key mismatch')
    changes={k:float((after[k]-before[k]).abs().max()) for k in before}
    adaptive={k:float((after[k]-before[k]*(1-lrs[k]*weight_decay)).abs().max()) for k in before}
    return changes,adaptive

def validate_h_witness(gradients,changes,adaptive_changes):
    expected={'local0','local1','local_output','block12','block23','output'}
    if set(gradients)!=expected or set(changes)!=expected or set(adaptive_changes)!=expected:raise ValueError('incomplete H learning witness')
    if not all(math.isfinite(x) and x>0 for x in list(gradients.values())+list(changes.values())+list(adaptive_changes.values())):raise ValueError('invalid H learning witness')
    return True


def initial_signature(weights,fold,ids,seed,arm):
    reset(seed);model=p18.Proposal(weights,True,arm).cuda();model.train()
    initial=hashes((n,p) for n,p in model.named_parameters() if not n.startswith('head.local.'))
    rng=rng_hashes();count=sum(len(fold.gt[i]) for i in ids[:16]);x,y,_=fold.batch(ids[:4],seed,0,True)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
    assert logits.dtype==torch.float32
    loss=p18.loss_for_microbatch(logits,y.cuda(),count,arm);loss.backward()
    gradients=hashes((n,p.grad) for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.local.'))
    signature=dict(initial=initial,rng=rng,logits=hashes([('logits',logits)]),loss=float(loss),gradients=gradients)
    del model;gc.collect();torch.cuda.empty_cache();return signature


def run_h(model,fold,ids,seed):
    opt=reference_optimizer(model)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    watched={'local0':model.head.local.layers[0].weight,'local1':model.head.local.layers[3].weight,
        'local_output':model.head.local.output.weight,'block12':model.encoder.blocks[12].attn.qkv.weight,
        'block23':model.encoder.blocks[23].attn.qkv.weight,'output':model.head.out.weight}
    frozen=p18.frozen_hashes(model);model.train();precision={};handles=[]
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('local',model.head.local.output),('output',model.head.out)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    for start in (0,16):
        selected=ids[start:start+16];count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            x,y,n=fold.batch(selected[begin:begin+4],seed,0,True)
            rx,ry,rn=original.Fold.batch(fold,selected[begin:begin+4],seed,0,True)
            assert torch.equal(x,rx) and torch.equal(y,ry) and n==rn
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
            assert logits.dtype==torch.float32 and precision=={'encoder':'torch.bfloat16','local':'torch.bfloat16','output':'torch.bfloat16'}
            y=y.cuda();loss=p18.loss_for_microbatch(logits,y,count,'H');ref=original.focal_numerator(logits,y)/max(1,count)
            assert torch.equal(loss,ref)
            torch.testing.assert_close(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(ref,logits,retain_graph=True)[0],rtol=0,atol=0)
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        if start==16:
            gradients={k:float(p.grad.float().norm()) for k,p in watched.items()}
            before={k:p.detach().cpu().clone() for k,p in watched.items()}
            lrs={k:next(g['lr'] for g in opt.param_groups if any(p is q for q in g['params'])) for k,p in watched.items()}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
    for h in handles:h.remove()
    changes,adaptive=measure_k_step(before,{k:p.detach().cpu() for k,p in watched.items()},lrs,.01)
    validate_h_witness(gradients,changes,adaptive)
    assert p18.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return dict(gradients=gradients,changes=changes,adaptive_changes=adaptive,second_step_only=True,all_trainable_finite=True,AMP_verified=True,actual_data_loss_gradient_equal=True,all_frozen_unchanged=True)


from train_proposal_p17 import schedule_epoch

def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p18.Path(data)/name)==digest
    assert original.sha(p18.Path(data)/'fold1/tissue.npy')==p18.TISSUE_SHA
    fold=p18.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p18.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt]);p18.validate_cohort(fit,audit)
    smoke_fit=list(fit[:256]);updates={};initial={};measured={}
    for seed in (19,1):
        states=[]
        for arm in ('P3','C'):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p18.Proposal(weights,True,arm))
            states.append(dict(state=hashes(model.state_dict().items()),rng=rng_hashes()));del model;gc.collect()
        assert states[0]==states[1];initial[str(seed)]=True
        ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke_fit).tolist()
        assert len(ids)==len(set(ids))==256 and set(ids)==set(smoke_fit) and not set(ids)&set(audit)
        results=[]
        for arm,f in (('P3',reference),('C',fold)):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p18.Proposal(weights,True,arm)).cuda()
            results.append(run(model,f,ids[:32],seed,None if arm=='P3' else arm));del model;gc.collect();torch.cuda.empty_cache()
        assert results[0]==results[1];updates[str(seed)]=True
        assert initial_signature(weights,fold,ids,seed,'C')==initial_signature(weights,fold,ids,seed,'H')
        reset(seed);model=p18.Proposal(weights,True,'H').cuda();measured[str(seed)]=run_h(model,fold,ids[:32],seed)
        del model;gc.collect();torch.cuda.empty_cache()
    p18.dump(out,dict(status='PASS',seeds=[19,1],initial_P3_C_equal=initial,two_step_parity_by_seed=updates,
        C_full_gradient_parameter_AdamW_RNG_equal=True,H_by_seed=measured,H_C_initial_logits_loss_unclipped_base_gradients_RNG_equal=True,smoke_fit=smoke_fit,fold1_only=True,no_audit_predictions=True))

if __name__=='__main__':main()
