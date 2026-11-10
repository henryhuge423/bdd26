"""Same-GPU real UNI control parity and fit-only sampling witnesses for P26."""
import gc
import copy
import hashlib
import json
import math
import random
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p26 as p26


LEARNING={}

def backward_candidate(model, anchor, losses):
    return p26.backward_effective_batch(model, anchor, losses)

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


def reference_optimizer(model):
    # Exact original runner metadata as well as numeric rates; no P26 helper used.
    return torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5,initial_lr=1e-5)],betas=(.9,.999),weight_decay=.01)


def run(model,fold,ids,seed,arm=None):
    opt=reference_optimizer(model)
    if arm is not None:opt=p26.make_optimizer(model,arm)
    expected=[3e-4, 1e-5]
    assert len(opt.param_groups)==len(expected)
    assert all(math.isclose(g['lr'],v,rel_tol=1e-14) for g,v in zip(opt.param_groups,expected))
    if arm is not None:
        p26.schedule_epoch(opt,129)
        assert all(math.isclose(g['lr'],v*.1,rel_tol=1e-14) for g,v in zip(opt.param_groups,expected))
        p26.schedule_epoch(opt,0)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    frozen=p26.frozen_hashes(model);traces=[];model.train()
    anchor=p26.snapshot_anchor(model) if arm=='R' else None
    precision={};handles=[]
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    for start in (0,16):
        selected=ids[start:start+16];assert len(selected)==16
        count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        def losses():
            for begin in range(0,len(selected),4):
                x,y,_=fold.batch(selected[begin:begin+4],seed,0,True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(x.cuda())
                    assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
                assert logits.dtype==torch.float32
                loss=(original.focal_numerator(logits,y.cuda())/max(1,count) if arm is None
                    else p26.loss_for_microbatch(logits,y.cuda(),count,arm))
                assert torch.isfinite(loss)
                yield loss
        state={}
        real_penalty=p26.anchor_penalty
        def capture_penalty(current_model,current_anchor):
            observed={str(b):model.encoder.blocks[b].attn.qkv.weight for b in (12,18,23)}
            old={k:p.detach().clone() for k,p in observed.items()}
            data_grads={k:p.grad.detach().clone() for k,p in observed.items()}
            data_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for _,p in trainable]))
            reg=real_penalty(current_model,current_anchor)
            assert torch.isfinite(reg) and float(reg)>0
            expected={k:.01*(p.detach()-anchor[f'blocks.{k}.attn.qkv.weight']) for k,p in observed.items()}
            assert all(torch.isfinite(v).all() and float(v.norm())>0 for v in expected.values())
            state.update(observed=observed,old=old,data_grads=data_grads,data_norm=data_norm,expected=expected,reg=reg)
            return reg
        if arm is None:
            for loss in losses(): loss.backward()
        elif arm=='R' and start==16:
            p26.anchor_penalty=capture_penalty
            try:
                backward_candidate(model,anchor,losses())
            finally:
                p26.anchor_penalty=real_penalty
            assert state, 'production helper did not call anchor penalty'
            observed=state['observed'];old=state['old'];data_grads=state['data_grads']
            data_norm=state['data_norm'];expected=state['expected'];reg=state['reg']
            assert all(torch.allclose(p.grad,data_grads[k]+expected[k],atol=1e-7,rtol=1e-5) for k,p in observed.items())
            deltas={k:float((p.grad-data_grads[k]).abs().max()) for k,p in observed.items()}
            assert all(math.isfinite(v) and v>0 for v in deltas.values())
            q=observed['12'];data_grad=data_grads['12']
            clone=torch.nn.Parameter(q.detach().clone())
            group=next(g for g in opt.param_groups if any(p is q for p in g['params']))
            counter=torch.optim.AdamW([clone],lr=group['lr'],betas=group['betas'],weight_decay=group['weight_decay'],eps=group['eps'])
            counter.state[clone]=copy.deepcopy(opt.state[q])
            clone.grad=data_grad*min(1.,1./(float(data_norm)+1e-6))
            counter.step()
            witness=dict(anchor_loss=float(reg.detach()),anchor_gradient_norm={k:float(v.norm()) for k,v in expected.items()},data_gradient_norm={k:float(v.norm()) for k,v in data_grads.items()},gradient_delta_max=deltas)
        else:
            backward_candidate(model,anchor,losses())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in trainable)
        if start==16:
            for p in (model.encoder.blocks[23].attn.qkv.weight,model.head.out.weight):
                assert math.isfinite(float(p.grad.norm())) and float(p.grad.norm())>0
        gradients=hashes((n,p.grad) for n,p in trainable)
        watched=dict(block12=model.encoder.blocks[12].attn.qkv.weight,block23=model.encoder.blocks[23].attn.qkv.weight,output=model.head.out.weight)
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1,error_if_nonfinite=True);opt.step()
        if arm=='R' and start==16:
            difference=float((q.detach()-clone.detach()).abs().max())
            assert difference>0 and math.isfinite(difference)
            beyond={k:float((p.detach()-old[k]*(1-group['lr']*group['weight_decay'])).abs().max()) for k,p in observed.items()}
            assert all(math.isfinite(v) and v>0 for v in beyond.values())
            witness['update_difference_max']=difference
            witness['beyond_weight_decay_max']=beyond
            LEARNING[str(seed)]=witness

        traces.append(dict(gradients=gradients,parameters=hashes(model.adapted_state().items()),
            optimizer_state=optimizer_hash(opt.state_dict()),rng=rng_hashes()))
    for h in handles:h.remove()
    assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16')
    assert p26.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return traces


def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p26.Path(data)/name)==digest
    assert original.sha(p26.Path(data)/'fold1/tissue.npy')==p26.TISSUE_SHA
    fold=p26.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p26.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    p26.validate_cohort(fit,audit);smoke_fit=p26.smoke_ids(fit,fold.gt);updates={};initial={}
    for seed in (19,1):
        states=[]
        for arm in ('P3','C','R'):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p26.Proposal(weights,True,arm))
            states.append(dict(state=hashes(model.state_dict().items()),rng=rng_hashes()))
            del model;gc.collect()
        assert states[0]==states[1]==states[2];initial[str(seed)]=True
        ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke_fit).tolist()
        assert len(ids)==len(set(ids))==134 and set(ids)==set(smoke_fit) and not set(ids)&set(audit)
        results=[]
        for arm,f in (('P3',reference),('C',fold),('R',fold)):
            reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p26.Proposal(weights,True,arm)).cuda()
            results.append(run(model,f,ids[:32],seed,None if arm=='P3' else arm))
            del model;gc.collect();torch.cuda.empty_cache()
        assert results[0]==results[1]
        updates[str(seed)]=True
    p26.dump(out,dict(status='PASS',learning=LEARNING,seeds=[19,1],exact_initial_state_equal=initial,
        initial_P3_C_R_equal=True,actual_two_optimizer_steps_equal=True,two_step_parity_by_seed=updates,
        all_trainable_gradients_equal=True,all_adapted_parameters_equal=True,all_optimizer_states_equal=True,
        all_update_rng_equal=True,all_frozen_encoder_tensors_unchanged=True,R_anchor_gradient_and_update=True,
        original_order_equal=True,exact_optimizer_membership=True,AMP_verified=True,smoke_fit=smoke_fit,
        fold1_only=True,no_audit_predictions=True))

if __name__=='__main__':main()
