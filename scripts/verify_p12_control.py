"""Same-GPU real UNI parity: P12 C versus original P3 T1, no audit output."""
import gc
import hashlib
import json
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p12 as p12


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


def run(model,fold,ids,use_p12):
    opt=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5)],betas=(.9,.999),weight_decay=.01)
    traces=[];model.train()
    for start in (0,16):
        selected=ids[start:start+16];den=max(1,sum(len(fold.gt[i]) for i in selected));opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            x,y,_=fold.batch(selected[begin:begin+4],19,0,True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
            loss=(p12.loss_for_microbatch(logits,y.cuda(),sum(len(fold.gt[i]) for i in selected),'C') if use_p12 else original.focal_numerator(logits,y.cuda())/den);loss.backward()
        gradients=hashes((n,p.grad) for n,p in model.named_parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
        traces.append(dict(gradients=gradients,parameters=hashes(model.adapted_state().items()),optimizer_state=optimizer_hash(opt.state_dict()),cpu_rng=hashes([('cpu',torch.get_rng_state())]),cuda_rng=hashes([(str(i),r) for i,r in enumerate(torch.cuda.get_rng_state_all())])))
    return hashes(model.adapted_state().items()),traces


def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    initial={}
    for seed in (19,1):
        states=[]
        for arm in ('C','R'):
            torch.manual_seed(seed);model=p12.Proposal(weights,True,arm)
            states.append(dict(state=hashes(model.state_dict().items()),rng=hashes([('cpu',torch.get_rng_state())])));del model;gc.collect()
        assert states[0]==states[1];initial[str(seed)]=True
    fold=p12.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p12.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    p12.validate_cohort(fit,audit)
    # Match the P12 smoke membership and epoch0 ordering exactly, not full-cohort shuffle.
    smoke_fit=list(fit[:256])
    ids=np.random.default_rng(np.random.SeedSequence([19,0,123])).permutation(smoke_fit).tolist()[:32]
    results=[]
    for implementation,f in ((original,reference),(p12,fold)):
        torch.manual_seed(19);model=implementation.Proposal(weights,True).cuda()
        results.append(run(model,f,ids,implementation is p12));del model;gc.collect();torch.cuda.empty_cache()
    assert results[0]==results[1]
    # R structural witness on actual positive GPU features; post-ReLU zeros excluded.
    dropout=p12.DropoutPointHead('R').dropout.cuda().train()
    ones=torch.ones(64,256,4,4,device='cuda');dropped=dropout(ones)
    assert set(dropped.unique().tolist())=={0.,1.25}
    assert torch.equal(dropped,dropped[:,:,:1,:1].expand_as(dropped))
    fraction=float((dropped==0).float().mean());assert .18<fraction<.22
    # Full model same-state eval equivalence; train stochasticity after nonzero output.
    torch.manual_seed(19);model=p12.Proposal(weights,True,'R').cuda()
    x,_,_=fold.batch(smoke_fit[:2],19,0,False);x=x.cuda()
    with torch.no_grad():model.head.out.weight.fill_(.01)
    model.train()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        first=model(x);second=model(x)
    assert not torch.equal(first,second)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x)
    torch.manual_seed(19);control=p12.Proposal(weights,True,'C').cuda().eval()
    control.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.equal(prediction,control(x)) and torch.equal(prediction,model(x))
    assert not model.head.dropout.training
    p12.dump(out,dict(dropout_structure_pass=True,dropout_fraction=fraction,real_train_stochastic=True,same_state_eval_equal=True,all_update_parameters_equal=True,all_update_rng_equal=True,status='PASS',seeds=[19,1],exact_initial_state_equal=initial,actual_two_optimizer_steps_equal=True,all_trainable_gradients_equal=True,all_adapted_parameters_equal=True,all_optimizer_states_equal=True,smoke_membership=256,smoke_fit=smoke_fit,optimizer_images=ids,fold1_only=True,dropout_initial_states_equal=True))


if __name__=='__main__':main()
