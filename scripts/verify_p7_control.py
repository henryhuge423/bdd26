"""Same-GPU real UNI parity: P7 C versus original P3 T1, no audit output."""
import gc
import hashlib
import json
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p7 as p7


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


def run(model,fold,ids):
    opt=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5)],betas=(.9,.999),weight_decay=.01)
    traces=[];model.train()
    for start in (0,16):
        selected=ids[start:start+16];den=max(1,sum(len(fold.gt[i]) for i in selected));opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            x,y,_=fold.batch(selected[begin:begin+4],19,0,True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
            loss=p7.focal_numerator(logits,y.cuda())/den;loss.backward()
        gradients=hashes((n,p.grad) for n,p in model.named_parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
        traces.append(dict(gradients=gradients,optimizer_state=optimizer_hash(opt.state_dict())))
    return hashes(model.adapted_state().items()),traces


def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    initial={}
    for seed in (19,1):
        states=[]
        for arm in ('C','J'):
            torch.manual_seed(seed);model=p7.Proposal(weights,True)
            states.append(hashes(model.state_dict().items()));del model;gc.collect()
        assert states[0]==states[1];initial[str(seed)]=True
    fold=p7.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p7.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    p7.validate_cohort(fit,audit)
    # Match the P7 smoke membership and epoch0 ordering exactly, not full-cohort shuffle.
    ids=np.random.default_rng(np.random.SeedSequence([19,0,123])).permutation(fit[:256]).tolist()[:32]
    results=[]
    for implementation,f in ((original,reference),(p7,fold)):
        torch.manual_seed(19);model=implementation.Proposal(weights,True).cuda()
        results.append(run(model,f,ids));del model;gc.collect();torch.cuda.empty_cache()
    assert results[0]==results[1]
    p7.dump(out,dict(status='PASS',seeds=[19,1],exact_initial_state_equal=initial,actual_two_optimizer_steps_equal=True,all_trainable_gradients_equal=True,all_adapted_parameters_equal=True,all_optimizer_states_equal=True,smoke_membership=256,optimizer_images=ids,fold1_only=True))


if __name__=='__main__':main()
