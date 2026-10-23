"""Same-GPU real UNI parity: P11 C versus original P3 T1, no audit output."""
import gc
import hashlib
import json
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p11 as p11


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


def run(model,fold,ids,use_p11):
    opt=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5)],betas=(.9,.999),weight_decay=.01)
    traces=[];model.train()
    for start in (0,16):
        selected=ids[start:start+16];den=max(1,sum(len(fold.gt[i]) for i in selected));opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            x,y,_=fold.batch(selected[begin:begin+4],19,0,True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda())
            loss=(p11.loss_for_microbatch(logits,y.cuda(),sum(len(fold.gt[i]) for i in selected),'C') if use_p11 else original.focal_numerator(logits,y.cuda())/den);loss.backward()
        gradients=hashes((n,p.grad) for n,p in model.named_parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
        traces.append(dict(gradients=gradients,optimizer_state=optimizer_hash(opt.state_dict())))
    return hashes(model.adapted_state().items()),traces


def verify_gaussian_gpu():
    # Independent numpy FP64 equation/analytic derivative; production uses FP32.
    from proposal_p3_utils import gaussian_target
    objects=[(1,(6,6),True,10),(4,(0,0),False,3),(1,(6,7),True,5),(1,(6,6),True,8)]
    y=np.stack([gaussian_target(objects,13,13),gaussian_target(objects[1:2],13,13),np.zeros((13,13),np.float32)])
    z=np.linspace(-12,12,y.size,dtype=np.float32).reshape(y.shape);z[0,0,0]=-80;z[0,-1,-1]=80
    yd=y.astype(np.float64);zd=z.astype(np.float64)
    a=yd*(yd==1).sum((-2,-1),keepdims=True)/np.maximum(yd.sum((-2,-1),keepdims=True),1)
    p=1/(1+np.exp(-zd));q=1-p;lp=-np.logaddexp(0,-zd);lq=-np.logaddexp(0,zd);b=(1-yd)**4*(yd<1)
    value=-(a*lp*q*q+b*lq*p*p).sum()/7
    grad=(a*(2*p*q*q*lp-q**3)+b*(p**3-2*p*p*q*lq))/7
    zz=torch.tensor(z,device='cuda',requires_grad=True);yy=torch.tensor(y,device='cuda')
    loss=p11.loss_for_microbatch(zz,yy,7,'G');loss.backward()
    assert np.isclose(float(loss),value,rtol=2e-6,atol=1e-5)
    assert np.allclose(zz.grad.cpu().numpy(),grad,rtol=3e-5,atol=2e-7)
    w=p11.positive_weights(yy)
    assert torch.allclose(w.sum((-2,-1)),(yy==1).sum((-2,-1)).float(),rtol=1e-5,atol=1e-4)
    # Positions outside support have the identical original negative gradients.
    cc=zz.detach().clone().requires_grad_();p11.loss_for_microbatch(cc,yy,7,'C').backward()
    assert torch.equal(cc.grad[yy==0],zz.grad[yy==0])
    return dict(gaussian_gpu_loss_gradient_equal=True,positive_mass_conserved=True,outside_support_negative_gradients_equal=True,gradient_rtol=3e-5,gradient_atol=2e-7,mass_rtol=1e-5,mass_atol=1e-4)


def main():
    torch.set_num_threads(4);weights,data,out=sys.argv[1:]
    gaussian=verify_gaussian_gpu()
    initial={}
    for seed in (19,1):
        states=[]
        for arm in ('C','G'):
            torch.manual_seed(seed);model=p11.Proposal(weights,True,arm)
            states.append(hashes(model.state_dict().items()));del model;gc.collect()
        assert states[0]==states[1];initial[str(seed)]=True
    fold=p11.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p11.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    p11.validate_cohort(fit,audit)
    # Match the P11 smoke membership and epoch0 ordering exactly, not full-cohort shuffle.
    smoke_fit=list(fit[:256])
    ids=np.random.default_rng(np.random.SeedSequence([19,0,123])).permutation(smoke_fit).tolist()[:32]
    results=[]
    for implementation,f in ((original,reference),(p11,fold)):
        torch.manual_seed(19);model=implementation.Proposal(weights,True).cuda()
        results.append(run(model,f,ids,implementation is p11));del model;gc.collect();torch.cuda.empty_cache()
    assert results[0]==results[1]
    p11.dump(out,dict(**gaussian,status='PASS',seeds=[19,1],exact_initial_state_equal=initial,actual_two_optimizer_steps_equal=True,all_trainable_gradients_equal=True,all_adapted_parameters_equal=True,all_optimizer_states_equal=True,smoke_membership=256,smoke_fit=smoke_fit,optimizer_images=ids,fold1_only=True,objective_initial_states_equal=True))


if __name__=='__main__':main()
