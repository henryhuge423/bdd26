"""Fit-only real-V100 full original parity and input-scale learning witness."""
import gc,json,math,sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p20 as p20
import verify_p18_control as v

def validate_learning(gradients,changes,adaptive):
    expected={'block12','block23','output'}
    if any(set(row)!=expected for row in (gradients,changes,adaptive)):
        raise ValueError('incomplete P20 second-step witness')
    if not all(math.isfinite(x) and x>0 for row in (gradients,changes,adaptive) for x in row.values()):
        raise ValueError('invalid P20 second-step witness')
    return True

def run_u(model,fold,ids,seed):
    opt=v.reference_optimizer(model);frozen=p20.frozen_hashes(model);model.train()
    watched=dict(block12=model.encoder.blocks[12].attn.qkv.weight,
        block23=model.encoder.blocks[23].attn.qkv.weight,output=model.head.out.weight)
    geometry=[];dtype=[]
    def observe(module,inputs,output):
        assert tuple(inputs[0].shape[1:])==(3,512,512)
        assert len(output[1])==4 and all(tuple(t.shape[1:])==(1025,1024) for t in output[1])
        geometry.append(dict(input=list(inputs[0].shape),taps=[list(t.shape) for t in output[1]]))
    def precision(module,inputs,output):dtype.append(str(output.dtype))
    handles=[model.encoder.register_forward_hook(observe),model.head.out.register_forward_hook(precision)]
    for start in (0,16):
        selected=ids[start:start+16];opt.zero_grad(set_to_none=True)
        count=sum(len(fold.gt[i]) for i in selected)
        for begin in range(0,16,4):
            x,y,n=fold.batch(selected[begin:begin+4],seed,0,True)
            rx,ry,rn=original.Fold.batch(fold,selected[begin:begin+4],seed,0,True)
            assert torch.equal(x,rx) and torch.equal(y,ry) and n==rn
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x.cuda());assert torch.is_autocast_enabled('cuda')
            assert logits.shape==(4,256,256) and logits.dtype==torch.float32
            y=y.cuda();loss=p20.loss_for_microbatch(logits,y,count,'U')
            ref=original.focal_numerator(logits,y)/max(1,count)
            assert torch.equal(loss,ref)
            torch.testing.assert_close(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(ref,logits,retain_graph=True)[0],rtol=0,atol=0)
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        if start==16:
            gradients={k:float(p.grad.norm()) for k,p in watched.items()}
            before={k:p.detach().cpu().clone() for k,p in watched.items()}
            lrs={k:next(g['lr'] for g in opt.param_groups if any(p is q for q in g['params'])) for k,p in watched.items()}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
    for handle in handles:handle.remove()
    changes,adaptive=v.measure_k_step(before,{k:p.detach().cpu() for k,p in watched.items()},lrs,.01)
    validate_learning(gradients,changes,adaptive)
    assert p20.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    assert len(geometry)==8 and dtype==['torch.bfloat16']*8
    return dict(gradients=gradients,changes=changes,adaptive_changes=adaptive,second_step_only=True,
        all_trainable_finite=True,AMP_verified=True,actual_data_loss_gradient_equal=True,all_frozen_unchanged=True,
        actual_encoder_geometry=geometry,output_dtype='torch.float32')

def main():
    weights,data,out=sys.argv[1:];torch.set_num_threads(4)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p20.Path(data)/name)==digest
    assert original.sha(p20.Path(data)/'fold1/tissue.npy')==p20.TISSUE_SHA
    fold=p20.Fold(data,1,'C');reference=original.Fold(data,1)
    fit,audit=p20.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt]);p20.validate_cohort(fit,audit)
    smoke_fit=list(fit[:134]);parity={};initial={};measured={}
    old=v.p18;v.p18=p20
    try:
        for seed in (19,1):
            signatures=[]
            for arm in ('P3','C','U'):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p20.Proposal(weights,True,arm))
                signatures.append(dict(state=v.hashes(model.state_dict().items()),rng=v.rng_hashes()));del model;gc.collect()
            assert signatures[0]==signatures[1]==signatures[2];initial[str(seed)]=True
            ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke_fit).tolist()
            assert not set(ids)&set(audit)
            results=[]
            for arm,f in (('P3',reference),('C',fold)):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p20.Proposal(weights,True,arm)).cuda()
                results.append(v.run(model,f,ids[:32],seed,None if arm=='P3' else 'C'))
                del model;gc.collect();torch.cuda.empty_cache()
            assert results[0]==results[1];parity[str(seed)]=True
            v.reset(seed);model=p20.Proposal(weights,True,'U').cuda()
            measured[str(seed)]=run_u(model,fold,ids[:32],seed)
            del model;gc.collect();torch.cuda.empty_cache()
    finally:v.p18=old
    p20.dump(out,dict(status='PASS',seeds=[19,1],initial_P3_C_U_state_RNG_equal=initial,
        two_step_parity_by_seed=parity,C_full_gradient_parameter_AdamW_RNG_equal=True,U_by_seed=measured,
        smoke_fit=smoke_fit,fold1_only=True,no_audit_predictions=True))

if __name__=='__main__':main()
