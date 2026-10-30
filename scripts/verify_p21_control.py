"""Fit-only full original-C parity and semantic auxiliary learning/feature witnesses."""
import gc,json,math,sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p21 as p21
import verify_p18_control as v

def validate_learning(gradients,changes,adaptive,auxiliary_feature_gradient):
    expected={'block12','block23','output','auxiliary'}
    if any(set(row)!=expected for row in (gradients,changes,adaptive)):
        raise ValueError('incomplete P21 second-step witness')
    if not all(math.isfinite(x) and x>0 for row in (gradients,changes,adaptive) for x in row.values()):
        raise ValueError('invalid P21 update or decay-only witness')
    if not math.isfinite(auxiliary_feature_gradient) or auxiliary_feature_gradient<=0:
        raise ValueError('auxiliary-only loss did not reach shared features')
    return True

def run_a(model,fold,ids,seed):
    opt=v.reference_optimizer(model)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    assert any(model.head.auxiliary.weight is q for q in opt.param_groups[0]['params'])
    watched=dict(block12=model.encoder.blocks[12].attn.qkv.weight,block23=model.encoder.blocks[23].attn.qkv.weight,output=model.head.out.weight,auxiliary=model.head.auxiliary.weight)
    frozen=p21.frozen_hashes(model);model.train();precision={};handles=[];feature_gradient=0.
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('output',model.head.out),('auxiliary',model.head.auxiliary)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    counts=[0]*5
    for start in (0,16):
        selected=ids[start:start+16];count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            batch=selected[begin:begin+4];x,y,n=fold.batch(batch,seed,0,True)
            rx,ry,rn=original.Fold.batch(fold,batch,seed,0,True)
            assert torch.equal(x,rx) and torch.equal(y[:,0],ry) and n==rn
            counts=[a+int(b) for a,b in zip(counts,(y[:,1:]==1).sum((0,2,3)).tolist())]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,aux=model.forward_with_aux(x.cuda());assert torch.is_autocast_enabled('cuda')
            assert logits.shape==(4,256,256) and aux.shape==(4,5,256,256)
            assert logits.dtype==aux.dtype==torch.float32
            assert precision==dict(encoder='torch.bfloat16',output='torch.bfloat16',auxiliary='torch.bfloat16')
            y=y.cuda();main=original.focal_numerator(logits,y[:,0])/max(1,count)
            auxiliary=original.focal_numerator(aux,y[:,1:])/max(1,count)
            loss=p21.loss_for_microbatch(logits,y,count,'A',aux)
            torch.testing.assert_close(loss,main+auxiliary,rtol=2e-7,atol=1e-7)
            if start==16:
                g=torch.autograd.grad(auxiliary,model.head.stem.weight,retain_graph=True)[0]
                feature_gradient=max(feature_gradient,float(g.norm()))
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        if start==16:
            gradients={k:float(p.grad.norm()) for k,p in watched.items()}
            before={k:p.detach().cpu().clone() for k,p in watched.items()}
            lrs={k:next(g['lr'] for g in opt.param_groups if any(p is q for q in g['params'])) for k,p in watched.items()}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
    for h in handles:h.remove()
    changes,adaptive=v.measure_k_step(before,{k:p.detach().cpu() for k,p in watched.items()},lrs,.01)
    validate_learning(gradients,changes,adaptive,feature_gradient)
    assert p21.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    model.eval();x,_,_=fold.batch(ids[:2],seed,0,False);calls=[]
    h=model.head.auxiliary.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        before=model(x.cuda());model.head.auxiliary.weight.add_(10);model.head.auxiliary.bias.add_(10);after=model(x.cuda())
    h.remove();assert not calls and torch.equal(before,after)
    return dict(gradients=gradients,changes=changes,adaptive_changes=adaptive,auxiliary_only_shared_feature_gradient=feature_gradient,second_step_only=True,all_trainable_finite=True,AMP_verified=True,all_frozen_unchanged=True,inference_excludes_auxiliary=True,exact_optimizer_membership=True,auxiliary_optimizer_lr=3e-4,observed_class_positive_pixels=counts)

def main():
    weights,data,out=sys.argv[1:];torch.set_num_threads(4)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p21.Path(data)/name)==digest
    assert original.sha(p21.Path(data)/'fold1/tissue.npy')==p21.TISSUE_SHA
    fold=p21.Fold(data,1,'C');reference=original.Fold(data,1);afold=p21.Fold(data,1,'A')
    fit,audit=p21.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt]);p21.validate_cohort(fit,audit)
    smoke=p21.smoke_ids(fit,fold.gt)
    class_counts=[sum(sum(g[0]==c for g in fold.gt[i]) for i in smoke) for c in range(1,6)]
    assert all(v>0 for v in class_counts)
    parity={};initial={};learning={}
    previous=v.p18;v.p18=p21
    try:
        for seed in (19,1):
            signatures=[]
            for arm in ('P3','C','A'):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p21.Proposal(weights,True,arm))
                signatures.append(dict(state=v.hashes((n,t) for n,t in model.state_dict().items() if not n.startswith('head.auxiliary.')),rng=v.rng_hashes()))
                del model;gc.collect()
            assert signatures[0]==signatures[1]==signatures[2];initial[str(seed)]=True
            ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke).tolist();assert not set(ids)&set(audit)
            results=[]
            for arm,f in (('P3',reference),('C',fold)):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p21.Proposal(weights,True,arm)).cuda()
                results.append(v.run(model,f,ids[:32],seed,None if arm=='P3' else 'C'))
                del model;gc.collect();torch.cuda.empty_cache()
            assert results[0]==results[1];parity[str(seed)]=True
            v.reset(seed);model=p21.Proposal(weights,True,'A').cuda();learning[str(seed)]=run_a(model,afold,ids[:32],seed)
            del model;gc.collect();torch.cuda.empty_cache()
    finally:v.p18=previous
    p21.dump(out,dict(status='PASS',seeds=[19,1],initial_original_parameter_RNG_equal=initial,two_step_parity_by_seed=parity,C_full_gradient_parameter_AdamW_RNG_equal=True,A_by_seed=learning,smoke_fit=smoke,smoke_class_object_counts=class_counts,fold1_only=True,no_audit_predictions=True))

if __name__=='__main__':main()
