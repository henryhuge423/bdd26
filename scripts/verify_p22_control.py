"""Fit-only full original-C parity and full-depth (M) learning witnesses for P22."""
import gc,json,math,sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p22 as p22
import verify_p18_control as v

WATCH=('block0','block11','block12','block23','output')

def validate_learning(gradients,changes,adaptive):
    expected=set(WATCH)
    if any(not isinstance(row,dict) or set(row)!=expected for row in (gradients,changes,adaptive)):
        raise ValueError('incomplete P22 second-step witness')
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for row in (gradients,changes,adaptive) for x in row.values()):
        raise ValueError('invalid P22 update or decay-only witness')
    return True

def watched_tensors(model):
    b=model.encoder.blocks
    return dict(block0=b[0].attn.qkv.weight,block11=b[11].attn.qkv.weight,block12=b[12].attn.qkv.weight,block23=b[23].attn.qkv.weight,output=model.head.out.weight)

def run_m(model,fold,ids,seed):
    opt=v.reference_optimizer(model)
    assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-5]
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    encoder_group={id(p) for p in opt.param_groups[1]['params']}
    assert all(id(p) in encoder_group for b in model.encoder.blocks for p in b.parameters())
    assert not any(p.requires_grad for n,p in model.encoder.named_parameters() if not n.startswith('blocks.'))
    watched=watched_tensors(model)
    frozen=p22.frozen_hashes(model);assert frozen and not any(n.startswith('blocks.') for n in frozen)
    model.train();precision={};handles=[]
    for key,module in [('encoder',model.encoder.blocks[23].attn.qkv),('block0',model.encoder.blocks[0].attn.qkv),('output',model.head.out)]:
        handles.append(module.register_forward_hook(lambda m,a,o,key=key:precision.update({key:str(o.dtype)})))
    for start in (0,16):
        selected=ids[start:start+16];count=sum(len(fold.gt[i]) for i in selected);opt.zero_grad(set_to_none=True)
        for begin in range(0,16,4):
            batch=selected[begin:begin+4];x,y,n=fold.batch(batch,seed,0,True)
            rx,ry,rn=original.Fold.batch(fold,batch,seed,0,True)
            assert torch.equal(x,rx) and torch.equal(y,ry) and n==rn
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x.cuda());assert torch.is_autocast_enabled('cuda')
            assert logits.shape==(4,256,256) and logits.dtype==torch.float32
            assert precision==dict(encoder='torch.bfloat16',block0='torch.bfloat16',output='torch.bfloat16')
            y=y.cuda();ref=original.focal_numerator(logits,y)/max(1,count)
            loss=p22.loss_for_microbatch(logits,y,count,'M');assert torch.equal(loss,ref)
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        if start==16:
            gradients={k:float(p.grad.float().norm()) for k,p in watched.items()}
            before={k:p.detach().cpu().clone() for k,p in watched.items()}
            lrs={k:next(g['lr'] for g in opt.param_groups if any(p is q for q in g['params'])) for k,p in watched.items()}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
    for h in handles:h.remove()
    changes,adaptive=v.measure_k_step(before,{k:p.detach().cpu() for k,p in watched.items()},lrs,.01)
    validate_learning(gradients,changes,adaptive)
    assert lrs=={k:(3e-4 if k=='output' else 1e-5) for k in WATCH},lrs
    assert p22.frozen_hashes(model)==frozen and all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return dict(gradients=gradients,changes=changes,adaptive_changes=adaptive,second_step_only=True,all_trainable_finite=True,AMP_verified=True,all_frozen_unchanged=True,exact_optimizer_membership=True,all24_blocks_in_encoder_group=True,encoder_lr=1e-5,trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),frozen_encoder_tensors=len(frozen))

def main():
    weights,data,out=sys.argv[1:];torch.set_num_threads(4)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85)
    assert original.sha(weights)==original.WEIGHT_SHA
    for name,digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):assert original.sha(p22.Path(data)/name)==digest
    assert original.sha(p22.Path(data)/'fold1/tissue.npy')==p22.TISSUE_SHA
    fold=p22.Fold(data,1,'C');reference=original.Fold(data,1);mfold=p22.Fold(data,1,'M')
    fit,audit=p22.split_indices(fold.tissue,[any(g[0]==4 and g[2] for g in row) for row in fold.gt]);p22.validate_cohort(fit,audit)
    smoke=p22.smoke_ids(fit,fold.gt)
    parity={};initial={};learning={};depth={}
    previous=v.p18;v.p18=p22
    try:
        for seed in (19,1):
            signatures=[];trainable_sets=[]
            for arm in ('P3','C','M'):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p22.Proposal(weights,True,arm))
                signatures.append(dict(state=v.hashes(model.state_dict().items()),rng=v.rng_hashes()))
                trainable_sets.append(sorted(n for n,p in model.named_parameters() if p.requires_grad))
                del model;gc.collect()
            assert signatures[0]==signatures[1]==signatures[2];initial[str(seed)]=True
            blocks=lambda names:sorted({int(n.split('.')[2]) for n in names if n.startswith('encoder.blocks.')})
            assert trainable_sets[0]==trainable_sets[1] and set(trainable_sets[1])<set(trainable_sets[2])
            assert blocks(trainable_sets[1])==list(range(12,24)) and blocks(trainable_sets[2])==list(range(24))
            assert all(n.startswith('encoder.blocks.') or n.startswith('head.') for n in trainable_sets[2])
            depth[str(seed)]=dict(C=len(trainable_sets[1]),M=len(trainable_sets[2]))
            ids=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(smoke).tolist();assert not set(ids)&set(audit)
            results=[]
            for arm,f in (('P3',reference),('C',fold)):
                v.reset(seed);model=(original.Proposal(weights,True) if arm=='P3' else p22.Proposal(weights,True,arm)).cuda()
                results.append(v.run(model,f,ids[:32],seed,None if arm=='P3' else 'C'))
                del model;gc.collect();torch.cuda.empty_cache()
            assert results[0]==results[1];parity[str(seed)]=True
            v.reset(seed);model=p22.Proposal(weights,True,'M').cuda();learning[str(seed)]=run_m(model,mfold,ids[:32],seed)
            del model;gc.collect();torch.cuda.empty_cache()
    finally:v.p18=previous
    p22.dump(out,dict(status='PASS',seeds=[19,1],initial_full_state_RNG_equal=initial,two_step_parity_by_seed=parity,C_full_gradient_parameter_AdamW_RNG_equal=True,M_by_seed=learning,trainable_counts_by_seed=depth,smoke_fit=smoke,fold1_only=True,no_audit_predictions=True))

if __name__=='__main__':main()
