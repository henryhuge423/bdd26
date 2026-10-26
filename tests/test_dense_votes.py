import importlib.util,math
from pathlib import Path
import torch

def api():
    p=Path(__file__).resolve().parents[1]/'scripts'/'probe_dense_votes.py'
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('densevote',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_dense_targets_include_every_foreground_pixel_and_singletons():
    from nucseg.setmask.matching import make_targets
    m=api();inst=torch.tensor([[[1,2,2,2,2]]]);typ=torch.tensor([[[4,1,1,1,1]]]);target=make_targets(inst,typ)[0];t=m.dense_targets(target)
    assert t['foreground'].sum()==5 and t['classes'][0,0]==3
    assert torch.allclose(t['offset'][1,0],torch.tensor([0.,1/32,0.,-1/32,-2/32]))
    pred=torch.zeros(8,1,5,requires_grad=True);pred.data[1:3]=t['offset'];pred.data[1,0,0]+=2
    terms=m.dense_losses(pred,t)
    assert torch.allclose(terms['offset'],torch.tensor(.5))
    assert torch.allclose(terms['class'],torch.tensor(math.log(5)))
    sum(terms.values()).backward();assert bool(torch.isfinite(pred.grad).all())

def example():
    p=torch.zeros(8,5,5);p[0]=-20;p[0,2,1]=p[0,2,3]=8;p[2,2,1]=1/32;p[2,2,3]=-1/32;p[6,2,1]=p[6,2,3]=20
    return p

def test_foreground_offsets_vote_back_to_predicted_common_point():
    m=api();points,confidence,classes,votes=m.vote_proposals(example())
    assert points.tolist()==[[2,2]] and classes.tolist()==[4]
    assert votes[2,2]>1.99 and confidence[0]>.99

def test_single_pixel_and_empty_foreground_are_supported_without_area_filter():
    m=api();p=torch.zeros(8,5,5);p[0]=-20;p[0,0,0]=8;p[6,0,0]=20
    assert m.vote_proposals(p)[0].tolist()==[[0,0]]
    p[0,0,0]=-20;assert m.vote_proposals(p)[0].shape==(0,2)

def test_dense_vote_decoder_has_no_gt_and_returns_classed_mask():
    m=api();p=torch.zeros(8,33,33);p[0]=-20;p[0,16,16]=8;p[6,16,16]=20
    class Readout:
        def __call__(self,x):
            v=torch.zeros(len(x),133);v[:,128]=2;v[:,131:]=math.log(math.expm1(32));return v
    inst,typ,aux=m.decode_points(p,Readout(),torch.zeros(64,33,33),torch.zeros(128,33,33),torch.zeros(576),torch.ones(576))
    assert inst[16,16]==1 and typ[16,16]==4 and inst[0,0]==typ[0,0]==0

def test_empty_gt_losses_and_dense_head_gradients_are_finite():
    from nucseg.setmask.matching import make_targets
    m=api();head=m.DenseVoteHead();pred=head(torch.randn(1,64,5,5))[0]
    target=make_targets(torch.zeros(1,5,5,dtype=torch.long),torch.zeros(1,5,5,dtype=torch.long))[0]
    terms=m.dense_losses(pred,m.dense_targets(target));assert terms['offset']==terms['class']==0
    sum(terms.values()).backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
