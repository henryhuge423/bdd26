import importlib.util,math
from pathlib import Path
import torch

def api():
    p=Path(__file__).with_name('probe_predicted_points.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('predpoint',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_center_gaussian_and_focal_have_known_values_and_empty_support():
    m=api();y=m.center_target(torch.tensor([[2,2]]),5,5,'cpu')
    assert y[2,2]==1 and abs(float(y[2,3])-math.exp(-.5))<1e-7
    assert torch.allclose(m.center_loss(torch.zeros(1,1),torch.ones(1,1),1),torch.tensor(.25*math.log(2)))
    loss=m.center_loss(torch.zeros(2,2,requires_grad=True),torch.zeros(2,2),0)
    assert torch.allclose(loss,torch.tensor(math.log(2)));loss.backward()

def test_predicted_peaks_are_stable_bounded_and_empty_safe():
    m=api();z=torch.full((5,5),-20.);z[0,4]=8;z[4,0]=8;z[4,4]=7
    p,s=m.select_points(z,capacity=2)
    assert p.tolist()==[[0,4],[4,0]] and len(s)==2
    assert m.select_points(torch.full((5,5),-20.))[0].shape==(0,2)

def test_gt_free_decoder_preserves_zero_background_and_predicted_class():
    m=api();logits=torch.full((6,33,33),-20.);logits[0,16,16]=8;logits[4,16,16]=10
    class Readout:
        def __call__(self,x):
            p=torch.zeros(len(x),133);p[:,128]=2;p[:,131:]=math.log(math.expm1(32));return p
    inst,typ,aux=m.decode_points(logits,Readout(),torch.zeros(64,33,33),torch.zeros(128,33,33),torch.zeros(576),torch.ones(576))
    assert inst[16,16]==1 and typ[16,16]==4 and inst[0,0]==typ[0,0]==0
    assert aux['points'].tolist()==[[16,16]]

def test_center_head_has_finite_trainable_gradients():
    m=api();h=m.CenterHead();z=h(torch.randn(1,64,9,9))[0];t=m.center_target(torch.tensor([[4,4]]),9,9,'cpu')
    loss=m.center_loss(z[0],t,1)+torch.nn.functional.cross_entropy(z[1:,4,4][None],torch.tensor([3]));loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in h.parameters())

def test_quality_allocation_cannot_pass_with_underfit_or_precision_loss():
    m=api();a={'dead_recall':.2,'recall':.3,'precision':.4,'dead_typed_f1':.2}
    s={**a,'dead_recall':.3,'fit_recall':.9};rows={seed:{'A0':a,'S':dict(s)} for seed in ('19','1')}
    assert m.point_allocation(rows)['promising']
    rows['19']['S']['fit_recall']=.79;assert not m.point_allocation(rows)['promising']
    rows['19']['S']['fit_recall']=.9;rows['1']['S']['precision']=.39;assert not m.point_allocation(rows)['promising']
