import importlib.util,math
from pathlib import Path
import torch

def api():
    p=Path(__file__).with_name('probe_point_readout.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('pointread',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_relative_quadratic_logits_have_known_native_pixel_values():
    m=api();params=torch.zeros(1,133);params[0,128]=2;params[0,131:]=math.log(math.expm1(32))
    logits=m.point_logits(params,torch.zeros(128,3,3),torch.tensor([[1,1]]))
    assert torch.allclose(logits[0],torch.tensor([[1.9375,1.96875,1.9375],[1.96875,2.,1.96875],[1.9375,1.96875,1.9375]]))

def test_local_readout_uses_legal_point_features_and_explicit_coordinates():
    m=api();f=torch.arange(64*3*3).reshape(64,3,3).float();points=torch.tensor([[0,2],[2,0]])
    x=m.point_inputs(f,points,torch.zeros(64),torch.ones(64))
    assert torch.equal(x[:,:64],torch.stack((f[:,0,2],f[:,2,0])))
    assert torch.equal(x[:,64:],torch.tensor([[1.,-1.],[-1.,1.]]))

def test_shared_readout_receives_finite_gradients_with_frozen_features():
    m=api();torch.manual_seed(19);head=m.PointReadout();f=torch.randn(64,3,3);basis=torch.randn(128,3,3);old=basis.clone();point=torch.tensor([[1,1]])
    params=head(m.point_inputs(f,point,torch.zeros(64),torch.ones(64)));z=m.point_logits(params,basis,point)
    truth=torch.zeros(1,3,3);truth[0,1,1]=1;loss=m.native_mask_loss(z,truth);loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert basis.grad is None and f.grad is None and torch.equal(basis,old)
