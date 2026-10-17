import importlib.util
from pathlib import Path
import torch

def api():
    p=Path(__file__).with_name('probe_context_readout.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('ctx',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_descriptor_has_ordered_nine_neighbours_and_point_coordinates():
    m=api();f=torch.arange(25).reshape(1,5,5).expand(64,-1,-1).float();p=torch.tensor([[2,2]])
    x=m.point_inputs(f,p,torch.zeros(576),torch.ones(576))
    assert x.shape==(1,578)
    assert torch.equal(x[0,:576].reshape(9,64)[:,0],torch.tensor([0.,2.,4.,10.,12.,14.,20.,22.,24.]))
    assert torch.equal(x[0,576:],torch.zeros(2))

def test_boundary_sampling_clamps_and_never_wraps():
    m=api();f=torch.arange(25).reshape(1,5,5).expand(64,-1,-1).float();p=torch.tensor([[0,0]])
    x=m.point_inputs(f,p,torch.zeros(576),torch.ones(576))
    assert torch.equal(x[0,:576].reshape(9,64)[:,0],torch.tensor([0.,0.,4.,0.,0.,4.,20.,20.,24.]))
    assert torch.equal(x[0,576:],torch.tensor([-1.,-1.]))

def test_context_readout_has_finite_gradients_and_fixed_basis():
    m=api();torch.manual_seed(19);head=m.PointReadout();f=torch.randn(64,5,5);basis=torch.randn(128,5,5);old=basis.clone();p=torch.tensor([[2,2]])
    z=m.point_logits(head(m.point_inputs(f,p,torch.zeros(576),torch.ones(576))),basis,p)
    truth=torch.zeros(1,5,5);truth[0,2,2]=1;m.native_mask_loss(z,truth).backward()
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in head.parameters())
    assert torch.equal(basis,old) and basis.grad is None and f.grad is None
