import importlib.util
from pathlib import Path
import math
import torch

def module():
    p=Path(__file__).with_name('probe_free_masks.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('free_masks',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_dot_product_preserves_original_native_route():
    m=module();e=torch.tensor([[2.,-1.],[0.,3.]])
    f=torch.tensor([[[1.,2.]],[[3.,4.]]])
    assert torch.allclose(m.mask_logits(e,f),torch.tensor([[[-1.,0.]],[[9.,12.]]])/math.sqrt(2))

def test_loss_is_original_five_bce_plus_five_smoothed_dice():
    m=module();z=torch.zeros(1,2,2,requires_grad=True);t=torch.tensor([[[1.,0.],[0.,0.]]])
    loss=m.mask_loss(z,t)
    assert torch.allclose(loss,torch.tensor(5*math.log(2)+2.5))
    loss.backward();assert z.grad[0,0,0]<0 and z.grad[0,0,1]>0

def test_each_embedding_can_fit_its_object_without_feature_mutation():
    m=module();f=torch.tensor([[[1.,-1.],[-1.,-1.]],[[-1.,1.],[-1.,-1.]]])
    original=f.clone();e=torch.nn.Parameter(torch.zeros(2,2));truth=torch.tensor([[[1.,0.],[0.,0.]],[[0.,1.],[0.,0.]]])
    opt=torch.optim.AdamW([e],lr=.1,weight_decay=0)
    for _ in range(60):
        opt.zero_grad();loss=m.mask_loss(m.mask_logits(e,f),truth);loss.backward();opt.step()
    assert torch.equal(f,original) and f.grad is None
    assert torch.equal(m.object_iou(m.mask_logits(e,f),truth),torch.ones(2))

def test_iou_uses_paired_identity_and_strict_positive_mask():
    m=module();z=torch.tensor([[[1.,-1.]],[[1.,1.]]]);t=torch.tensor([[[1.,0.]],[[0.,1.]]])
    assert torch.equal(m.object_iou(z,t),torch.tensor([1.,.5]))
