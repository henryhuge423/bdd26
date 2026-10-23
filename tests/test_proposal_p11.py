"""Independent behavioral contracts for P11 positive-mass redistribution."""
import math
import numpy as np
import pytest
import torch
from proposal_p3_utils import focal_numerator,gaussian_target
import train_proposal_p11 as p11

def target_cases():
    objects=[(1,(6,6),True,10),(4,(0,0),False,3),(1,(6,7),True,5),(1,(6,6),True,8)]
    return torch.from_numpy(np.stack([gaussian_target(objects[:1],13,13),gaussian_target(objects[1:2],13,13),gaussian_target(objects,13,13),np.zeros((13,13),np.float32)]))

def reference(z,y,weights,den):
    value=0.;gradient=[]
    for zz,yy,aa in zip(z.flatten().tolist(),y.flatten().tolist(),weights.flatten().tolist()):
        p=1/(1+math.exp(-zz));q=1-p
        lp=-float(np.logaddexp(0,-zz));lq=-float(np.logaddexp(0,zz))
        b=(1-yy)**4 if yy<1 else 0
        value+=-aa*lp*q*q-b*lq*p*p
        gradient.append((aa*(2*p*q*q*lp-q**3)+b*(p**3-2*p*p*q*lq))/den)
    return value/den,torch.tensor(gradient).reshape(z.shape)

def test_mass_is_conserved_for_boundary_overlap_duplicate_and_empty():
    y=target_cases();w=p11.positive_weights(y)
    assert torch.allclose(w.sum((-2,-1)),(y==1).sum((-2,-1)).float(),atol=2e-6)
    assert w.dtype==torch.float32 and not w[-1].any()
    assert w[0,6,7]>0 and w[0,6,6]<1
    assert torch.equal(w[y==0],torch.zeros_like(w[y==0]))

@pytest.mark.parametrize('arm',['C','G'])
def test_loss_and_gradient_match_independent_scalar_equation(arm):
    y=target_cases();z=torch.linspace(-8,8,y.numel()).reshape(y.shape).requires_grad_()
    w=(y==1).float() if arm=='C' else y*(y==1).sum((-2,-1),keepdim=True)/y.sum((-2,-1),keepdim=True).clamp_min(1)
    expected,grad=reference(z.detach(),y,w,7)
    actual=p11.loss_for_microbatch(z,y,7,arm);actual.backward()
    assert float(actual)==pytest.approx(expected,rel=2e-6,abs=1e-6)
    assert torch.allclose(z.grad,grad,rtol=3e-5,atol=2e-7)

def test_control_loss_and_all_gradients_are_exact_original():
    y=target_cases();z=torch.randn_like(y,requires_grad=True);other=z.detach().clone().requires_grad_()
    a=p11.loss_for_microbatch(z,y,7,'C');b=focal_numerator(other,y)/7
    a.backward();b.backward();assert torch.equal(a,b) and torch.equal(z.grad,other.grad)

def test_binary_targets_and_empty_images_preserve_original():
    y=torch.zeros(2,5,5);y[0,2,2]=1;z=torch.randn_like(y)
    assert torch.equal(p11.loss_for_microbatch(z,y,1,'G'),p11.loss_for_microbatch(z,y,1,'C'))
    y.zero_();assert torch.equal(p11.loss_for_microbatch(z,y,0,'G'),focal_numerator(z,y))

def test_empty_target_keeps_negative_term_and_extreme_logits_are_finite():
    y=torch.tensor([[[0.,.6,1.,0.]]]);z=torch.tensor([[[-80.,80.,-80.,80.]]],requires_grad=True)
    v=p11.loss_for_microbatch(z,y,2,'G');v.backward()
    assert torch.isfinite(v) and torch.isfinite(z.grad).all()
    assert z.grad[0,0,2]<0 and z.grad[0,0,3]>0

def test_gaussian_shoulders_add_positive_gradient_without_changing_negative():
    y=target_cases()[:1];z=torch.full_like(y,-2.,requires_grad=True)
    c=p11.loss_for_microbatch(z,y,1,'C');gc=torch.autograd.grad(c,z,retain_graph=True)[0]
    g=p11.loss_for_microbatch(z,y,1,'G');gg=torch.autograd.grad(g,z)[0]
    assert gg[0,6,7]<gc[0,6,7]
    assert torch.equal(gg[y==0],gc[y==0])

@pytest.mark.parametrize('arm',['C','G'])
def test_true_tail_batch_and_microbatch_accumulation(arm):
    y=target_cases();z=torch.randn_like(y,requires_grad=True)
    full=p11.loss_for_microbatch(z,y,5,arm)
    parts=sum(p11.loss_for_microbatch(z[i:i+1],y[i:i+1],5,arm) for i in range(4))
    assert torch.allclose(full,parts,rtol=1e-6)
    assert torch.allclose(torch.autograd.grad(full,z,retain_graph=True)[0],torch.autograd.grad(parts,z)[0],rtol=1e-6)

@pytest.mark.parametrize('bad',[True,-1,1.5,None])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):p11.loss_for_microbatch(torch.zeros(1,2,2),torch.zeros(1,2,2),bad,'G')

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-0.1,1.1])
def test_invalid_target_rejected(bad):
    y=torch.tensor([[[bad]]])
    with pytest.raises(ValueError):p11.loss_for_microbatch(torch.zeros_like(y),y,0,'G')

def test_invalid_shapes_and_arm_rejected():
    for z,y,a in [(torch.zeros(2,2),torch.zeros(2,2),'G'),(torch.zeros(0,2,2),torch.zeros(0,2,2),'G'),(torch.zeros(1,2,2),torch.zeros(2,2,2),'G'),(torch.zeros(1,2,2),torch.zeros(1,2,2),'S')]:
        with pytest.raises(ValueError):p11.loss_for_microbatch(z,y,1,a)

def test_config_and_checkpoint_are_arm_bound():
    c=p11.run_config('C',19);g=p11.run_config('G',19)
    assert c['trainable_encoder_blocks']==g['trainable_encoder_blocks']==list(range(12,24))
    assert c['positive_supervision']=='exact_center' and g['positive_supervision']=='gaussian_mass_redistribution'
    p11.validate_config(g,g)
    with pytest.raises(ValueError):p11.validate_config(c,g)
    for a,s in [('G',2),('S',19),('C',True)]:
        with pytest.raises(ValueError):p11.run_config(a,s)

def test_cli_rejects_expansion():
    base=['--data','d','--weights','w','--out','o','--arm','G']
    assert p11.parse_args(base).seed==19
    for extra in [['--phase','B'],['--seed','2'],['--arm','S']]:
        with pytest.raises(SystemExit):p11.parse_args(base+extra)
