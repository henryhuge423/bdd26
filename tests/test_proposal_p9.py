import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_proposal_p9 as p9
import train_proposal_p3 as p3

def test_analytical_dead_positive_value_and_gradient():
 x=torch.tensor([[[.3,-.7,.8,-.2]]],requires_grad=True);y=torch.tensor([[[1.,1.,0.,.5]]]);w=torch.tensor([[[4.,1.,4.,4.]]])
 loss=p9.loss_for_microbatch(x,y,w,2,'D');p=x.sigmoid()
 expected=-(4*F.logsigmoid(x[0,0,0])*(1-p[0,0,0])**2+F.logsigmoid(x[0,0,1])*(1-p[0,0,1])**2+F.logsigmoid(-x[0,0,2])*p[0,0,2]**2+F.logsigmoid(-x[0,0,3])*p[0,0,3]**2*.5**4)/2
 assert torch.allclose(loss,expected)
 assert torch.allclose(torch.autograd.grad(loss,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])

@pytest.mark.parametrize('arm',['C','D'])
@pytest.mark.parametrize('empty',[True,False])
def test_no_dead_exact_original(arm,empty):
 x=torch.linspace(-2,2,18).reshape(2,3,3).requires_grad_();y=torch.zeros_like(x)
 if not empty:y[0,1,1]=1
 den=0 if empty else 3;w=torch.ones_like(x)
 a=p9.loss_for_microbatch(x,y,w,den,arm);b=p3.focal_numerator(x,y)/max(1,den)
 assert torch.equal(a,b)
 assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])

def test_control_bypasses_weights_bitwise():
 x=torch.randn(2,3,3,requires_grad=True);y=torch.zeros_like(x);y[:,1,1]=1;w=torch.full_like(x,4)
 a=p9.loss_for_microbatch(x,y,w,3,'C');b=p3.focal_numerator(x,y)/3
 assert torch.equal(a,b)
 assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])

@pytest.mark.parametrize('arm',['C','D'])
def test_tail_microbatch_accumulation(arm):
 x=torch.randn(3,2,3,requires_grad=True);y=torch.zeros_like(x);y[0,0,0]=1;y[1,1,1]=1
 w=torch.ones_like(x);w[1,1,1]=4
 full=p9.loss_for_microbatch(x,y,w,3,arm)
 split=sum(p9.loss_for_microbatch(x[a:b],y[a:b],w[a:b],3,arm) for a,b in [(0,2),(2,3)])
 assert torch.allclose(full,split)
 assert torch.allclose(torch.autograd.grad(full,x,retain_graph=True)[0],torch.autograd.grad(split,x)[0])

def test_border_collision_maps():
 gt=[(4,(0,0),False,4),(1,(0,0),False,9),(4,(0,0),False,5),(4,(3,4),False,5),(2,(2,2),True,4)]
 w=p9.positive_weights(gt,4,5)
 assert w.shape==(4,5) and w.dtype==torch.float32
 assert w[0,0]==w[3,4]==4 and w[2,2]==1 and (w==4).sum()==2

@pytest.mark.parametrize('aug',[False,True])
@pytest.mark.parametrize('seed',[19,1])
def test_batch_and_weight_geometry_rng_parity(aug,seed):
 fold=object.__new__(p9.Fold);fold.images=np.arange(2*256*256*3,dtype=np.uint8).reshape(2,256,256,3)
 fold.gt=[[(4,(0,255),False,4),(1,(30,70),True,8)],[]]
 reference=object.__new__(p3.Fold);reference.images=fold.images;reference.gt=fold.gt
 a=fold.batch([0,1],seed,3,aug);b=reference.batch([0,1],seed,3,aug)
 assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
 torch_state=torch.get_rng_state();np_state=np.random.get_state()
 w=fold.positive_weights([0,1],seed,3,aug)
 assert torch.equal(torch.get_rng_state(),torch_state)
 assert all(np.array_equal(x,y) for x,y in zip(np.random.get_state(),np_state))
 expected=p9.positive_weights(fold.gt[0])
 if aug:
  rng=np.random.default_rng(np.random.SeedSequence([seed,3,0,987]));_,expected=p3.augment_pair(torch.zeros(3,256,256),expected,rng)
 assert torch.equal(w[0],expected) and (w[1]==1).all()
 assert (a[1][w==4]==1).all()

@pytest.mark.parametrize('bad',[-1,True,1.5])
def test_invalid_denominator(bad):
 with pytest.raises(ValueError):p9.loss_for_microbatch(torch.zeros(1,2,2),torch.zeros(1,2,2),torch.ones(1,2,2),bad,'D')

@pytest.mark.parametrize('value',[0.,2.,float('nan')])
def test_invalid_weights(value):
 with pytest.raises(ValueError):p9.loss_for_microbatch(torch.zeros(1,2,2),torch.zeros(1,2,2),torch.full((1,2,2),value),1,'D')

def test_shapes_and_arm_rejected():
 for x,y,w,arm in [(torch.zeros(1,2,2),torch.zeros(1,2,2),torch.ones(2,2),'D'),(torch.zeros(2,2),torch.zeros(2,2),torch.ones(2,2),'D'),(torch.zeros(1,2,2),torch.zeros(1,2,2),torch.ones(1,2,2),'I')]:
  with pytest.raises(ValueError):p9.loss_for_microbatch(x,y,w,1,arm)

def test_config_cli_fold_contract():
 assert p9.run_config('D',19)['positive_weight']['dead']==4
 assert p9.run_config('C',1)['positive_weight']['dead']==1
 p9.validate_config(p9.run_config('D',1),p9.run_config('D',1))
 with pytest.raises(ValueError):p9.validate_config(p9.run_config('C',1),p9.run_config('D',1))
 for arm,seed in [('I',1),('D',2)]:
  with pytest.raises(ValueError):p9.run_config(arm,seed)
 for fold,arm in [(2,'C'),(1,'I')]:
  with pytest.raises(ValueError):p9.Fold('/not/read',fold,arm)
 with pytest.raises(SystemExit):p9.parse_args(['--data','x','--weights','x','--out','x','--arm','I'])
