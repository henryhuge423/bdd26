import importlib.util
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('train_proposal_p20') is not None, 'P20 working-scale implementation absent'
    import train_proposal_p20
    return train_proposal_p20

@pytest.mark.parametrize('arm,size',[('C',256),('U',512)])
def test_frozen_scale_batch_and_config(arm,size):
    p=module();c=p.run_config(arm,19)
    assert c['input_size']==size and c['output_size']==256
    assert c['effective_batch']==16 and c['microbatch']==4
    assert c['trainable_encoder_blocks']==list(range(12,24))
    assert c['target_sigma']==1 and c['target_radius']==3
    assert c['augmentation']==dict(kind='original')
    with pytest.raises(ValueError):p.validate_config(c,p.run_config('U' if arm=='C' else 'C',19))

@pytest.mark.parametrize('arm,seed,micro',[('B',19,4),('U',2,4),('C',True,4),('U',1,16),('U',1,True)])
def test_reject_unfrozen_config(arm,seed,micro):
    with pytest.raises(ValueError):module().run_config(arm,seed,micro)

class TokenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.));self.seen=None
    def forward(self,x):
        self.seen=x.detach().clone();n=x.shape[-1]//16
        t=F.avg_pool2d(x,16).flatten(2).transpose(1,2)
        # Unequal channel functions avoid the layer-norm scale invariance.
        t=torch.cat([t.repeat(1,1,341),t[:,:,:1]],2)*self.scale
        t=t+torch.linspace(-1,1,1024,device=x.device)[None,None]
        t=torch.cat([t.new_zeros(len(t),1,1024),t],1)
        return t[:,0],[t,t,t,t]

class CaptureHead(torch.nn.Module):
    def __init__(self):super().__init__();self.shapes=None
    def forward(self,feats):
        self.shapes=[tuple(x.shape) for x in feats]
        return F.interpolate(feats[0][:,:1],size=(256,256),mode='bilinear',align_corners=False)[:,0]

def synthetic_model(arm,real_head=False):
    p=module();m=p.Proposal.__new__(p.Proposal);torch.nn.Module.__init__(m)
    m.arm=arm;m.encoder=TokenEncoder()
    if real_head:
        from train_proposal_p3 import PointHead
        m.head=PointHead()
        with torch.no_grad():m.head.out.weight.fill_(.01)
    else:m.head=CaptureHead()
    return m

@pytest.mark.parametrize('arm,size',[('C',256),('U',512)])
def test_actual_encoder_input_and_native_output(arm,size):
    m=synthetic_model(arm);x=torch.rand(1,3,256,256)
    result=m(x)
    expected=x if size==256 else F.interpolate(x,size=(512,512),mode='bilinear',align_corners=False)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    torch.testing.assert_close(m.encoder.seen,(expected-mean)/std,rtol=0,atol=0)
    assert m.head.shapes==[(1,1024,size//16,size//16)]*4
    assert result.shape==(1,256,256)
    result.square().mean().backward();assert m.encoder.scale.grad.abs()>0

def test_control_is_exact_original_forward():
    from train_proposal_p3 import Proposal
    m=synthetic_model('C');x=torch.rand(2,3,256,256)
    torch.testing.assert_close(m(x),Proposal.forward(m,x),rtol=0,atol=0)

def test_real_original_head_accepts_32_grid_and_backpropagates():
    m=synthetic_model('U',True);x=torch.rand(1,3,256,256)
    y=m(x);assert y.shape==(1,256,256)
    y.square().mean().backward()
    assert torch.isfinite(m.encoder.scale.grad) and m.encoder.scale.grad.abs()>0
    assert m.head.proj[0].weight.grad.norm()>0 and m.head.out.weight.grad.norm()>0

@pytest.mark.parametrize('shape',[(1,3,512,512),(1,1,256,256),(3,256,256),(0,3,256,256)])
def test_reject_non_native_input(shape):
    with pytest.raises(ValueError):synthetic_model('U')(torch.zeros(shape))

def test_original_loss_and_tail():
    p=module();from train_proposal_p3 import focal_numerator
    x=torch.randn(6,3,3,requires_grad=True);y=torch.zeros_like(x);y[:,1,1]=1
    loss=p.loss_for_microbatch(x,y,11,'U')
    torch.testing.assert_close(loss,focal_numerator(x,y)/11,rtol=0,atol=0)
    parts=p.loss_for_microbatch(x[:4],y[:4],11,'C')+p.loss_for_microbatch(x[4:],y[4:],11,'U')
    torch.testing.assert_close(loss,parts)
    torch.testing.assert_close(torch.autograd.grad(loss,x,retain_graph=True)[0],torch.autograd.grad(parts,x)[0])
    batches=list(p.effective_batches(list(range(2134)),16))
    assert len(batches)==134 and len(batches[-1])==6 and sum(batches,[])==list(range(2134))
