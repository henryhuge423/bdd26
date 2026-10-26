import importlib
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('train_proposal_p18') is not None, 'P18 implementation absent'
    return importlib.import_module('train_proposal_p18')

def test_branch_zero_output_and_geometry():
    p=module();m=p.RGBResidual();x=torch.rand(2,3,256,256)
    assert m(x).shape==(2,256,256)
    assert torch.count_nonzero(m(x))==0
    assert m.layers[0].stride==(2,2) and m.layers[0].weight.shape==(32,3,3,3)
    assert m.layers[3].weight.shape==(32,32,3,3)
    assert m.output.weight.shape==(1,32,1,1)

def test_branch_learns_input_and_both_layers_after_two_steps():
    p=module();torch.manual_seed(19);m=p.RGBResidual();x=torch.rand(2,3,256,256)
    target=torch.zeros(2,256,256);target[:,30,30]=1
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
    for step in range(2):
        opt.zero_grad();p.loss_for_microbatch(m(x),target,2,'H').backward()
        if step==1:
            assert m.layers[0].weight.grad.norm()>0 and m.layers[3].weight.grad.norm()>0
        opt.step()
    assert not torch.equal(m(x),m(x.flip(-1)))
    fresh=p.RGBResidual();fresh.load_state_dict(m.state_dict());assert torch.equal(m(x),fresh(x))

@pytest.mark.parametrize('arm',['C','H'])
def test_loss_config_and_original_fold(arm):
    p=module();import train_proposal_p3 as orig
    assert p.Fold.__mro__[1] is orig.Fold
    logits=torch.randn(2,7,7,requires_grad=True);target=torch.zeros_like(logits);target[:,3,3]=1
    actual=p.loss_for_microbatch(logits,target,3,arm)
    assert torch.equal(actual,orig.focal_numerator(logits,target)/3)
    assert torch.equal(p.loss_for_microbatch(logits,target,0,arm),orig.focal_numerator(logits,target))
    c=p.run_config(arm,19);assert c['trainable_encoder_blocks']==list(range(12,24))
    assert c['architecture']==('original_UNI_last12' if arm=='C' else 'UNI_last12_plus_RGB_residual32')
    with pytest.raises(ValueError):p.validate_config(c,p.run_config('H' if arm=='C' else 'C',19))

@pytest.mark.parametrize('arm',['K','S','',None])
def test_invalid_arms_rejected(arm):
    p=module()
    with pytest.raises(ValueError):p.run_config(arm,19)

def test_branch_factory_preserves_rng():
    p=module();torch.manual_seed(1);state=torch.get_rng_state().clone();p.make_residual()
    assert torch.equal(torch.get_rng_state(),state)

@pytest.mark.parametrize('shape',[(1,3,128,128),(1,1,256,256),(3,256,256)])
def test_branch_rejects_wrong_geometry(shape):
    p=module()
    with pytest.raises(ValueError):p.RGBResidual()(torch.zeros(shape))
