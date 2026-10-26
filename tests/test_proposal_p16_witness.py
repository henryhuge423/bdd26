"""A false R optimizer-update success must be rejected before GPU submission."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_p16_control as witness


def test_measure_real_finite_encoder_head_updates():
    before={'encoder':torch.tensor([1.,2.]),'output':torch.zeros(2)}
    after={'encoder':torch.tensor([1.25,2.]),'output':torch.tensor([0.,-.5])}
    assert witness.measure_updates(before,after)=={'encoder':.25,'output':.5}
    assert torch.equal(before['output'],torch.zeros(2))


@pytest.mark.parametrize('which',['encoder','output'])
def test_unchanged_parameter_rejects_false_success(which):
    before={'encoder':torch.ones(2),'output':torch.zeros(2)}
    after={k:v+1 for k,v in before.items()};after[which]=before[which].clone()
    with pytest.raises(AssertionError):witness.measure_updates(before,after)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_poststep_rejects_false_success(value):
    before={'encoder':torch.ones(2),'output':torch.zeros(2)}
    after={'encoder':torch.tensor([value,2.]),'output':torch.ones(2)}
    with pytest.raises(AssertionError):witness.measure_updates(before,after)


def test_missing_head_or_wrong_shape_is_rejected():
    before={'encoder':torch.ones(2),'output':torch.zeros(2)}
    for after in ({'encoder':torch.ones(2)*2},{'encoder':torch.ones(3),'output':torch.ones(2)}):
        with pytest.raises(AssertionError):witness.measure_updates(before,after)
