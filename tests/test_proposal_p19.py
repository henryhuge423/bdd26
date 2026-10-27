import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    path=Path(__file__).resolve().parents[1]/'scripts/train_proposal_p19.py'
    assert path.is_file(), 'P19 batching contract not implemented'
    import train_proposal_p19
    return train_proposal_p19

@pytest.mark.parametrize('arm,steps,tail',[('C',134,6),('B',67,22)])
def test_complete_epoch_and_tail(arm,steps,tail):
    m=module();c=m.run_config(arm,19,8)
    blocks=list(m.effective_batches(list(range(2134)),c['effective_batch']))
    assert len(blocks)==steps and len(blocks[-1])==tail
    assert sum(blocks,[])==list(range(2134))
    assert c['microbatch']==8 and c['architecture']=='original_UNI_last12'

@pytest.mark.parametrize('arm,seed,micro',[('H',19,8),('C',2,8),('B',True,8),('B',1,3),('C',19,True)])
def test_invalid_config(arm,seed,micro):
    with pytest.raises(ValueError):module().run_config(arm,seed,micro)

@pytest.mark.parametrize('count',[0,1,37])
def test_micro_gradients_use_whole_effective_denominator(count):
    m=module();x=torch.linspace(-2,2,5*9).reshape(5,3,3).requires_grad_();y=torch.zeros_like(x);y[1,1,1]=1
    whole=m.loss_for_microbatch(x,y,count,'C');g=torch.autograd.grad(whole,x)[0]
    parts=sum(m.loss_for_microbatch(x[i:i+2],y[i:i+2],count,'B') for i in range(0,5,2))
    assert torch.allclose(whole,parts,rtol=1e-6,atol=1e-6)
    assert torch.allclose(g,torch.autograd.grad(parts,x)[0],rtol=1e-6,atol=1e-6)

def test_selection_and_fallback():
    m=module();rows={4:dict(seconds=[10,10,10],passed=True,peak=4<<30),8:dict(seconds=[8,8,8],passed=True,peak=6<<30),16:dict(seconds=[6,6,6],passed=False,peak=8<<30)}
    assert m.select_microbatch(rows)==8
    rows[8]['seconds']=[9.6]*3
    assert m.select_microbatch(rows)==4
    rows[8]['seconds']=[float('nan')]*3
    assert m.select_microbatch(rows)==4
    rows[4]['passed']=False
    with pytest.raises(ValueError):m.select_microbatch(rows)

def test_numeric_acceptance_rejects_nonfinite_and_wrong_gradient():
    m=module();a=torch.arange(1,6,dtype=torch.float32)
    assert m.numeric_comparison(1.,1.001,a,a*1.001)['passed']
    for b in (-a,torch.full_like(a,float('nan')),a*2):
        assert not m.numeric_comparison(1.,1.,a,b)['passed']
    assert not m.numeric_comparison(1.,float('nan'),a,a)['passed']

def test_identical_zero_gradient_is_valid_but_missing_signal_is_not():
    m=module();z=torch.zeros(7)
    assert m.numeric_comparison(1.,1.,z,z)['passed']
    assert not m.numeric_comparison(1.,1.,z,torch.ones(7))['passed']
