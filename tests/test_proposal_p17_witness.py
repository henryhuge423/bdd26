import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from verify_p17_control import validate_k_witness

@pytest.mark.parametrize('key',['stage0','stage1','stage2','fusion','output'])
@pytest.mark.parametrize('value',[0.,float('nan'),float('inf')])
def test_k_witness_rejects_nonlearning_stage(key,value):
    good=dict(stage0=.1,stage1=.2,stage2=.3,fusion=.4,output=.5)
    bad=dict(good);bad[key]=value
    with pytest.raises(ValueError):validate_k_witness(bad,good,good)
    with pytest.raises(ValueError):validate_k_witness(good,bad,good)
    with pytest.raises(ValueError):validate_k_witness(good,good,bad)

def test_witness_requires_all_stages():
    good=dict(stage0=.1,stage1=.2,stage2=.3,fusion=.4,output=.5)
    assert validate_k_witness(good,good,good)
    with pytest.raises(ValueError):validate_k_witness({'output':1.},good,good)

@pytest.mark.parametrize('mode',['missing_step','decay_only'])
def test_second_step_must_include_gradient_update(mode):
    import torch
    from verify_p17_control import measure_k_step
    before={k:torch.tensor([1.,-2.]) for k in ('stage0','stage1','stage2','fusion','output')}
    after={k:v.clone() if mode=='missing_step' else v*(1-3e-4*.01) for k,v in before.items()}
    changes,adaptive=measure_k_step(before,after,{k:3e-4 for k in before},.01)
    with pytest.raises(ValueError):validate_k_witness({k:.1 for k in before},changes,adaptive)
