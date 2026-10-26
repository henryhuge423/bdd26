import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from verify_p17_control import validate_k_witness

@pytest.mark.parametrize('key',['stage0','stage1','stage2','fusion','output'])
@pytest.mark.parametrize('value',[0.,float('nan'),float('inf')])
def test_k_witness_rejects_nonlearning_stage(key,value):
    good=dict(stage0=.1,stage1=.2,stage2=.3,fusion=.4,output=.5)
    bad=dict(good);bad[key]=value
    with pytest.raises(ValueError):validate_k_witness(bad,good)
    with pytest.raises(ValueError):validate_k_witness(good,bad)

def test_witness_requires_all_stages():
    good=dict(stage0=.1,stage1=.2,stage2=.3,fusion=.4,output=.5)
    assert validate_k_witness(good,good)
    with pytest.raises(ValueError):validate_k_witness({'output':1.},good)
