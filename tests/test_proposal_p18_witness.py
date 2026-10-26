import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
def module():
    assert importlib.util.find_spec('verify_p18_control') is not None,'missing P18 learning verifier'
    return importlib.import_module('verify_p18_control')
KEYS={'local0','local1','local_output','block12','block23','output'}
def test_complete_witness_passes():
    v={k:.1 for k in KEYS};assert module().validate_h_witness(v,v,v)
@pytest.mark.parametrize('key',sorted(KEYS))
@pytest.mark.parametrize('value',[0.,float('nan'),float('inf'),-.1])
def test_missing_learning_rejected(key,value):
    v={k:.1 for k in KEYS};bad={**v,key:value}
    with pytest.raises(ValueError):module().validate_h_witness(v,bad,v)
    with pytest.raises(ValueError):module().validate_h_witness(bad,v,v)
    with pytest.raises(ValueError):module().validate_h_witness(v,v,bad)
def test_decay_only_and_incomplete_rejected():
    v={k:.1 for k in KEYS}
    with pytest.raises(ValueError):module().validate_h_witness(v,v,{k:0 for k in KEYS})
    with pytest.raises(ValueError):module().validate_h_witness({},v,v)
