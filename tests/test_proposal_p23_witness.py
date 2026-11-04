import copy
import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('verify_p23_control') is not None,'P23 witness missing'
    import verify_p23_control
    return verify_p23_control

def good():
    return dict(updates=2,analytic_equal=True,online_unchanged=True,rng_unchanged=True,
                keys_exact=True,finite=True,shadow_changed=True,shadow_differs_online=True)

def test_accept_complete_witness():
    assert module().validate_ema_witness(good()) is True

@pytest.mark.parametrize('key',list(good()))
def test_reject_missing_or_failed_evidence(key):
    p=module();r=good();r.pop(key)
    with pytest.raises(ValueError):p.validate_ema_witness(r)
    r=good();r[key]=0 if key=='updates' else False
    with pytest.raises(ValueError):p.validate_ema_witness(r)
