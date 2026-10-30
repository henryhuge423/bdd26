import importlib.util,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_witness_rejects_missing_zero_and_nonfinite_evidence():
    assert importlib.util.find_spec('verify_p21_control') is not None,'P21 witness missing'
    from verify_p21_control import validate_learning
    good={k:.1 for k in ('block12','block23','output','auxiliary')}
    assert validate_learning(good,good,good,.1)
    for broken in ({k:0 for k in good},{k:float('nan') for k in good},{'output':.1}):
        with pytest.raises(ValueError):validate_learning(good,good,broken,.1)
    with pytest.raises(ValueError):validate_learning(good,good,good,0)
