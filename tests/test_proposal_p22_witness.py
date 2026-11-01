import importlib.util,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_witness_rejects_missing_zero_and_nonfinite_evidence():
    assert importlib.util.find_spec('verify_p22_control') is not None,'P22 witness missing'
    from verify_p22_control import validate_learning
    keys=('block0','block11','block12','block23','output');good={k:.1 for k in keys}
    assert validate_learning(good,good,good)
    for broken in ({k:0 for k in good},{k:float('nan') for k in good},{'output':.1},{**good,'block0':-1.}):
        with pytest.raises(ValueError):validate_learning(good,good,broken)
        with pytest.raises(ValueError):validate_learning(broken,good,good)
