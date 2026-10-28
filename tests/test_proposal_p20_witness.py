"""Readiness must reject pure decay or incomplete/nonfinite learning evidence."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_p20_control import validate_learning

def test_valid_second_step_learning():
    row=dict(block12=1e-8,block23=1e-7,output=1e-4)
    assert validate_learning(row,row,row)

@pytest.mark.parametrize('field,value',[(0,0.),(0,float('nan')),(1,float('inf')),(2,0.),(2,float('nan'))])
def test_nonfinite_or_pure_decay_rejected(field,value):
    rows=[dict(block12=1.,block23=1.,output=1.) for _ in range(3)]
    rows[field]['block12']=value
    with pytest.raises(ValueError):validate_learning(*rows)

@pytest.mark.parametrize('field',[0,1,2])
def test_missing_layer_rejected(field):
    rows=[dict(block12=1.,block23=1.,output=1.) for _ in range(3)]
    del rows[field]['block23']
    with pytest.raises(ValueError):validate_learning(*rows)
