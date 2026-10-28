import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
def resource(arm):
 import train_proposal_p20 as p
 assert hasattr(p,'run_budget_hours'),'P20 per-arm resource limits absent'
 return p.run_budget_hours(arm)
@pytest.mark.parametrize('arm,hours',[('C',12),('U',36)])
def test_per_arm_limits_keep_total96(arm,hours):
 assert resource(arm)==hours
 assert 2*resource('C')+2*resource('U')==96
@pytest.mark.parametrize('arm',['B','H',None,True])
def test_invalid_resource_arm(arm):
 with pytest.raises(ValueError):resource(arm)
