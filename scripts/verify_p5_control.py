"""Verify real P5 models have identical initialization/capacity before GPU smoke."""
import json
import sys
import torch
from train_proposal_p5 import Proposal,dump

torch.set_num_threads(4)
weights,out=sys.argv[1:]
for seed in (19,1):
    torch.manual_seed(seed);low=Proposal(weights,False)
    torch.manual_seed(seed);native=Proposal(weights,True)
    assert low.state_dict().keys()==native.state_dict().keys()
    assert all(torch.equal(value,native.state_dict()[name]) for name,value in low.state_dict().items())
    assert [n for n,p in low.named_parameters() if p.requires_grad]==[n for n,p in native.named_parameters() if p.requires_grad]
    del low,native
dump(out,dict(status='PASS',seeds=[19,1],exact_initial_state_equal=True,trainable_parameter_names_equal=True))
print(json.dumps(dict(status='PASS')))
