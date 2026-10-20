"""Compute-node-only verification: two real optimizer updates match original P3 T1."""
import json, os, sys
from pathlib import Path
import torch
import train_proposal_p3 as old
import train_proposal_p4 as new

torch.set_num_threads(4)
weights,data,out=sys.argv[1:]
torch.manual_seed(19); reference=old.Proposal(weights,True).cuda()
torch.manual_seed(19); control=new.Proposal(weights,True).cuda();control.aux.requires_grad_(False)
assert all(torch.equal(value,control.state_dict()[key]) for key,value in reference.state_dict().items())
fold=old.Fold(data,1)
x,y,_=fold.batch(list(range(16)),19,0,True);x=x.cuda();y=y.cuda()
den=max(1,sum(len(fold.gt[i]) for i in range(16)))
def optimizer(model):
 return torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=3e-4),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5)],betas=(.9,.999),weight_decay=.01)
opts=[optimizer(m) for m in (reference,control)]
for step in range(2):
 for model,opt in zip((reference,control),opts):
  model.train();opt.zero_grad(set_to_none=True)
  for begin in range(0,16,4):
   with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x[begin:begin+4])
   (old.focal_numerator(logits,y[begin:begin+4])/den).backward()
  if model is reference:
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
  else:
   new.clip_separate([p for p in model.encoder.parameters() if p.requires_grad]+list(model.head.parameters()),[])
  opt.step()
 assert all(torch.equal(value,control.state_dict()[key]) for key,value in reference.state_dict().items()),step
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 assert torch.equal(reference.eval()(x[:2]),control.eval()(x[:2]))
Path(out).write_text(json.dumps(dict(status='PASS',optimizer_steps=2,base_parameters_equal=True,center_logits_equal=True,gpu=torch.cuda.get_device_name(0),job_id=os.environ['SLURM_JOB_ID']),indent=2))
