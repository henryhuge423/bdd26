import importlib.util
import pytest

def test_smoke_rejects_over24gib_before_pass():
    import train_proposal_p25 as p
    assert hasattr(p,'validate_smoke_memory'),'explicit runner memory guard missing'
    for value in (0,24*1024**3+1,float('nan'),float('inf')):
        with pytest.raises(ValueError):p.validate_smoke_memory(value)
    assert p.validate_smoke_memory(24*1024**3)
    assert p.validate_smoke_memory(1024**3)

@pytest.mark.parametrize('extra',[['--phase','B'],['--arm','L'],['--seed','2'],['--microbatch','8']])
def test_cli_refuses_unfrozen_recipe(extra):
    import train_proposal_p25 as p
    with pytest.raises(SystemExit):p.parse_args(['--data','x','--weights','x','--out','x','--arm','C']+extra)

def test_full24_checkpoint_roundtrip_keeps_early_blocks_and_frozen_tensors(monkeypatch,tmp_path):
    import torch
    from torch import nn
    import train_proposal_p25 as p
    from nucseg.cellvit import model as module
    class Fake(nn.Module):
        def __init__(self,**kw):
            super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(24)]);self.norm=nn.LayerNorm(1)
        def load_state_dict(self,state,strict=True):return None
    monkeypatch.setattr(module,'UNIEncoder',Fake)
    original_load=torch.load
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    torch.manual_seed(19);a=p.Proposal('unused',True,'D')
    with torch.no_grad():
        for b in (0,11,12,23):a.encoder.blocks[b].weight.add_(b+1)
    saved=dict(model=a.adapted_state(),config=p.run_config('D',19))
    path=tmp_path/'final.pth';torch.save(saved,path)
    torch.manual_seed(19);fresh=p.Proposal('unused',True,'D');frozen=fresh.encoder.norm.weight.clone()
    monkeypatch.setattr(torch,'load',original_load)
    loaded=torch.load(path,weights_only=False);p.validate_config(loaded['config'],p.run_config('D',19));p.load_adapted(fresh,loaded['model'])
    assert all(torch.equal(v,fresh.adapted_state()[k]) for k,v in a.adapted_state().items())
    assert torch.equal(frozen,fresh.encoder.norm.weight)
