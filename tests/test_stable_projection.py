import importlib.util
from pathlib import Path
import torch

def api():
    p=Path(__file__).with_name('probe_stable_projection.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('stable',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_only_mask_projection_frozen():
    from nucseg.setmask.model import QueryMaskHead
    h=QueryMaskHead(channels=4,dim=8,mask_dim=4,queries=2,layers=1,heads=2)
    params=api().trainable_head(h)
    assert all(p.requires_grad== (not n.startswith('mask_projection.')) for n,p in h.named_parameters())
    assert {id(p) for p in params}=={id(p) for n,p in h.named_parameters() if not n.startswith('mask_projection.')}

def test_optimizer_step_preserves_projection_and_updates_head():
    from nucseg.setmask.model import QueryMaskHead
    torch.manual_seed(19);h=QueryMaskHead(channels=4,dim=8,mask_dim=4,queries=2,layers=1,heads=2).eval()
    old={k:v.clone() for k,v in h.state_dict().items()}
    opt=torch.optim.AdamW(api().trainable_head(h),lr=.01)
    pred=h(torch.randn(1,4,8,8));loss=pred['class_logits'].square().sum()+pred['mask_embed'].square().sum()
    opt.zero_grad();loss.backward();opt.step()
    assert all(torch.equal(old[k],v) for k,v in h.state_dict().items() if k.startswith('mask_projection.'))
    assert not torch.equal(old['class_head.weight'],h.class_head.weight)
    assert all(p.grad is None for p in h.mask_projection.parameters())
