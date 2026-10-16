from dataclasses import asdict
import copy
import json

import pytest
import torch

from nucseg.setmask.engine import SetMaskConfig, load_config, prepare_run


def test_e7_round_is_explicit_and_gamma_is_the_only_recipe_difference(tmp_path):
    old = asdict(SetMaskConfig('Q', 19, str(tmp_path)))
    new = asdict(SetMaskConfig('Q', 19, str(tmp_path), round='E7'))
    assert {k for k in old if old[k] != new[k]} == {'round', 'gamma'}
    assert old['round'] == 'E6' and old['gamma'] == .85
    assert new['round'] == 'E7' and new['gamma'] == .99


def test_round_serialization_and_cross_round_resume_are_guarded(tmp_path):
    cfg = SetMaskConfig('B', 1, str(tmp_path / 'run'), round='E7')
    prepare_run(cfg, {'frozen': 'source'})
    assert load_config(tmp_path / 'run') == cfg
    with pytest.raises(ValueError):
        prepare_run(SetMaskConfig('B', 1, cfg.out_dir), {'frozen': 'source'}, resume=True)
    raw = json.loads((tmp_path / 'run/config.json').read_text()); raw['gamma'] = .98
    (tmp_path / 'run/config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): load_config(tmp_path / 'run')


@pytest.mark.parametrize('round_name', ['E8', '', None])
def test_unknown_round_is_refused(round_name):
    with pytest.raises(ValueError): SetMaskConfig('Q', 19, 'unused', round=round_name)


def test_caller_cannot_choose_arbitrary_gamma():
    with pytest.raises(TypeError): SetMaskConfig('Q', 19, 'unused', round='E7', gamma=.98)


def test_e7_real_schedule_and_resume_preserve_optimizer_updates():
    cfg = SetMaskConfig('Q', 19, 'unused', round='E7')
    def make():
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.AdamW([p], lr=cfg.lr, betas=(.85,.95))
        return p, opt, torch.optim.lr_scheduler.ExponentialLR(opt, cfg.gamma)
    def step(p, opt, sched):
        opt.zero_grad(); (p.square().sum()).backward(); opt.step(); sched.step()
    p, opt, sched = make()
    for _ in range(25): step(p, opt, sched)
    assert opt.param_groups[0]['lr'] == pytest.approx(3e-4 * .99**25)
    state = (p.detach().clone(), copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()))
    step(p, opt, sched)
    q, other, restored = make(); q.data.copy_(state[0]); other.load_state_dict(state[1]); restored.load_state_dict(state[2])
    step(q, other, restored)
    assert torch.equal(q, p)
    assert other.param_groups[0]['lr'] == opt.param_groups[0]['lr']
