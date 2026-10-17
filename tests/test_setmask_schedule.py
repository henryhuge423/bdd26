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


def analysis_api():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('e7_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_setmask_dev.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_analysis_reports_common_e7_round():
    configs={f'{a}_seed{s}':SetMaskConfig(a,s,'unused',round='E7') for a in ('B','Q') for s in (19,1)}
    assert analysis_api().common_round(configs)=='E7'


def test_mixed_round_rejected_before_any_provenance_or_fold_access(tmp_path,monkeypatch):
    import nucseg.setmask.engine as engine
    for a in ('B','Q'):
        for s in (19,1):
            run=tmp_path/f'{a}_seed{s}';run.mkdir()
            cfg=SetMaskConfig(a,s,str(run),round='E7' if a=='Q' else 'E6')
            (run/'config.json').write_text(json.dumps(asdict(cfg)))
    def forbid(*args):raise AssertionError('Mixed round reached data/input provenance')
    monkeypatch.setattr(engine,'make_provenance',forbid)
    with pytest.raises(ValueError,match='round'):analysis_api().verify_metadata(tmp_path)
