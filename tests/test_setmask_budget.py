"""User-authorized runtime extension must retain real bounded enforcement."""
import math
import pytest


def api():
    import nucseg.setmask.engine as engine
    return engine


def test_both_measured_run_estimates_fit_the_new_ten_hour_limit():
    m=api();cfg=m.SetMaskConfig('B',19,'unused')
    assert m.check_train_budget(cfg,16523.3849)>0
    assert m.check_train_budget(cfg,20947.999)>0
    assert m.check_train_budget(cfg,35999)==1


@pytest.mark.parametrize('elapsed',[36000,36001,float('nan'),float('inf'),-1])
def test_ten_hour_deadline_and_invalid_elapsed_values_fail_closed(elapsed):
    m=api();cfg=m.SetMaskConfig('Q',1,'unused')
    with pytest.raises((ValueError,RuntimeError)):m.check_train_budget(cfg,elapsed)


def test_runtime_extension_does_not_unfreeze_scientific_recipe():
    from dataclasses import asdict
    m=api();cfg=m.SetMaskConfig('Q',19,'unused');payload=asdict(cfg)
    assert payload['max_train_seconds']==36000
    assert (cfg.epochs,cfg.unfreeze_epoch,cfg.batch_size,cfg.workers,cfg.split)==(130,25,16,8,1)
    with pytest.raises(TypeError):m.SetMaskConfig('Q',19,'unused',max_train_seconds=40000)
