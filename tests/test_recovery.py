"""Recovery post-processing variants and the light metric vs. the official code paths."""

import numpy as np
import pytest

from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernet.data import hv_targets
from nucseg.hovernet.engine import _post
from nucseg.metrics.light import image_stats, summarize
from nucseg.metrics.pannuke_eval import evaluate
from nucseg.postproc.recovery import Recovery, postprocess, proc_np_hv

N = 40


def _fake_outputs(inst, typ, rng):
    """Plausible network outputs from GT: blurred/noisy foreground, noisy HV, soft types; some nuclei
    faded out of the foreground (like missed Dead nuclei) but kept in the TP branch."""
    import cv2
    fg = (inst > 0).astype(np.float32)
    ids = [i for i in np.unique(inst) if i]
    for i in ids[::5]:
        fg[inst == i] *= 0.3
    fg = np.clip(cv2.GaussianBlur(fg, (5, 5), 0) + rng.normal(0, 0.05, fg.shape), 0, 1)
    hv = hv_targets(inst) + rng.normal(0, 0.05, (256, 256, 2)).astype(np.float32)
    tp = np.eye(6, dtype=np.float32)[typ] * 0.8 + 0.2 / 6
    tp = tp / tp.sum(-1, keepdims=True)
    return fg.astype(np.float32), hv.astype(np.float32), tp


@pytest.fixture(scope="module")
def data():
    f = PanNukeFold(1)
    if not (f.dir / "inst.npy").exists():
        pytest.skip("PanNuke fold1 not prepared")
    rng = np.random.default_rng(0)
    outs = [_fake_outputs(np.asarray(f.inst[i]).astype(np.int32), np.asarray(f.type[i]).astype(np.int64), rng)
            for i in range(N)]
    return f, outs


def test_default_equals_official(data):
    _, outs = data
    for fg, hv, tp in outs:
        ref_inst, ref_type = _post((np.concatenate([tp.argmax(-1)[..., None].astype(np.float32), fg[..., None], hv], -1),))
        inst, typ = postprocess(fg, hv, tp, Recovery())
        assert np.array_equal(inst, ref_inst) and np.array_equal(typ, ref_type)


def test_orphan_blob_recovered():
    fg = np.zeros((256, 256), np.float32)
    fg[100:104, 100:104] = 1.0          # 16 px blob: survives min_size, its marker dies in the 5x5 opening
    hv = np.zeros((256, 256, 2), np.float32)
    assert proc_np_hv(np.dstack([fg, hv]), orphans=False).max() == 0
    rec = proc_np_hv(np.dstack([fg, hv]), orphans=True)
    assert (rec > 0).sum() == 16 and len(np.unique(rec)) == 2


def test_variants_change_output_and_restore_official(data):
    _, outs = data
    fg, hv, tp = outs[0]
    a = postprocess(fg, hv, tp, Recovery(beta=1.0, thr=0.3, orphans=True))[0]
    b = postprocess(fg, hv, tp, Recovery())[0]
    assert not np.array_equal(a, b)
    ref = _post((np.concatenate([tp.argmax(-1)[..., None].astype(np.float32), fg[..., None], hv], -1),))[0]
    assert np.array_equal(b, ref)  # module global restored after a variant call


def test_light_metric_matches_full_eval(data):
    f, outs = data
    preds = [postprocess(fg, hv, tp, Recovery(beta=0.5, k_dead=2, thr=0.4, orphans=True)) for fg, hv, tp in outs]
    inst = np.stack([p[0] for p in preds])
    typ = np.stack([p[1] for p in preds])
    gt_ch = f.gt_channels[:N]
    gi, gt_ = np.asarray(f.inst[:N]), np.asarray(f.type[:N])
    tissue = np.asarray(f.tissue[:N])
    light = summarize(np.stack([image_stats(gt_ch[i], gi[i], gt_[i], inst[i], typ[i]) for i in range(N)]), tissue)
    full = evaluate(gt_ch, gi, gt_, tissue, inst, typ, workers=4)["summary"]
    assert light["mPQ"] == pytest.approx(full["official"]["mPQ"], abs=1e-12)
    assert light["bPQ"] == pytest.approx(full["official"]["bPQ"], abs=1e-12)
    assert list(light["per_class_PQ"].values()) == pytest.approx(list(full["per_class_PQ"].values()), abs=1e-12, nan_ok=True)
