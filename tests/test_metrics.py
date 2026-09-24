"""Check our fast metrics against the official PanNuke-metrics / HoVer-Net implementations."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "PanNuke-metrics"))
sys.path.insert(0, str(ROOT / "third_party" / "hover_net"))

import utils as official  # noqa: E402  PanNuke-metrics/utils.py
from metrics import stats_utils as hn  # noqa: E402  hover_net/metrics/stats_utils.py

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.errors import classify_errors  # noqa: E402
from nucseg.metrics.instance import aji_from_overlap, overlap, pq  # noqa: E402
from nucseg.metrics.pannuke_eval import evaluate  # noqa: E402

N = 120  # images used from fold 1


def _perturb(inst, typ, rng):
    """Realistic segmentation errors: erosion/dilation, merges, drops, spurious blobs, class flips."""
    inst = inst.astype(np.int32).copy()
    typ = typ.copy()
    ids = [i for i in np.unique(inst) if i]
    for i in ids:
        r = rng.random()
        m = inst == i
        if r < 0.10:
            inst[m] = 0
            typ[m] = 0
        elif r < 0.25:
            e = ndi.binary_erosion(m, iterations=2)
            inst[m & ~e] = 0
            typ[m & ~e] = 0
        elif r < 0.35:
            d = ndi.binary_dilation(m, iterations=2) & (inst == 0)
            inst[d] = i
            typ[d] = typ[m][0]
        elif r < 0.45:
            typ[m] = rng.integers(1, 6)
    # merge a few touching pairs
    for i in ids[: len(ids) // 8]:
        m = inst == i
        if not m.any():
            continue
        nb = np.unique(inst[ndi.binary_dilation(m, iterations=2) & ~m])
        nb = nb[(nb > 0) & (nb != i)]
        if len(nb):
            inst[inst == nb[0]] = i
            typ[inst == i] = typ[m][0]
    # spurious nuclei
    for _ in range(3):
        y, x = rng.integers(10, 246, 2)
        blob = np.zeros_like(inst, bool)
        blob[y - 4:y + 4, x - 4:x + 4] = True
        blob &= inst == 0
        inst[blob] = inst.max() + 1
        typ[blob] = rng.integers(1, 6)
    return inst, typ


@pytest.fixture(scope="module")
def fold():
    f = PanNukeFold(1)
    if not (f.dir / "inst.npy").exists():
        pytest.skip("PanNuke fold1 not prepared")
    return f


@pytest.fixture(scope="module")
def preds(fold):
    rng = np.random.default_rng(0)
    out = [_perturb(np.asarray(fold.inst[i]), np.asarray(fold.type[i]), rng) for i in range(N)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def _to_channels(inst, typ):
    """Our (inst, type) prediction -> official 6-channel format."""
    ch = np.zeros(inst.shape + (6,), np.int32)
    for i in np.unique(inst):
        if i == 0:
            continue
        m = inst == i
        c = np.bincount(typ[m], minlength=6)[1:].argmax()
        ch[..., c][m] = i
    return ch


def test_pq_and_aji_match_official(fold, preds):
    for i in range(N):
        t = official.remap_label(np.asarray(fold.inst[i]).astype(np.int32))
        p = official.remap_label(preds[0][i])
        if t.max() == 0:
            continue
        (dq, sq, pq_off), _ = official.get_fast_pq(t, p)
        ours = pq(t, p)
        assert ours.dq == pytest.approx(dq, abs=1e-9)
        assert ours.sq == pytest.approx(sq, abs=1e-9)
        assert ours.pq == pytest.approx(pq_off, abs=1e-9)
        ov = overlap(t, p)
        assert aji_from_overlap(ov) == pytest.approx(hn.get_fast_aji(t, p), abs=1e-6)
        assert aji_from_overlap(ov, plus=True) == pytest.approx(hn.get_fast_aji_plus(t, p), abs=1e-6)


def test_split_metrics_match_official_run_py(fold, preds):
    """Re-run the exact loop of PanNuke-metrics run.py and compare mPQ, bPQ, per-class PQ."""
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    true = fold.gt_channels[:N]
    tissue = fold.tissue[:N]
    pred = np.stack([_to_channels(preds[0][i], preds[1][i]) for i in range(N)])
    mpq_all, bpq_all = [], []
    for i in range(N):
        tb = official.binarize(true[i, :, :, :5].astype(np.int32))
        pb = official.binarize(pred[i, :, :, :5])
        bpq_all.append(np.nan if len(np.unique(tb)) == 1 else official.get_fast_pq(tb, pb)[0][2])
        row = []
        for j in range(5):
            t = official.remap_label(true[i, :, :, j].astype(np.int32))
            p = official.remap_label(pred[i, :, :, j].astype(np.int32))
            row.append(np.nan if len(np.unique(t)) == 1 else official.get_fast_pq(t, p)[0][2])
        mpq_all.append(row)
    img_mpq = np.array([np.nanmean(r) for r in mpq_all])
    bpq_all = np.array(bpq_all)
    tissues = [t for t in dict.fromkeys(tissue)]
    off_mpq = np.nanmean([np.nanmean(img_mpq[tissue == t]) for t in tissues])
    off_bpq = np.nanmean([np.nanmean(bpq_all[tissue == t]) for t in tissues])
    off_cls = np.nanmean(np.array(mpq_all), 0)

    res = evaluate(true, fold.inst[:N], fold.type[:N], tissue, preds[0], preds[1], workers=8)["summary"]
    assert res["official"]["mPQ"] == pytest.approx(off_mpq, abs=1e-9)
    assert res["official"]["bPQ"] == pytest.approx(off_bpq, abs=1e-9)
    for c, v in zip(res["per_class_PQ"].values(), off_cls):
        assert c == pytest.approx(v, abs=1e-9, nan_ok=True)
    assert res["strict"]["mPQ"] <= res["official"]["mPQ"] + 1e-12


def test_gt_vs_gt_is_perfect(fold):
    res = evaluate(fold.gt_channels[:N], fold.inst[:N], fold.type[:N], fold.tissue[:N],
                   fold.inst[:N], fold.type[:N], workers=8)["summary"]
    assert res["official"]["bPQ"] == pytest.approx(1.0, abs=1e-5)
    assert res["errors"]["gt"]["matched"] == sum(res["errors"]["gt"].values())


def test_error_taxonomy_toy():
    t = np.zeros((20, 40), int)
    t[2:8, 2:8] = 1; t[2:8, 8:14] = 2          # two touching nuclei -> merged by pred 1
    t[10:18, 2:12] = 3                         # one nucleus -> split into two preds
    t[2:8, 30:36] = 4                          # missed entirely
    p = np.zeros_like(t)
    p[2:8, 2:14] = 1
    p[10:18, 2:7] = 2; p[10:18, 7:12] = 3
    p[12:16, 30:36] = 5                        # spurious
    gs, ps, _ = classify_errors(overlap(t, p))
    assert list(gs) == [1, 1, 2, 3]            # merged, merged, split, missed_bg
    assert list(ps) == [1, 2, 2, 3]            # fp_merge, fp_split, fp_split, fp_bg


def test_fast_retype_matches_full_eval(fold, preds):
    from nucseg.metrics.fast_retype import RetypeEvaluator
    from nucseg.metrics.pannuke_eval import instance_classes
    from nucseg.text.retype import paint_types

    inst, typ = preds
    gt = fold.gt_channels[:N]
    tissue = np.asarray(fold.tissue[:N])
    rows = [(j, i) for j in range(N) for i in np.unique(inst[j]) if i]
    ii, iid = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    ev = RetypeEvaluator(gt, inst, ii, iid, tissue, workers=4)
    rng = np.random.default_rng(1)
    for trial in range(2):
        cls = np.concatenate([instance_classes(inst[j], typ[j])[1] for j in range(N)])
        if trial:
            cls = np.where(rng.random(len(cls)) < 0.4, rng.integers(0, 6, len(cls)), cls)
        painted = paint_types(inst, ii, iid, cls)
        full = evaluate(gt, np.asarray(fold.inst[:N]), np.asarray(fold.type[:N]), tissue, inst, painted, workers=4)
        fast, per_class = ev.mpq(cls)
        assert fast == pytest.approx(full["summary"]["official"]["mPQ"], abs=1e-9)
        assert per_class == pytest.approx(list(full["summary"]["per_class_PQ"].values()), abs=1e-9, nan_ok=True)
