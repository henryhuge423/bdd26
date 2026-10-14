"""E4 runner tests: gate arithmetic on synthetic records, guards, binding shape."""
import json
import numpy as np
import pytest
from pathlib import Path

import scripts.probe_visibility as pv
from scripts.probe_visibility import analyze_records, main


_N = [0]


def _rec(cond, group, miss, cls=4, img=0, tissue="Breast", frag=">=100", full=True, gid=None,
         area=250):
    _N[0] += 1
    return dict(image=img, tissue=tissue, gt_id=_N[0] if gid is None else gid,
                cls=cls, area_native=area, original_full=full, cond=cond, group=group,
                frag_area_bin=frag, miss=miss)


def test_h1_h2_h3_gate_arithmetic():
    rec = []
    # H1: Dead 贴边 miss 1/2 vs 内部 1/4 → gap = 0.25
    rec += [_rec("C0", "border", True), _rec("C0", "border", False),
            _rec("C0", "interior", False), _rec("C0", "interior", False),
            _rec("C0", "interior", False), _rec("C0", "interior", True)]
    # H2: 截断 C0 .25 -> C2 .75 (+.5)；完整 C0 .25 -> C2 .5 (+.25) ⇒ DiD = 0.25
    gid = 100
    for m0, m2 in [(0, 1), (1, 0), (0, 1), (0, 1)]:
        gid += 1
        rec += [_rec("C0", "truncated", bool(m0), img=1, gid=gid),
                _rec("C2", "truncated", bool(m2), img=1, gid=gid)]
    for m0, m2 in [(0, 0), (0, 1), (1, 1), (0, 0)]:
        gid += 1
        rec += [_rec("C0", "stays_full", bool(m0), img=1, gid=gid),
                _rec("C2", "stays_full", bool(m2), img=1, gid=gid)]
    gid += 1
    rec += [_rec("C0", "truncated", False, img=2, gid=gid),
            _rec("C-int", "truncated", False, img=2, gid=gid)]
    gid += 1
    rec += [_rec("C0", "stays_full", False, img=2, gid=gid),
            _rec("C-int", "stays_full", False, img=2, gid=gid)]
    # H3: <30 档 miss 1.0；>=100 档 = H2 的 4 个截断行（miss 1,0,1,1）+ 本行 False → .6
    rec += [_rec("C2", "truncated", True, cls=4, frag="<30", img=3),
            _rec("C2", "truncated", False, cls=4, frag=">=100", img=3)]
    out = analyze_records(rec, n_boot=200, seed=1)
    g = out["gates"]
    # H1: border 1/2 vs interior 1/4 → gap = 0.25
    assert g["H1"]["dead_gap"] == pytest.approx(0.25, abs=1e-12)
    # H2: 截断 C0 .25 -> C2 .75 (+.5)；完整 C0 .25 -> C2 .5 (+.25) ⇒ DiD = 0.25
    assert g["H2"]["did_c2"] == pytest.approx(0.25, abs=1e-12)
    assert g["H2"]["did_c_int"] == 0.0
    # H3: 1.0 − 0.6 = 0.4（全部 C2 截断行共享分母，与真实数据语义一致）
    assert g["H3"]["gap"] == pytest.approx(0.4, abs=1e-12)


def test_gates_require_ci_lower_positive():
    # 全零差距 → CI 覆盖 0 → H1 pass=False（门要 lo>0）
    rec = [_rec("C0", "border", False), _rec("C0", "interior", False, img=1)]
    out = analyze_records(rec, n_boot=100, seed=1)
    assert out["gates"]["H1"]["dead_gap"] == 0.0 and not out["gates"]["H1"]["pass"]


def test_h1_gate_is_area_stratified():
    """spec §4：H1 是面积分段（<100/100-400/≥400 原生 px）内的贴边−内部差距（直接标准化），
    不是粗差距。构造面积混杂：小核 bin 贴边 miss 高、大核 bin 双侧全 0——
    标准化 gap = .5·(2/3) + .5·0 = 1/3，粗差距 = 2/4 = 0.5。"""
    rec = []
    # bin <100（area=50）：贴边 3 核 miss 1,1,0；内部 1 核 miss 0
    rec += [_rec("C0", "border", True, area=50), _rec("C0", "border", True, area=50),
            _rec("C0", "border", False, area=50), _rec("C0", "interior", False, area=50)]
    # bin ≥400（area=500）：贴边 1 核、内部 3 核，全 miss 0
    rec += [_rec("C0", "border", False, area=500, img=1),
            _rec("C0", "interior", False, area=500, img=1),
            _rec("C0", "interior", False, area=500, img=1),
            _rec("C0", "interior", False, area=500, img=1)]
    out = analyze_records(rec, n_boot=100, seed=1)
    h1 = out["gates"]["H1"]
    assert h1["dead_gap"] == pytest.approx(1 / 3, abs=1e-12)          # 门统计量 = 标准化
    assert h1["crude_gap"] == pytest.approx(0.5, abs=1e-12)           # 粗差距仅描述性
    assert h1["bins"]["<100"]["border_rate"] == pytest.approx(2 / 3, abs=1e-12)
    assert h1["bins"]["<100"]["interior_rate"] == 0.0
    assert h1["bins"][">=400"]["border_n"] == 1 and h1["bins"][">=400"]["interior_n"] == 3


def test_fold_and_ckpt_guards(tmp_path, monkeypatch):
    # fold3 拒绝（allowlist 内路径，fold guard 先于 config 读取）
    with pytest.raises(SystemExit, match="reads fold2"):
        main(["--run", "runs/cellvit_uni/split1", "--fold", "3",
              "--out", str(tmp_path / "o")])
    # split≠1 拒绝：必须真的走到 config.json 读取分支（2026-10-14 评审——旧用例
    # 止步于 allowlist guard，从未执行 split guard）
    run = tmp_path / "r"; run.mkdir()
    (run / "config.json").write_text(json.dumps({"split": 3}))
    monkeypatch.setattr(pv, "ALLOWED_RUN", run.resolve())
    with pytest.raises(SystemExit, match="split=3"):
        pv.main(["--run", str(run), "--fold", "2", "--out", str(tmp_path / "o")])


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "already"; out.mkdir(); (out / "probe.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--run", "runs/cellvit_uni/split1", "--fold", "2", "--out", str(out)])
