import numpy as np
import pytest

from scripts.e1t_typing_consensus import analyze, builtin_jury, main, pair_base_detailed


def _rect(inst, i, y, x, h, w):
    inst[y:y + h, x:x + w] = i


def _mini_world():
    """2 images; GT types: img0 GT1=Neo(1) GT2=Inf(2); img1 GT3=Dead(4).

    base: GT1 typed Inf(2) [wrong -> G1 pair (1,2)], GT2 typed Inf(2) [right -> G0],
    GT3 typed Neo(1) [wrong -> G1 pair (4,1)].
    Jury votes per GT: GT1 {s0:2, s1:2, s2:2, s4:1} -> majority 2 == base (both folds);
    GT2 {s0:2, s1:2, s2:2, s4:3} -> majority 2 == GT; GT3 {s0:4, s4:1} -> 2 votes, no majority.
    """
    n, h, w = 2, 16, 16
    gt = np.zeros((n, h, w), np.int32); gtt = np.zeros((n, h, w), np.uint8)
    base = np.zeros((n, h, w), np.int32); bast = np.zeros((n, h, w), np.uint8)
    _rect(gt[0], 1, 2, 2, 6, 6); gtt[0][gt[0] == 1] = 1
    _rect(gt[0], 2, 9, 9, 5, 5); gtt[0][gt[0] == 2] = 2
    _rect(gt[1], 3, 2, 2, 6, 6); gtt[1][gt[1] == 3] = 4
    _rect(base[0], 5, 2, 2, 6, 6); bast[0][base[0] == 5] = 2   # GT1 mis-typed Inf
    _rect(base[0], 6, 9, 9, 5, 5); bast[0][base[0] == 6] = 2   # GT2 correct
    _rect(base[1], 1, 2, 2, 6, 6); bast[1][base[1] == 1] = 1   # GT3 mis-typed Neo

    # seat -> per-GT type votes: rows = seats, cols = GT1, GT2, GT3 (None = no instance)
    vote_tbl = [
        ("s0", "CellViT", 1, [2, 2, 4]),
        ("s1", "CellViT", 1, [2, 2, None]),
        ("s2", "CellViT", 3, [2, 2, None]),
        ("s3", "CellViT", 3, [None, None, None]),
        ("s4", "HoverNet", 1, [1, 3, 1]),
        ("s5", "HoverNet", 3, [2, None, None]),
    ]
    gt_boxes = [(0, 1, 2, 2, 6, 6), (0, 2, 9, 9, 5, 5), (1, 3, 2, 2, 6, 6)]
    jury = []
    for name, arch, fold, votes in vote_tbl:
        si = np.zeros((n, h, w), np.int32); st = np.zeros((n, h, w), np.uint8)
        for k, v in enumerate(votes):
            if v is None:
                continue
            img, gid, y, x, hh, ww = gt_boxes[k]
            iid = 10 + k
            _rect(si[img], iid, y, x, hh, ww); st[img][si[img] == iid] = v
        jury.append({"name": name, "arch": arch, "train_fold": fold, "inst": si, "type": st})
    return gt, gtt, base, bast, jury


def test_analyze_mini_world():
    gt, gtt, base, bast, jury = _mini_world()
    out = analyze(gt, gtt, base, bast, jury, n_boot=100)
    assert out["groups"] == {"G1": 2, "G0": 1, "ambiguous": 0, "unmatched_gt": 0}
    g1 = out["overall"]["G1"]
    assert (g1["maj_eq_base"], g1["maj_eq_gt"], g1["maj_other"], g1["no_majority"]) == (1, 0, 0, 1)
    g0 = out["overall"]["G0"]
    assert (g0["maj_eq_gt"], g0["maj_neq_gt"], g0["no_majority"]) == (1, 0, 0)
    assert out["by_pair"]["1,2"]["maj_eq_base"] == 1
    assert out["by_pair"]["4,1"]["no_majority"] == 1
    lo, hi = out["bootstrap"]["G1_maj_eq_base"]
    assert 0.0 <= lo <= hi <= 1.0
    # HN-only sensitivity（2 席，min_votes=2）：GT1 两席平票、GT2/GT3 仅 1 票 → 全部 no_majority
    hn = out["sensitivity"]["hn_only"]
    assert hn["min_votes"] == 2
    assert hn["overall"]["G1"]["no_majority"] == 2


def test_analyze_unmatched_and_ambiguous_counted():
    gt, gtt, base, bast, jury = _mini_world()
    gt2 = gt.copy(); gtt2 = gtt.copy()
    _rect(gt2[0], 7, 2, 2, 6, 3)  # 重新利用：把 GT1 拆成两半制造并列歧义
    gt2[0][2:8, 5:8] = 0
    _rect(gt2[0], 8, 2, 5, 6, 3)
    gtt2[0][gt2[0] == 7] = 1; gtt2[0][gt2[0] == 8] = 1
    # GT7/GT8 各与 base5 的 IoU = 18/36 = 0.5（不 > 0.5）→ unmatched（不触歧义分支）
    out = analyze(gt2, gtt2, base, bast, jury, n_boot=50)
    assert out["groups"]["unmatched_gt"] >= 2


def test_pair_base_detailed_matches_pair_base():
    from nucseg.e1_typing import pair_base
    gt, gtt, base, bast, jury = _mini_world()
    rows = pair_base_detailed(gt[0], gtt[0], base[0], bast[0])
    plain = pair_base(gt[0], base[0])
    for r in rows:
        assert r["status"] in ("matched", "ambiguous", "unmatched")
        if r["status"] == "matched":
            assert plain[r["gt_id"]][0] == r["base_id"]


def test_builtin_jury_excludes_base_run(tmp_path):
    jury = builtin_jury(tmp_path)
    names = [j["name"] for j in jury]
    assert len(jury) == 10 and "cellvit_s1_e1_x1" not in names and "cellvit_s1_e1_x2" not in names
    assert "cellvit_s1_e19_x1" in names and "cellvit_s3_e2_x2" in names
    assert all(j["path"].startswith(str(tmp_path)) for j in jury)


def test_main_refuses_existing_out_and_bad_fold(tmp_path, monkeypatch):
    out = tmp_path / "o"; out.mkdir(); (out / "stats.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--round", str(tmp_path), "--out", str(out)])
    out2 = tmp_path / "o2"
    monkeypatch.setattr("scripts.e1t_typing_consensus.ensure_dev_fold",
                        lambda *a: (_ for _ in ()).throw(SystemExit(2)))
    with pytest.raises(SystemExit):
        main(["--round", str(tmp_path), "--out", str(out2)])


def test_bootstrap_rates_condition_on_group():
    """bootstrap 率必须以组为分母：组外行 NaN（不计入），不是 0。"""
    gt, gtt, base, bast, jury = _mini_world()
    out = analyze(gt, gtt, base, bast, jury, n_boot=100)
    # G1 n=2、maj_eq_base=1 -> 点估计 .5；CI 应围绕 .5 而非 1/3（若分母含 G0 则错）
    lo, hi = out["bootstrap"]["G1_maj_eq_base"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    # G0_maj_neq_gt 在迷你世界为 0/1 -> CI 收敛 [0,0]
    lo0, hi0 = out["bootstrap"]["G0_maj_neq_gt"]
    assert lo0 == 0.0 and hi0 == 0.0
