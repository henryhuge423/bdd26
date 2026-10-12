import json

import numpy as np
import pytest

import scripts.run_e3_quality as R


def _rect(a, i, y, x, h, w):
    a[y:y + h, x:x + w] = i


@pytest.fixture()
def world():
    n, h, w = 2, 16, 16
    gi = np.zeros((n, h, w), np.int32); gt = np.zeros((n, h, w), np.uint8)
    _rect(gi[0], 1, 1, 1, 4, 4); gt[0][gi[0] == 1] = 1      # G1 Neo
    _rect(gi[0], 2, 1, 8, 6, 6); gt[0][gi[0] == 2] = 4      # G2 Dead（base 漏检）
    _rect(gi[1], 3, 1, 1, 6, 6); gt[1][gi[1] == 3] = 2      # G3 Inf
    bi = np.zeros((n, h, w), np.int32); bt = np.zeros((n, h, w), np.uint8)
    _rect(bi[0], 1, 1, 1, 4, 4); bt[0][bi[0] == 1] = 1
    _rect(bi[1], 2, 1, 1, 6, 6); bt[1][bi[1] == 2] = 2
    xi = bi.copy(); xt = bt.copy()
    _rect(xi[0], 5, 1, 8, 6, 6); xt[0][xi[0] == 5] = 4       # C1 = G2 位置，typed Dead
    _rect(xi[0], 6, 8, 1, 6, 6); xt[0][xi[0] == 6] = 1       # C2 纯 FP
    _rect(xi[1], 4, 8, 8, 5, 5); xt[1][xi[1] == 4] = 3       # C3 纯 FP（img1）
    gc = np.zeros((n, h, w, 5), np.uint8)
    for j in range(n):
        for c in range(1, 6):
            gc[j, ..., c - 1] = (gt[j] == c)
    prob = {0: {1: _row(0.99, 1), 5: _row(0.95, 4), 6: _row(0.85, 1)},
            1: {2: _row(0.99, 2), 4: _row(0.88, 3)}}
    rec_apply = {"pair": np.array(["split1_seed1"] * 3),
                 "image": np.array([0, 0, 1]), "id": np.array([5, 6, 4]),
                 "area": np.array([36, 36, 25], float), "pmax": np.array([.95, .85, .88]),
                 "border": np.array([False] * 3), "class": np.array([4, 1, 3]),
                 "matched": np.array([True, False, False]), "iou_best": np.array([1.0, 0, 0])}
    p2cfg = {"max_area": 200, "min_prob": 0.9, "interior_only": False}
    return dict(bi=bi, bt=bt, xi=xi, xt=xt, gi=gi, gt=gt, gc=gc,
                tissue=np.array(["Breast", "Bladder"]), prob=prob, rec_apply=rec_apply, p2cfg=p2cfg)


def _row(pmax, cls):
    v = np.full(6, (1 - pmax) / 5)
    v[cls] = pmax
    return v


def _loader(w, p_low, p_sup):
    def load(a):
        return {"bi": w["bi"], "bt": w["bt"], "xi": w["xi"], "xt": w["xt"],
                "gi": w["gi"], "gt": w["gt"], "gc": w["gc"], "tissue": w["tissue"],
                "prob": w["prob"], "rec_train": None, "sup_train": None, "sup_excl": None,
                "rec_apply": w["rec_apply"], "sup_apply": np.array([1, 0, 0], float),
                "jury_names": ["j"] * 13, "p_low": p_low, "p_sup": p_sup}
    return load


def test_sweep_menu_counts_and_a1_identity(world):
    w = world
    menu = R.build_menu_loaded(w["rec_apply"], np.array([.9, .4, .2]), np.array([.9, .4, .2]), w["p2cfg"])
    rows = R.sweep(menu, w["bi"], w["bt"], w["xi"], w["xt"], w["prob"],
                   w["gc"], w["gi"], w["gt"], w["tissue"], w["p2cfg"])
    by = {r["name"]: r for r in rows}
    assert len(rows) == 18
    assert by["identity"]["added"] == 0
    assert by["a1_incumbent"]["added"] == 1                 # 只加 C1（p≥.9 且 area≥30）
    assert by["a2_f0_c100"]["added"] == 3                   # C1+C2+C3
    assert by["a2_f30_c100"]["added"] == 2                  # C1+C2（C3 area 25<30）
    assert by["a3_p0.5"]["added"] == 1                      # 只有 C1 的 p̂≥.5
    assert by["a1_incumbent"]["Dead_PQ"] > by["identity"]["Dead_PQ"]  # C1 补上 Dead
    m_by = {m["name"]: m["allowed"] for m in menu}
    assert R._unmatched_of(w["rec_apply"], m_by["a1_incumbent"]) == 0
    assert R._unmatched_of(w["rec_apply"], m_by["a2_f0_c100"]) == 2


def test_gates_numbers():
    a1 = {"name": "a1", "mPQ": .48, "bPQ": .65, "Dead_PQ": .170, "strict_dead": .134,
          "added": 1500, "unmatched": 850}
    strong = {**a1, "Dead_PQ": .173, "bPQ": .6505, "strict_dead": .134, "unmatched": 400}
    weak = {**a1, "Dead_PQ": .1705, "bPQ": .6490, "strict_dead": .131, "unmatched": 700}
    lean = {**a1, "Dead_PQ": .1695, "bPQ": .6520, "strict_dead": .1345, "unmatched": 500}
    g = R.evaluate_gates({"A3": strong, "A4": weak}, a1)
    assert g["A3"]["primary"] and not g["A3"]["secondary"]
    assert not g["A4"]["primary"] and not g["A4"]["secondary"]
    g2 = R.evaluate_gates({"A3": lean}, a1)
    assert g2["A3"]["secondary"]           # |ΔDead|≤.001、ΔbPQ≥.0005、unmatched≤75%×850
    assert not g2["A3"]["primary"]


def test_run_end_to_end_with_injected_loader(world, tmp_path, monkeypatch):
    w = world
    p2_json = tmp_path / "p2sel.json"
    # 第一次（无断言）直接用 sweep 拿 A1 行数值写进伪冻结 json
    menu = R.build_menu_loaded(w["rec_apply"], np.array([.9, .4, .2]), np.array([.9, .4, .2]), w["p2cfg"])
    rows = {r["name"]: r for r in R.sweep(menu, w["bi"], w["bt"], w["xi"], w["xt"], w["prob"],
                                          w["gc"], w["gi"], w["gt"], w["tissue"], w["p2cfg"])}
    a1 = rows["a1_incumbent"]
    sel_row = {**w["p2cfg"], **{k: v for k, v in a1.items() if k != "_stats"},
               "name": "a200_p0.9_interior0"}
    p2_json.write_text(json.dumps({"selected": sel_row}))
    ef_json = tmp_path / "efsel.json"
    ef_json.write_text(json.dumps({"selected": {"name": "a30",
                                                   **{k: v for k, v in a1.items()
                                                      if k not in ("_stats", "name")}}}))
    monkeypatch.setattr(R, "fit_logistic", lambda *a, **k: {"mock": True})
    monkeypatch.setattr(R, "predict", lambda fit, X: np.array([.9, .4, .2][:len(X)]))
    out = tmp_path / "e3out"
    argv = ["--round", str(tmp_path), "--records", str(tmp_path / "r.npz"),
            "--apply-records", str(tmp_path / "a.npz"), "--base-val", str(tmp_path / "b.npz"),
            "--x2-val", str(tmp_path / "x.npz"), "--p2-selection", str(p2_json),
            "--ef-selection", str(ef_json),
            "--fold2-support", str(tmp_path / "s.npz"), "--out", str(out),
            "--n-boot", "50"]
    R.main(argv, loader=_loader(w, np.array([.9, .4, .2]), np.array([.9, .4, .2])))
    res = json.loads((out / "results.json").read_text())
    assert len(res["rows"]) == 18 and res["selected"]["A1"]["added"] == 1
    assert "gates" in res and "bootstrap" in res and "provenance" in res
    # 断言失败路径：扰动 frozen 值
    bad = {**{k: v for k, v in a1.items() if k != "_stats"}, "mPQ": a1["mPQ"] + 1e-9}
    p2_json.write_text(json.dumps({"selected": bad}))
    out2 = tmp_path / "e3out2"
    with pytest.raises(SystemExit):
        R.main(argv[:-1] + [str(out2)],
               loader=_loader(w, np.array([.9, .4, .2]), np.array([.9, .4, .2])))


def test_main_guards(tmp_path, monkeypatch):
    out = tmp_path / "o"; out.mkdir(); (out / "results.json").write_text("{}")
    with pytest.raises(SystemExit):
        R.main(["--round", str(tmp_path), "--records", "x", "--apply-records", "x",
                "--base-val", "x", "--x2-val", "x", "--p2-selection", "x",
                "--ef-selection", "x", "--fold2-support", "x", "--out", str(out)])
    def boom(*a):
        raise SystemExit(2)
    monkeypatch.setattr(R, "ensure_dev_fold", boom)
    with pytest.raises(SystemExit):
        R.main(["--round", str(tmp_path), "--records", "x", "--apply-records", "x",
                "--base-val", "x", "--x2-val", "x", "--p2-selection", "x",
                "--ef-selection", "x", "--fold2-support", "x", "--out", str(tmp_path / "o2")])


def test_a1_drift_assert_is_the_failure_path(world, tmp_path, monkeypatch):
    """失败路径必须打在 A1 漂移断言上（消息含 'A1 replay drifted'），不是更早的输入守卫。"""
    import pytest
    w = world
    menu = R.build_menu_loaded(w["rec_apply"], np.array([.9, .4, .2]), np.array([.9, .4, .2]),
                               w["p2cfg"])
    rows = {r["name"]: r for r in R.sweep(menu, w["bi"], w["bt"], w["xi"], w["xt"], w["prob"],
                                          w["gc"], w["gi"], w["gt"], w["tissue"], w["p2cfg"])}
    a1 = rows["a1_incumbent"]
    p2_json = tmp_path / "p2.json"
    p2_json.write_text(json.dumps({"selected": {**w["p2cfg"],
                                                **{k: v for k, v in a1.items() if k != "_stats"},
                                                "name": "a200_p0.9_interior0"}}))
    ef_json = tmp_path / "ef.json"
    ef_json.write_text(json.dumps({"selected": {"name": "a30",
                                                **{k: v for k, v in a1.items()
                                                   if k not in ("_stats", "name")},
                                                "mPQ": a1["mPQ"] + 1e-9}}))  # 仅扰动 EF 行
    monkeypatch.setattr(R, "fit_logistic", lambda *a, **k: {"mock": True})
    monkeypatch.setattr(R, "predict", lambda fit, X: np.array([.9, .4, .2][:len(X)]))
    out = tmp_path / "e3drift"
    argv = ["--round", str(tmp_path), "--records", "x", "--apply-records", "x",
            "--base-val", "x", "--x2-val", "x", "--p2-selection", str(p2_json),
            "--ef-selection", str(ef_json), "--fold2-support", "x",
            "--out", str(out), "--n-boot", "50"]
    with pytest.raises(SystemExit) as ei:
        R.main(argv, loader=_loader(w, np.array([.9, .4, .2]), np.array([.9, .4, .2])))
    assert "A1 replay drifted" in str(ei.value)


def test_per_image_triples_official_nan_convention():
    """mPQ 列 = GT 在场类的平均；GT 缺席即使有预测也不计入（官方 NaN 排除，非 0）。"""
    stats = np.zeros((1, 6, 5))
    stats[0, 1] = (10, 0, 2, 8.0, 12)      # 类1：GT 在场
    stats[0, 4] = (0, 3, 0, 0.0, 0)        # 类4：GT 缺席但有 3 个预测
    stats[0, 0] = (10, 3, 2, 8.0, 14)      # 二值行
    t = R._per_image_triples(stats)
    assert t[0, 0] == t[0, 0] and abs(t[0, 0] - _pq_ref(10, 0, 2, 8.0)) < 1e-12


def _pq_ref(tp, fp, fn, sum_iou):
    d = tp + 0.5 * fp + 0.5 * fn
    return (tp / d) * (sum_iou / (tp + 1e-6)) if tp else 0.0  # 官方 SQ 分母 +1e-6
