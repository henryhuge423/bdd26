"""Stage-report-3 extraction: every table number re-derived from stored artifacts.

Covers the 2026-10-02..05 phase: P0-P3 v2 corrected standing, typing-confidence
audit, matched-seed P2 validation, existence audit, and the EF-P2 round.

Dumps (stdout; save to runs/analysis/report3_numbers.txt):
  [A] P0 v2 corrected standing (seed19, 3-split): x1 / x1+M/S / x2+du2 / x2+du2+M/S
  [B] P1 complementarity: val candidate pools, interior-Dead match, type swap
  [C] P2 v2: per-split frozen selections + test endpoints + deltas + bootstrap CIs
  [D] P3 scale probe (128 split1-val images): train x infer grid
  [E] typing audit: matched typing acc (x1 vs x2, Dead), argmax disagreement,
      arbitration, P2 added/rejected candidate quality
  [F] matched-seed round: decision rule, per-split seed means/stds, CIs, sign counts,
      per-pair deltas, M/S secondary, raw-du2 per-pair Dead deltas
  [G] EF round: selections, decision, per-split means/CIs, per-pair attribution
      (EF - plain P2), absolute standing of the three arms on the 9 pairs
  [H] existence audit: pooled + per-axis match rates (recomputed from records.npz),
      including the sub-30px slice shares used by the a30 menu

Run: ~/.conda/envs/nuclei/bin/python scripts/report3_numbers.py
"""
import json

import numpy as np

R = "runs/analysis"
EP = ["mPQ", "bPQ", "mPQ+", "strict mPQ", "Dead PQ", "strict Dead PQ"]


def jload(p):
    with open(p) as f:
        return json.load(f)


def fmt(v, nd=4):
    return f"{v:+.{nd}f}" if (isinstance(v, float) and (v < 0 or (v * 10**nd >= 5))) else f"{v:.{nd}f}"


# ---------------------------------------------------------------- [A] P0 v2 standing
print("========== [A] P0 v2 corrected standing (seed 19, 3-split means) ==========")
rv = jload(f"{R}/p0p3_20261002_v2/results_verified.json")
ARMS = [("x1 (raw)", "base_identity"), ("x1 + M/S v2", "base_ms"),
        ("x2 + du2 (raw)", "x2_identity"), ("x2 + du2 + M/S v2", "x2_ms")]
mean_keys = {"mPQ": "mPQ", "bPQ": "bPQ", "mPQ+": "mPQ+", "Dead PQ": "DeadPQ",
             "strict Dead PQ": "strictDeadPQ", "strict mPQ": "strictmPQ", "Dead F_c": "DeadFc"}
for label, key in ARMS:
    m = rv["p0"][key]["mean"]
    print(f" {label:20s} " + " ".join(f"{k}={m[v]:.6f}" for k, v in mean_keys.items() if v in m))
print(" per-split x1+M/S:", {s: {k: round(v, 6) for k, v in rv["p0"]["base_ms"]["splits"][str(s)].items()}
                             for s in (1, 2, 3)})
print(" val M/S selections (base / x2):")
for s in (1, 2, 3):
    b = rv["p0_selections"][str(s)]["base"]
    x = rv["p0_selections"][str(s)]["x2"]
    print(f"  split{s}: base frac={b.get('frac')} a_min={b.get('a_min')} | "
          f"x2 frac={x.get('frac')} a_min={x.get('a_min')}")
print()

# ---------------------------------------------------------------- [B] P1
print("========== [B] P1 complementarity (x1 vs P0-selected x2+du2+M/S) ==========")
for s in (1, 2, 3):
    v = rv["p1"][str(s)]["val"]
    c = v["diagnostics"]["candidates"]
    oracle = v["scores"]["oracle_add"]["mPQ"] - v["scores"]["base"]["mPQ"]
    print(f" split{s} val candidates: {json.dumps(c)} oracle_add_dmPQ={oracle:+.6f}")
for s in (1, 2, 3):
    t = rv["p1"][str(s)]["test"]
    print(f" split{s} test base_dead: {json.dumps(t['diagnostics']['base_dead'])}")
    print(f" split{s} test x2_dead  : {json.dumps(t['diagnostics']['x2_dead'])}")
print(" p1_summary:", json.dumps(rv["p1_summary"], indent=1))
print()

# ---------------------------------------------------------------- [C] P2 v2
print("========== [C] P2 v2 (conservative additions, seed 19) ==========")
for s in (1, 2, 3):
    sel = rv["p2"]["selections"][str(s)]["selected"]
    row = rv["p2"]["splits"][str(s)]
    status = rv["p2"]["selections"][str(s)]["status"]
    print(f" split{s} [{status}] selected: {json.dumps(sel)}")
    print(f" split{s} test: " + " ".join(f"{k}={row[v]:.6f}" for k, v in mean_keys.items() if v in row))
print(" mean:", {k: round(v, 6) for k, v in rv["p2"]["mean"].items()})
print(" delta vs x1+M/S:", {k: round(v, 6) for k, v in rv["p2_delta_vs_base_ms"].items()})
for s in (1, 2, 3):
    b = rv["p2_bootstrap"][str(s)]
    print(f" bootstrap split{s}: delta={json.dumps({k: round(x, 6) for k, x in b['delta'].items()})}")
    print(f"   ci95={json.dumps({k: [round(x, 6) for x in v] for k, v in b['ci95'].items()})}")
print()

# ---------------------------------------------------------------- [D] P3
print("========== [D] P3 scale probe (split1 val fold2, 128 images) ==========")
for cond in ("train1_infer1", "train1_infer2", "train2_infer1", "train2_infer2"):
    m = rv["p3"][cond]["metrics"]
    sc = rv["p3"][cond]["sample_counts"]
    print(f" {cond}: " + " ".join(f"{k}={v:.6f}" for k, v in m.items())
          + f" | counts={json.dumps(sc)}")
print()

# ---------------------------------------------------------------- [E] typing audit
print("========== [E] typing-confidence audit (2026-10-03) ==========")
ts = jload(f"{R}/typing_audit_20261003/summary.json")
print(" three_split_mean arms:")
for role in ("test", "val"):
    for arm in ("x1", "x2"):
        d = ts["three_split_mean"][role]["arms"][arm]
        if d is None:
            continue
        print(f"  {arm} ({role} equivalents): acc={d.get('acc_given_matched'):.6f} "
              f"dead={d.get('dead_acc_given_matched'):.6f}")
print(" per-split (val, splitN_val keys):")
for k in ts["per_split"]:
    a = ts["per_split"][k]["arms"]
    print(f"  {k}: x1 acc={a['x1']['acc_given_matched']:.4f} dead={a['x1']['dead_acc_given_matched']:.4f} "
          f"| x2 acc={a['x2']['acc_given_matched']:.4f} dead={a['x2']['dead_acc_given_matched']:.4f} "
          f"| disagree x1={a['x1']['argmax_disagree_rate']:.4f} x2={a['x2']['argmax_disagree_rate']:.4f}")
# test-side files for arbitration + P2 candidate quality
tot = {"br": 0, "xr": 0, "bw": 0, "brc": 0.0, "xrc": 0.0, "stable": 0}
for s in (1, 2, 3):
    d = jload(f"{R}/typing_audit_20261003/split{s}_test.json")
    da = d["agreement"]["disagreement_attribution"]
    tot["br"] += da["base_right_x2_wrong"]
    tot["xr"] += da["base_wrong_x2_right"]
    tot["bw"] += da["both_wrong"]
    tot["brc"] += da["base_right_x2_wrong_conf_sum"]
    tot["xrc"] += da["base_wrong_x2_right_conf_sum"]
    tot["stable"] += d["agreement"]["stable_pairs"]
    pa, pr = d["p2_added"], d["p2_rejected"]
    print(f" split{s} test: stable_pairs={d['agreement']['stable_pairs']} "
          f"base_right={da['base_right_x2_wrong']} x2_right={da['base_wrong_x2_right']} "
          f"both_wrong={da['both_wrong']} "
          f"conf(x2 right)={da['base_wrong_x2_right_conf_sum']/max(da['base_wrong_x2_right'],1):.4f} "
          f"conf(base right)={da['base_right_x2_wrong_conf_sum']/max(da['base_right_x2_wrong'],1):.4f}")
    if pa["n"]:
        bm = pa["by_class"]
        n_all = sum(bm[c]["n"] for c in bm)
        m_all = sum(bm[c].get("matched", 0) for c in bm)
        c_all = sum(bm[c].get("correct", 0) for c in bm)
        dead = bm.get("4", {})
        print(f"   p2_added n={n_all} matched={m_all} ({m_all/n_all:.4f}) "
              f"correct_given_matched={c_all/max(m_all,1):.4f} "
              f"dead: n={dead.get('n',0)} matched={dead.get('matched',0)} "
              f"correct={dead.get('correct',0)}")
    if pr["n"]:
        bm = pr["by_class"]
        n_all = sum(bm[c]["n"] for c in bm)
        m_all = sum(bm[c].get("matched", 0) for c in bm)
        c_all = sum(bm[c].get("correct", 0) for c in bm)
        dead = bm.get("4", {})
        print(f"   p2_rejected n={n_all} matched={m_all} ({m_all/n_all:.4f}) "
              f"correct_given_matched={c_all/max(m_all,1):.4f} "
              f"dead: n={dead.get('n',0)} matched={dead.get('matched',0)} "
              f"correct={dead.get('correct',0)}")
print(f" pooled test arbitration: stable={tot['stable']} base_right={tot['br']} "
      f"x2_right={tot['xr']} both_wrong={tot['bw']} conf_x2right={tot['xrc']/tot['xr']:.4f} "
      f"conf_baseright={tot['brc']/tot['br']:.4f}")
# val-side typing quality of the split2 p>=.9 Dead additions (findings: val .850)
for s in (2, 3):
    d = jload(f"{R}/typing_audit_20261003/split{s}_val.json")
    pa = d["p2_added"]
    if pa["n"]:
        dead = pa["by_class"].get("4", {})
        if dead.get("matched"):
            print(f" split{s} val added Dead: n={dead['n']} matched={dead['matched']} "
                  f"correct={dead['correct']} acc={dead['correct']/dead['matched']:.4f}")
print()

# ---------------------------------------------------------------- [F] matched-seed round
print("========== [F] matched-seed round (2026-10-04/05) ==========")
ms = jload(f"{R}/matched_seed_20261004/stats/stats.json")
print(" decision:", json.dumps(ms["decision"]))
print(" overall three_split_mean:",
      {k: round(v, 6) for k, v in ms["overall"]["three_split_mean"].items()})
print(" overall max_split_seed_std:",
      {k: round(v, 6) for k, v in ms["overall"]["max_split_seed_std"].items()})
print(" overall sign_counts:",
      {k: f"{v['nonneg']}/9" for k, v in ms["overall"]["sign_counts"].items()})
for s in ("1", "2", "3"):
    d = ms["splits"][s]
    b = jload(f"{R}/matched_seed_20261004/stats/bootstrap_split{s}.json")
    print(f" split{s} seed_mean:", {k: round(v, 6) for k, v in d["seed_mean"].items()})
    print(f" split{s} seed_std :", {k: round(v, 6) for k, v in d["seed_std"].items()})
    print(f" split{s} signs    :", {k: f"{v['nonneg']}/3" for k, v in d["sign_counts"].items()})
    for ep in ("mPQ", "bPQ", "Dead PQ"):
        print(f"   boot {ep}: point={b[ep]['point']:.6f} ci95=[{b[ep]['ci95'][0]:.6f},{b[ep]['ci95'][1]:.6f}]")
print(" per-pair deltas (p2 - baseline):")
for pair, d in ms["pairs"].items():
    print(f"  {pair:14s} " + " ".join(f"{k}={d['delta'][k]:+.5f}" for k in EP[:2] + ["Dead PQ"]))
print(" M/S secondary per-pair (ms_delta):")
for pair, d in ms["pairs"].items():
    print(f"  {pair:14s} " + " ".join(f"{k}={d['ms_delta'][k]:+.5f}" for k in EP[:2] + ["Dead PQ"]))
ms_ms = {k: float(np.mean([d["ms_delta"][k] for d in ms["pairs"].values()])) for k in EP}
print(" M/S secondary 3-split means:", {k: round(v, 6) for k, v in ms_ms.items()})
print(" P2 val selections per pair:")
for s in (1, 2, 3):
    for seed in (19, 1, 2):
        sel = jload(f"{R}/matched_seed_20261004/p2/split{s}_seed{seed}/selection.json")
        row = sel["selected"] if "selected" in sel else None
        name = row.get("name") if isinstance(row, dict) else sel.get("selected_name")
        status = sel.get("status")
        added = row.get("added") if isinstance(row, dict) else None
        print(f"  split{s}_seed{seed}: {status} {name} added={added}")
# raw du2 per-pair Dead deltas (x2 identity - base identity) for the 5 grid-era pairs
print(" raw-du2 per-pair Dead deltas (grid-era pairs, from p0 test evals):")
import os
for s in (1, 2, 3):
    for seed in (19, 1, 2):
        p0sel = jload(f"{R}/matched_seed_20261004/p0/split{s}_seed{seed}/selection.json")
        vals = {}
        for arm in ("base", "x2"):
            for cand in ("identity", "ms"):
                d = f"{R}/matched_seed_20261004/p0/split{s}_seed{seed}/test/{arm}/{cand}/eval/summary.json"
                if os.path.exists(d):
                    vals[(arm, cand)] = jload(d)["per_class_PQ"]["Dead"]
        ri = "identity" if ("base", "identity") in vals else None
        if ri and ("x2", "identity") in vals:
            print(f"  split{s}_seed{seed}: raw du2 dDead={vals[('x2','identity')]-vals[('base','identity')]:+.5f}")
print()

# ---------------------------------------------------------------- [G] EF round
print("========== [G] EF-P2 round (2026-10-05) ==========")
ef = jload(f"{R}/existence_ef_20261005/stats/stats.json")
print(" decision:", json.dumps(ef["decision"]))
print(" overall three_split_mean:",
      {k: round(v, 6) for k, v in ef["overall"]["three_split_mean"].items()})
print(" overall max_split_seed_std:",
      {k: round(v, 6) for k, v in ef["overall"]["max_split_seed_std"].items()})
print(" overall sign_counts:",
      {k: f"{v['nonneg']}/9" for k, v in ef["overall"]["sign_counts"].items()})
for s in ("1", "2", "3"):
    d = ef["splits"][s]
    b = jload(f"{R}/existence_ef_20261005/stats/bootstrap_split{s}.json")
    print(f" split{s} seed_mean:", {k: round(v, 6) for k, v in d["seed_mean"].items()})
    print(f" split{s} seed_std :", {k: round(v, 6) for k, v in d["seed_std"].items()})
    for ep in ("mPQ", "bPQ", "Dead PQ"):
        print(f"   boot {ep}: point={b[ep]['point']:.6f} ci95=[{b[ep]['ci95'][0]:.6f},{b[ep]['ci95'][1]:.6f}]")
print(" per-pair deltas (EF-P2 - baseline):")
for pair, d in ef["pairs"].items():
    print(f"  {pair:14s} " + " ".join(f"{k}={d['delta'][k]:+.5f}" for k in EP[:2] + ["Dead PQ"]))
print(" attribution per pair (EF-P2 - plain P2):")
for pair in ms["pairs"]:
    dm, de = ms["pairs"][pair]["p2"], ef["pairs"][pair]["p2"]
    print(f"  {pair:14s} " + " ".join(f"{k}={de[k]-dm[k]:+.5f}" for k in EP[:2] + ["Dead PQ"]))
att = {k: float(np.mean([ef["pairs"][p]["p2"][k] - ms["pairs"][p]["p2"][k] for p in ms["pairs"]]))
       for k in EP}
print(" attribution 3-split means:", {k: round(v, 6) for k, v in att.items()})
print(" EF val selections per pair:")
for s in (1, 2, 3):
    for seed in (19, 1, 2):
        sel = jload(f"{R}/existence_ef_20261005/p2ef/split{s}_seed{seed}/selection.json")
        name = sel.get("selected_name") or (sel.get("selected") or {}).get("name")
        print(f"  split{s}_seed{seed}: {sel.get('status')} {name}")
print(" absolute standing on the 9 pairs (3-split seed means):")
for label, st in [("x1+M/S baseline", "baseline"), ("P2 (plain)", "p2")]:
    vals = {k: float(np.mean([ms["pairs"][p][st][k] for p in ms["pairs"]])) for k in EP}
    print(f"  {label:18s} " + " ".join(f"{k}={v:.6f}" for k, v in vals.items()))
vals = {k: float(np.mean([ef["pairs"][p]["p2"][k] for p in ef["pairs"]])) for k in EP}
print(f"  {'EF-P2':18s} " + " ".join(f"{k}={v:.6f}" for k, v in vals.items()))
print()

# ---------------------------------------------------------------- [H] existence audit
print("========== [H] existence audit (val folds, 17,359 frozen-P2 additions) ==========")
z = np.load(f"{R}/existence_audit_20261005/records.npz")
added, matched = z["added"], z["matched"]
print(f" n_eligible={len(added)} n_added={int(added.sum())} "
      f"pooled_match={matched[added].mean():.6f} (n={int(matched[added].sum())})")
pairs = z["pair"][added]
m = matched[added]
per_pair = [m[pairs == p].mean() for p in sorted(set(pairs.tolist()))]
print(f" per-pair mean of rates={np.mean(per_pair):.6f}")
for axis in ("area", "dist", "circularity", "extent", "pmax"):
    tab = jload(f"{R}/existence_audit_20261005/summary.json")["tables"][f"added_by_{axis}"]
    line = f" {axis}: "
    for b in tab:
        if b["n"]:
            line += f"{b['bin']} n={b['n']} match={b['matched']/b['n']:.3f} dead_typed={b['dead_typed']} | "
    print(line)
zdead = z["typed"][added] & (z["class"][added] == 4)
print(f" dead_typed total={int(zdead.sum())}; gt_already_by_base among added="
      f"{int(z['gt_already_by_base'][added].sum())}")
sub30 = jload(f"{R}/existence_audit_20261005/summary.json")["tables"]["added_by_area"][0]
tot_m = matched[added].sum()
print(f" sub-30px: n={sub30['n']} ({sub30['n']/int(added.sum()):.1%} of additions) "
      f"match={sub30['matched']/sub30['n']:.3f} "
      f"({sub30['matched']/tot_m:.1%} of matched, dead_typed={sub30['dead_typed']}/{int(zdead.sum())})")
