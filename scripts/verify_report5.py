#!/usr/bin/env python
"""Independent verification of stage-report-V numbers from the rawest artifacts.

Recomputes the report's headline numbers from frozen JSON/npz artifacts and the
official baseline evaluation summaries, cross-checks them against the frozen
experiment outputs, and emits one portable JSON snapshot to stdout:

    python scripts/verify_report5.py > docs/report/stage_report_5/verified_numbers.json

Sources (relative to the repository root):
  runs/analysis/typing_oracle_20261009/summary.json
  runs/cellvit_uni/split{1,2,3}/eval_test_fold{3,3,1}/summary.json   (baseline anchor)
  runs/analysis/typing_probe_20261010/results.json
  runs/analysis/e1_consensus_20261010/{consensus,null_control,e1m_analysis}.json,
      per_candidate.npz, e1m_pass{A,B}.csv
  runs/analysis/e1_packet_20261010/KEY/sample.json                   (E1-M source map)
  runs/analysis/e1_input_check_20261010_split1_seed1/records.npz
  runs/analysis/e1t_typing_20261011/stats.json
  runs/analysis/e3_quality_20261011/results.json

Does not modify any experiment artifact. Raises ValueError on any mismatch
beyond the stated tolerance. Bootstrap intervals are checked for internal
consistency (they bracket the frozen point estimates); they are not re-drawn.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

EPS = 1e-12
R = Path("runs/analysis")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


out = {"stage": 5, "checks": {}}

# ---------------------------------------------------------------- [A] oracle
o = jload(R / "typing_oracle_20261009" / "summary.json")
official = {}
for split, fold in ((1, 3), (2, 3), (3, 1)):
    s = jload(Path(f"runs/cellvit_uni/split{split}/eval_test_fold{fold}/summary.json"))
    official[split] = s["official"]["mPQ"]
    row = next(r for r in o["rows"] if r["split"] == split)
    require(abs(row["current"] - s["official"]["mPQ"]) <= EPS,
            f"oracle split{split} current != official eval summary")
mean = {k: sum(r[k] for r in o["rows"]) / 3 for k in ("current", "oracle", "inf_conn")}
for k in mean:
    require(abs(mean[k] - o["mean"][k]) <= EPS, f"oracle mean {k} mismatch")
require(all(r["inf_conn"] < r["oracle"] for r in o["rows"]), "inf_conn must sit below oracle")
out["checks"]["oracle"] = {
    "current_mean": mean["current"], "oracle_mean": mean["oracle"],
    "inf_conn_mean": mean["inf_conn"],
    "oracle_gain": mean["oracle"] - mean["current"],
    "inf_conn_gain": mean["inf_conn"] - mean["current"],
    "inf_conn_share_of_gain": (mean["inf_conn"] - mean["current"]) / (mean["oracle"] - mean["current"]),
    "anchored_to_official": official,
}

# ------------------------------------------------------------------- [B] T
t = jload(R / "typing_probe_20261010" / "results.json")
require(t["t0_identity"] == t["baseline_mpq"], "T0 identity != baseline")
vals = t["decision"]["values"]
recomputed = {
    "c1_delta_ge_.003": vals["delta"] >= 0.003,
    "c2_ci_lo_gt_0": vals["lo"] > 0,
    "c3_strict_not_down": vals["strict_delta"] >= 0,
    "c4_dead_within_.002": vals["dead_delta"] >= -0.002,
    "c5_beats_t1": bool(vals["beats_t1"]),
    "c6_seeds_same_direction": bool(vals["seeds_same_direction"]),
}
for k, v in recomputed.items():
    require(t["decision"][k] == v, f"typing-probe gate {k} inconsistent with stored values")
require(not t["decision"]["pass_gate"], "gate must fail given the stored values")
require(abs(t["results"]["T2"]["mean_mpq"] - t["baseline_mpq"] - vals["delta"]) <= EPS,
        "T2 delta inconsistent with mean_mpq")
require(abs(t["results"]["T1"]["score"]["mpq"] - t["baseline_mpq"]
            - t["results"]["T1"]["bootstrap_vs_t0"]["delta"]) <= EPS, "T1 delta inconsistent")
out["checks"]["typing_probe"] = {
    "baseline": t["baseline_mpq"], "T1": t["results"]["T1"]["score"]["mpq"],
    "T2_mean": t["results"]["T2"]["mean_mpq"], "T3_mean": t["results"]["T3"]["mean_mpq"],
    "gate_recomputed": recomputed, "pass_gate": False,
    "gt_region_diag_acc": t["results"]["T2"]["gt_region_diag"]["acc"],
    "t2_matched_typing": t["results"]["T2"]["matched_typed_correct"],
    "t1_matched_typing": t["results"]["T1"]["matched_typed_correct"],
}

# ----------------------------------------------------------------- [C] E1-C
e1c = R / "e1_consensus_20261010"
c = jload(e1c / "consensus.json")
nc = jload(e1c / "null_control.json")
z = np.load(e1c / "per_candidate.npz")
iou = z["iou"]
sup = iou >= c["primary_thr"]
cnt = sup.sum(1)
jf = z["jury_fold"]
ja = np.array([str(x) for x in z["jury_arch"]])
crossfold = ((sup & (jf[None, :] == 3)).sum(1) > 0) | ((sup & (ja[None, :] != "CellViT")).sum(1) > 0)
rec = np.load(R / "e1_input_check_20261010_split1_seed1" / "records.npz")
added = rec["added"].astype(bool)
ioub = rec["iou_best"]
matched = rec["matched"].astype(bool)
require(len(added) == len(cnt) == c["n_candidates"], "candidate row counts disagree")
require(int((matched & added).sum()) == 669, "accepted matched count != 669")
e1c_check = {"null_mean_support": nc["mean_support"], "null_ge3_rate": nc["ge3_rate"],
             "accepted_matched_669": 669}
for src, mask in (("S2_added", added), ("S3_rejected", ~added)):
    s = c["per_source"][src]
    require(abs(cnt[mask].mean() - s["mean_support"]) < 1e-6, f"{src} mean support mismatch")
    for k in range(len(c["jury"]) + 1):
        if str(k) in s["matched_rate_by_support"]:
            sel = mask & (cnt == k)
            require(int(sel.sum()) == s[f"support_hist_thr0.3"][k], f"{src} hist@{k} mismatch")
            if sel.any():
                require(abs(matched[sel].mean() - s["matched_rate_by_support"][str(k)]) < 1e-9,
                        f"{src} match rate @support {k} mismatch")
    zero = mask & (ioub == 0)
    part = mask & (ioub > 0) & (ioub <= 0.5)  # overlap present but below the >0.5 match line
    ge3 = (cnt >= 3) & crossfold
    e1c_check[src] = {
        "n": int(mask.sum()), "mean_support": float(cnt[mask].mean()),
        "zero_overlap": int(zero.sum()), "zero_ge3_crossfold": int((zero & ge3).sum()),
        "zero_ge3_rate": float((zero & ge3).sum() / zero.sum()),
        "partial_overlap": int(part.sum()), "partial_ge3_crossfold": int((part & ge3).sum()),
        "mean_support_zero": float(cnt[zero].mean()),
        "unmatched_le1": int((~matched[mask] & (cnt[mask] <= 1)).sum()),
    }
    require(e1c_check[src]["unmatched_le1"] == s["unmatched_le1"], f"{src} unmatched_le1 mismatch")
# null-control internal consistency
hist = np.array(nc["support_hist"], dtype=float)
require(abs((hist @ np.arange(len(hist))) / nc["n"] - nc["mean_support"]) < 1e-9,
        "null control mean support inconsistent with histogram")
require(abs(hist[3:].sum() / nc["n"] - nc["ge3_rate"]) < 1e-9, "null ge3 rate inconsistent")
out["checks"]["e1c"] = e1c_check

# ----------------------------------------------------------------- [D] E1-M
def read_rev(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = r["is_nucleus"]
    return rows


pa = read_rev(e1c / "e1m_passA.csv")
pb = read_rev(e1c / "e1m_passB.csv")
sample = jload(R / "e1_packet_20261010" / "KEY" / "sample.json")["items"]
src_of = {k: v["source"] for k, v in sample.items()}
require(set(pa) == set(pb) == set(src_of), "E1-M ids do not match the packet sample")
e1m_json = jload(e1c / "e1m_analysis.json")["per_source"]
e1m_check = {}
for src in ("S1", "S2", "S3", "S4"):
    ids = [i for i in pa if src_of[i] == src]
    a = np.array([pa[i] for i in ids])
    b = np.array([pb[i] for i in ids])
    # stored convention: raw agreement counts uncertain==uncertain as agreement
    # over all rows; kappa is the two-class Cohen's kappa on pairwise-certain rows
    agree = float((a == b).mean())
    keep = (a != "uncertain") & (b != "uncertain")
    lab = {"real": 1, "non_nucleus": 0}
    ka = np.array([lab[x] for x in a[keep]])
    kb = np.array([lab[x] for x in b[keep]])
    po = (ka == kb).mean()
    pe = ka.mean() * kb.mean() + (1 - ka.mean()) * (1 - kb.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    require(abs(agree - e1m_json[src]["raw_agreement"]) < 1e-9, f"{src} raw agreement mismatch")
    require(abs(kappa - e1m_json[src]["kappa"]) < 1e-9, f"{src} kappa mismatch")
    e1m_check[src] = {"n": len(ids), "raw_agreement": agree, "kappa": kappa,
                      "ht_real_r1": e1m_json[src]["ht_real_r1"],
                      "ht_real_r2": e1m_json[src]["ht_real_r2"]}
out["checks"]["e1m"] = e1m_check

# ----------------------------------------------------------------- [E] E1-T
e = jload(R / "e1t_typing_20261011" / "stats.json")
g = e["groups"]
require(sum(g.values()) == 59872, "E1-T group total != 59,872 fold2 GT instances")
o1, o0 = e["overall"]["G1"], e["overall"]["G0"]
require(o1["n"] == g["G1"] and o0["n"] == g["G0"], "overall/group n mismatch")
require(sum(o1.values()) - o1["n"] == 0 or sum(v for k, v in o1.items() if k != "n") == o1["n"],
        "G1 outcome counts do not sum to n")
require(sum(v for k, v in o0.items() if k != "n") == o0["n"], "G0 outcome counts do not sum to n")
pt = {"G1_maj_eq_base": o1["maj_eq_base"] / o1["n"], "G1_maj_eq_gt": o1["maj_eq_gt"] / o1["n"],
      "G0_maj_neq_gt": o0["maj_neq_gt"] / o0["n"]}
for k, v in pt.items():
    lo, hi = e["bootstrap"][k]
    require(lo <= v <= hi, f"E1-T CI for {k} does not bracket the point estimate")
require(e["bootstrap"]["G1_maj_eq_base"][0] > e["bootstrap"]["G1_maj_eq_gt"][1],
        "pre-registered key requires CI separation of base vs GT rates")
for pair in ("2,3", "3,2"):
    v = e["by_pair"][pair]
    lo, hi = e["bootstrap"][f"pair_{pair}_maj_eq_base"]
    require(lo <= v["maj_eq_base"] / v["n"] <= hi, f"pair {pair} CI mismatch")
dead_cells = [k for k in e["by_pair"] if k.split(",")[0] == "4" or k.split(",")[1] == "4"]
n_dead = sum(e["by_pair"][p]["n"] for p in dead_cells)
out["checks"]["e1t"] = {
    "G1_n": g["G1"], "G0_n": g["G0"], "unmatched_gt": g["unmatched_gt"],
    "G1_base_rate": pt["G1_maj_eq_base"], "G1_gt_rate": pt["G1_maj_eq_gt"],
    "G0_flip_rate": pt["G0_maj_neq_gt"],
    "ci_G1_base": e["bootstrap"]["G1_maj_eq_base"], "ci_G1_gt": e["bootstrap"]["G1_maj_eq_gt"],
    "pair_2_3_base": e["by_pair"]["2,3"]["maj_eq_base"] / e["by_pair"]["2,3"]["n"],
    "pair_3_2_base": e["by_pair"]["3,2"]["maj_eq_base"] / e["by_pair"]["3,2"]["n"],
    "pair_5_1_gt": e["by_pair"]["5,1"]["maj_eq_gt"] / e["by_pair"]["5,1"]["n"],
    "pair_5_1_base": e["by_pair"]["5,1"]["maj_eq_base"] / e["by_pair"]["5,1"]["n"],
    "dead_cells_total_n": n_dead,
    "sens_iou05_base": (lambda v: v["maj_eq_base"] / v["n"])(e["sensitivity"]["iou_thr_05"]["overall"]["G1"]),
    "sens_minvotes5_base": (lambda v: v["maj_eq_base"] / v["n"])(e["sensitivity"]["min_votes_5"]["overall"]["G1"]),
    "sens_hn_only_base": (lambda v: v["maj_eq_base"] / v["n"])(e["sensitivity"]["hn_only"]["overall"]["G1"]),
    "sens_hn_only_gt": (lambda v: v["maj_eq_gt"] / v["n"])(e["sensitivity"]["hn_only"]["overall"]["G1"]),
    "provenance_base_sha256_matches": sha256(
        e["provenance"]["base"]["path"]) == e["provenance"]["base"]["sha256"],
}

# ------------------------------------------------------------------ [F] E3
q = jload(R / "e3_quality_20261011" / "results.json")
rows = {r["name"]: r for r in q["rows"]}
ident = rows["identity"]
a1 = rows["a1_incumbent"]
require(q["params"]["ef_variant"] == "a30d", "incumbent variant must be a30d")
# replay the frozen EF selection rule verbatim on each learning arm
sel_expected = {}
for arm in ("A2", "A3", "A4"):
    cand = [r for r in q["rows"]
            if r["arm"] == arm and r["mPQ"] >= ident["mPQ"] - 0.002
            and r["bPQ"] >= ident["bPQ"] - 0.002
            and r["strict_dead"] >= ident["strict_dead"]
            and r["interior_matched"] > ident["interior_matched"]]
    if not cand:
        sel_expected[arm] = "identity"
    else:
        cand.sort(key=lambda r: (-r["mPQ"], r["added"], r["name"]))
        sel_expected[arm] = cand[0]["name"]
    require(q["selected"][arm]["name"] == sel_expected[arm],
            f"{arm} selection {q['selected'][arm]['name']} != replayed {sel_expected[arm]}")
learned = [r for r in q["rows"] if r["arm"] in ("A2", "A3", "A4")]
best = max(r["strict_dead"] for r in learned)
require(best < ident["strict_dead"], "a learned row passed the strict-Dead floor; gate story broken")
require(a1["strict_dead"] > ident["strict_dead"], "A1 must clear the same floor")
for arm in ("A2", "A3", "A4"):
    gate = q["gates"][arm]
    sel = q["selected"][arm]
    for k, ref in (("Dead_PQ", a1["Dead_PQ"]), ("bPQ", a1["bPQ"]),
                   ("strict_dead", a1["strict_dead"]), ("mPQ", a1["mPQ"])):
        require(abs(gate["deltas"][k] - (sel[k] - ref)) <= EPS, f"{arm} delta {k} mismatch")
    require(gate["primary"] is False and gate["secondary"] is False, f"{arm} gates must be false")
    lo, hi = q["bootstrap"][arm]["mPQ"]["ci95"]
    require(lo <= sel["mPQ"] - a1["mPQ"] <= hi, f"{arm} mPQ CI does not bracket the delta")
require(abs(q["model"]["cv_auc_sup"] - q["model"]["cv_auc_low"] - 0.0157) < 0.0005,
        "support AUC gain != +.016 (post-audit value)")
bin84 = next(b for b in q["diagnostics"]["reliability_sup"] if abs(b["mean_p"] - 0.8425) < 0.005)
require(abs(bin84["rate"] - 0.576) < 0.0005, "sup-arm .84-bin empirical rate != .58")
out["checks"]["e3"] = {
    "identity": {k: ident[k] for k in ("mPQ", "bPQ", "Dead_PQ", "strict_dead", "interior_matched")},
    "a1": {**{k: a1[k] for k in ("mPQ", "bPQ", "Dead_PQ", "strict_dead", "interior_matched")},
           "added": a1["added"], "unmatched": a1["unmatched"]},
    "selection_replay": sel_expected,
    "best_learned_strict_dead": best, "strict_floor": ident["strict_dead"],
    "cv_auc_low": q["model"]["cv_auc_low"], "cv_auc_sup": q["model"]["cv_auc_sup"],
    "gates_all_false": True,
    "gate_deltas_vs_a1": q["gates"]["A3"]["deltas"],
    "bootstrap_mPQ_ci_vs_a1": q["bootstrap"]["A3"]["mPQ"]["ci95"],
    "a4_p0.5_descriptive": {
        "d_mPQ": rows["a4_p0.5"]["mPQ"] - a1["mPQ"], "d_bPQ": rows["a4_p0.5"]["bPQ"] - a1["bPQ"],
        "d_Dead": rows["a4_p0.5"]["Dead_PQ"] - a1["Dead_PQ"],
        "d_strict": rows["a4_p0.5"]["strict_dead"] - a1["strict_dead"],
        "added": rows["a4_p0.5"]["added"], "unmatched_frac": rows["a4_p0.5"]["unmatched"] / rows["a4_p0.5"]["added"]},
    "sup_bin84_rate": bin84["rate"],
}

print(json.dumps(out, indent=1))
