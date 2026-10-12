#!/usr/bin/env python
"""Stage-report-V table numbers, extracted from frozen analysis artifacts.

Print-only. Run from the repository root with the project environment:

    python scripts/report5_numbers.py > runs/analysis/report5_numbers.txt

Sections
[A] typing-oracle ceiling diagnostic (2026-10-09): interventions and per-class view
[B] typing probe T0-T3 (2026-10-10): endpoints and pre-registered gate values
[C] E1-C detector consensus + synthetic null control (2026-10-11)
[D] E1-M two-pass model pre-review agreement (2026-10-11)
[E] E1-T cross-detector typing consensus (2026-10-12)
[F] E3 candidate-quality learning dev round (2026-10-12)

No experiment artifact is modified; nothing is re-evaluated here. The
zero-overlap consensus counts in [C] are re-derived from per_candidate.npz
and the frozen candidate records, because consensus.json stores them only
in aggregated form.
"""
import csv
import json
import pathlib

import numpy as np

R = pathlib.Path("runs/analysis")
E1C = R / "e1_consensus_20261010"


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(v, nd=1):
    return f"{100 * v:.{nd}f}%"


print("=" * 66)
print("[A] typing-oracle ceiling (runs/analysis/typing_oracle_20261009)")
print("=" * 66)
o = jload(R / "typing_oracle_20261009" / "summary.json")
print(f"{'split':<8}{'current':>10}{'oracle':>10}{'inf_conn':>10}   deltas")
for row in o["rows"]:
    print(f"{row['split']:<8}{row['current']:>10.5f}{row['oracle']:>10.5f}"
          f"{row['inf_conn']:>10.5f}   +{row['oracle'] - row['current']:.5f}"
          f" / +{row['inf_conn'] - row['current']:.5f}")
m = o["mean"]
print(f"{'mean':<8}{m['current']:>10.5f}{m['oracle']:>10.5f}{m['inf_conn']:>10.5f}"
      f"   +{m['oracle'] - m['current']:.5f} / +{m['inf_conn'] - m['current']:.5f}")
r1 = o["rows"][0]
print("\nsplit 1 per-class PQ (current -> oracle):")
for name, cur, orc in zip(o["class_names"], r1["per_class_current"], r1["per_class_oracle"]):
    print(f"  {name:<14}{cur:.4f} -> {orc:.4f}")

print()
print("=" * 66)
print("[B] typing probe T0-T3 (runs/analysis/typing_probe_20261010)")
print("=" * 66)
t = jload(R / "typing_probe_20261010" / "results.json")
print(f"baseline (T0 identity, reproduces eval_fold2): {t['baseline_mpq']:.6f}")
rows = [("T0 majority vote", t["results"]["T0"]["score"]["mpq"], None),
        ("T1 instance-mean prob", t["results"]["T1"]["score"]["mpq"],
         t["results"]["T1"]["bootstrap_vs_t0"])]
for k, lab in (("T2", "T2 pooled tp_feat + linear"), ("T3", "T3 T2 + 24px ring")):
    rows.append((lab, t["results"][k]["mean_mpq"], None))
for lab, mpq, bs in rows:
    d = mpq - t["baseline_mpq"]
    ci = f"  CI [{bs['lo']:+.5f}, {bs['hi']:+.5f}]" if bs else ""
    print(f"  {lab:<26}{mpq:.6f}  Δ{d:+.5f}{ci}")
for k in ("T2", "T3"):
    r = t["results"][k]
    per_seed = ", ".join(f"{v['score']['mpq']:.6f}" for v in r["per_seed"].values())
    print(f"  {k} per-seed mPQ: {per_seed}; Dead mean {r['mean_dead']:.5f}; "
          f"strict mean {r['mean_strict']:.6f}")
d = t["decision"]
print(f"\ngate: pass_gate={d['pass_gate']}  " + "  ".join(f"{k}={v}" for k, v in d.items()
                                                          if isinstance(v, bool)))
print(f"      values: " + "  ".join(f"{k}={v:+.5f}" for k, v in d["values"].items()))
print(f"T2 GT-region diagnostic acc {t['results']['T2']['gt_region_diag']['acc']:.4f}; "
      f"T2 matched typing acc {t['results']['T2']['matched_typed_correct']:.4f}; "
      f"T1 matched typing acc {t['results']['T1']['matched_typed_correct']:.4f}")
print(f"cost (reported, not gated): pooling {t['cost']['pool_ms_per_img']:.0f} ms/img, "
      f"head {t['cost']['head_ms_val_total']:.1f} ms/fold")

print()
print("=" * 66)
print("[C] E1-C detector consensus (runs/analysis/e1_consensus_20261010)")
print("=" * 66)
c = jload(E1C / "consensus.json")
nc = jload(E1C / "null_control.json")
print(f"jury seats: {len(c['jury'])} (CellViT {sum(m['arch'] == 'CellViT' for m in c['jury'])}"
      f" + HoVer-Net {sum(m['arch'] != 'CellViT' for m in c['jury'])}); "
      f"support threshold IoU >= {c['primary_thr']}")
print(f"null control (n={nc['n']}): mean support {nc['mean_support']:.2f}, "
      f">=3 rate {pct(nc['ge3_rate'], 2)}, <=1 rate {pct(nc['le1_rate'])}")
for src in ("S2_added", "S3_rejected"):
    s = c["per_source"][src]
    mrs = s["matched_rate_by_support"]
    lo = np.average([mrs[str(k)] for k in range(0, 3)],
                    weights=[s[f"support_hist_thr0.3"][k] for k in range(0, 3)])
    hi = np.average([mrs[str(k)] for k in range(5, len(c["jury"]) + 1)
                     if str(k) in mrs and s[f"support_hist_thr0.3"][k] > 0],
                    weights=[s[f"support_hist_thr0.3"][k] for k in range(5, len(c["jury"]) + 1)
                             if str(k) in mrs and s[f"support_hist_thr0.3"][k] > 0])
    print(f"{src}: n={s['n']}, mean support {s['mean_support']:.2f}, "
          f"match rate 0-2 {pct(lo)}, >=5 {pct(hi)}, unmatched support<=1 "
          f"{s['unmatched_le1']} ({pct(s['unmatched_le1'] / s['n'])})")

z = np.load(E1C / "per_candidate.npz")
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
assert len(added) == len(cnt) == c["n_candidates"]
for name, m in (("S2_added", added), ("S3_rejected", ~added)):
    zero = m & (ioub == 0)
    part = m & (ioub > 0) & (ioub <= 0.5)  # overlap present but below the >0.5 match line
    ge3 = (cnt >= 3) & crossfold
    print(f"{name}: zero-overlap {int(zero.sum())}, of which >=3 cross-fold/arch "
          f"{int((zero & ge3).sum())} ({pct((zero & ge3).sum() / zero.sum())}); "
          f"partial-overlap {int(part.sum())}, of which >=3 cross-fold/arch "
          f"{int((part & ge3).sum())}; mean support zero-overlap {cnt[zero].mean():.2f}")
assert int((matched & added).sum()) == 669, "accepted-candidate match count drifted from 669"

print()
print("=" * 66)
print("[D] E1-M two-pass model pre-review (e1m_analysis.json)")
print("=" * 66)
e1m = jload(E1C / "e1m_analysis.json")
for src, s in e1m["per_source"].items():
    print(f"{src}: HT(real) passA {pct(s['ht_real_r1'])} passB {pct(s['ht_real_r2'])}; "
          f"uncertain passA {pct(s['uncertain_rate_r1'])} passB {pct(s['uncertain_rate_r2'])}; "
          f"raw agreement {s['raw_agreement']:.2f}, kappa {s['kappa']:+.2f}")

print()
print("=" * 66)
print("[E] E1-T typing consensus (runs/analysis/e1t_typing_20261011)")
print("=" * 66)
e = jload(R / "e1t_typing_20261011" / "stats.json")
g = e["groups"]
print(f"groups: G1 {g['G1']}, G0 {g['G0']}, ambiguous {g['ambiguous']}, "
      f"unmatched GT {g['unmatched_gt']}, total {sum(g.values())}")
o1, o0 = e["overall"]["G1"], e["overall"]["G0"]
print(f"G1: jury majority = base {o1['maj_eq_base']} ({pct(o1['maj_eq_base'] / o1['n'])}), "
      f"= GT {o1['maj_eq_gt']} ({pct(o1['maj_eq_gt'] / o1['n'])}), "
      f"other {o1['maj_other']}, no majority {o1['no_majority']}")
print(f"G0 intrinsic flip rate: {o0['maj_neq_gt']} ({pct(o0['maj_neq_gt'] / o0['n'])})")
b = e["bootstrap"]
for k, (lo, hi) in b.items():
    print(f"  CI {k}: [{lo:.4f}, {hi:.4f}]")
print("key (GT, base) cells:")
CLASSES = {0: "untyped", 1: "Neo", 2: "Inf", 3: "Conn", 4: "Dead", 5: "Epi"}
for pair in (("2,3"), ("3,2"), ("1,2"), ("1,3"), ("3,1"), ("5,1")):
    v = e["by_pair"][pair]
    gt_c, base_c = pair.split(",")
    print(f"  GT={CLASSES[int(gt_c)]}, base={CLASSES[int(base_c)]}: n={v['n']}, "
          f"majority=base {pct(v['maj_eq_base'] / v['n'])}, =GT {pct(v['maj_eq_gt'] / v['n'])}")
print("sensitivity: ", end="")
for k in ("iou_thr_05", "min_votes_5"):
    v = e["sensitivity"][k]["overall"]["G1"]
    v0 = e["sensitivity"][k]["overall"]["G0"]
    print(f"{k}: base {pct(v['maj_eq_base'] / v['n'])} / GT {pct(v['maj_eq_gt'] / v['n'])} "
          f"(G0 flip {pct(v0['maj_neq_gt'] / v0['n'])}); ", end="")
hn = e["sensitivity"]["hn_only"]["overall"]
print(f"hn_only: base {pct(hn['G1']['maj_eq_base'] / hn['G1']['n'])} / "
      f"GT {pct(hn['G1']['maj_eq_gt'] / hn['G1']['n'])} / no-majority "
      f"{pct(hn['G1']['no_majority'] / hn['G1']['n'])} (G0 flip "
      f"{pct(hn['G0']['maj_neq_gt'] / hn['G0']['n'])})")

print()
print("=" * 66)
print("[F] E3 candidate quality (runs/analysis/e3_quality_20261011)")
print("=" * 66)
q = jload(R / "e3_quality_20261011" / "results.json")
rows = {r["name"]: r for r in q["rows"]}
print(f"model: CV AUC low-dim {q['model']['cv_auc_low']:.3f} (C={q['model']['C_low']}), "
      f"+support {q['model']['cv_auc_sup']:.3f} (C={q['model']['C_sup']})")
for name in ("identity", "a1_incumbent"):
    r = rows[name]
    print(f"{name:<14} mPQ {r['mPQ']:.5f} bPQ {r['bPQ']:.5f} Dead {r['Dead_PQ']:.5f} "
          f"strictD {r['strict_dead']:.5f} interior-matched {r['interior_matched']} "
          f"added {r['added']} unmatched {r['unmatched']}")
best_learned = max((r for n, r in rows.items() if n not in ("identity", "a1_incumbent")),
                   key=lambda r: r["strict_dead"])
print(f"best learned strict Dead {best_learned['strict_dead']:.5f} ({best_learned['name']}) "
      f"vs identity floor {rows['identity']['strict_dead']:.5f}; "
      f"A1 strict Dead {rows['a1_incumbent']['strict_dead']:.5f}")
for arm in ("A2", "A3", "A4"):
    sel = q["selected"][arm]["name"]
    gate = q["gates"][arm]
    print(f"{arm}: selected {sel}; primary={gate['primary']} secondary={gate['secondary']} "
          + " ".join(f"Δ{k}={v:+.5f}" for k, v in gate["deltas"].items())
          + f"; mPQ CI [{q['bootstrap'][arm]['mPQ']['ci95'][0]:+.5f}, "
            f"{q['bootstrap'][arm]['mPQ']['ci95'][1]:+.5f}]")
a1 = rows["a1_incumbent"]
print("descriptive (not gated), vs A1:")
for name in ("a3_p0.4", "a4_p0.5"):
    r = rows[name]
    print(f"  {name}: ΔmPQ {r['mPQ'] - a1['mPQ']:+.5f} ΔbPQ {r['bPQ'] - a1['bPQ']:+.5f} "
          f"ΔDead {r['Dead_PQ'] - a1['Dead_PQ']:+.5f} ΔstrictD {r['strict_dead'] - a1['strict_dead']:+.5f} "
          f"added {r['added']} unmatched {r['unmatched']} ({pct(r['unmatched'] / r['added'])})")
sup_rel = [b for b in q["diagnostics"]["reliability_sup"] if abs(b["mean_p"] - 0.8425) < 0.005][0]
print(f"support-arm calibration shift: fold2 bin mean p {sup_rel['mean_p']:.3f} "
      f"-> empirical rate {sup_rel['rate']:.3f}")
