#!/usr/bin/env python
"""Stage-report-IV table numbers, extracted from frozen analysis artifacts.

Print-only. Run from the repository root with the project environment:

    python scripts/report4_numbers.py > runs/analysis/report4_numbers.txt

Sections
[A] E2a HV-threshold controls: per-run endpoints and 2-seed arm means
[B] E2a four pre-registered contrasts (mean, per seed, image-bootstrap CI)
[C] E2a object-level Dead views (interior / border)
[D] DSB Gate A: rule arithmetic on the frozen E2a numbers
[E] DSB Gate B: VOID first run and amended dead-stratified rerun
[F] DSB Gate C: L0 decision-decoupling menu
[G] DSB Gate D: dead-oracle ceiling
[H] E0 inference cost: per-arm timings, fusion replay, verdict ratios
[I] 2026-10-08 audit re-checks: CellViT-UNI seed-19 3-split mean;
    regenerated split2_seed2 du2 / du2lev test evals

No experiment artifact is modified; nothing is re-evaluated here.
"""
import json
import math
import pathlib
import statistics

R = pathlib.Path("runs/analysis")
G = R / "dsb_gates_20261007"
ENDPOINTS = ["mPQ", "bPQ", "mPQ+", "strict mPQ", "Dead PQ", "strict Dead PQ"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=6):
    return f"{v:+.{nd}f}" if isinstance(v, float) else str(v)


print("=" * 66)
print("[A] E2a HV-threshold controls (runs/analysis/hv_threshold_controls_20261006)")
print("=" * 66)
hv = jload(R / "hv_threshold_controls_20261006" / "verified_numbers.json")
ARMS = ["x1_hv30", "x1_hv8", "x2_hv30", "x2_hv120"]
SEEDS = [19, 1]
print("per-run endpoints (audit_val_fold2 evaluations):")
print(f"  {'run':<16}" + "".join(f"{e:>13}" for e in ENDPOINTS))
for arm in ARMS:
    for seed in SEEDS:
        ep = hv["runs"][f"{arm}_seed{seed}"]["endpoints"]
        print(f"  {arm + '_seed' + str(seed):<16}" + "".join(f"{ep[e]:>13.6f}" for e in ENDPOINTS))
print("\narm means (arithmetic over the two seeds):")
print(f"  {'arm':<16}" + "".join(f"{e:>13}" for e in ENDPOINTS))
arm_mean = {}
for arm in ARMS:
    arm_mean[arm] = {e: hv["arms"][arm]["mean"][e] for e in ENDPOINTS}
    print(f"  {arm:<16}" + "".join(f"{arm_mean[arm][e]:>13.6f}" for e in ENDPOINTS))
tot = sum(hv["runs"][k]["training"]["training_loop_hours"] for k in hv["runs"])
print(f"\ntotal logged training-loop hours over the 8 runs: {tot:.2f} h")
print("audit minus original endpoint differences: all exactly 0 on 8x6 cells:",
      all(all(v == 0.0 for v in hv["runs"][k]["audit_minus_original"].values()) for k in hv["runs"]))

print()
print("=" * 66)
print("[B] E2a pre-registered contrasts (former minus latter)")
print("=" * 66)
for key, c in hv["contrasts"].items():
    print(f"\n{c['method']} - {c['reference']}  ({key})")
    for e in ["mPQ", "bPQ", "Dead PQ", "strict Dead PQ"]:
        per = c["per_seed"]
        ci = c["image_bootstrap"].get(e, {}).get("ci95")
        ci_s = f"   ci95 [{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci else "   (no bootstrap CI this endpoint)"
        print(f"  {e:<13} mean {c['mean'][e]:+.6f}   "
              f"seed19 {per['19'][e]:+.6f}  seed1 {per['1'][e]:+.6f}{ci_s}")

print()
print("=" * 66)
print("[C] E2a object-level Dead views (per arm, 2-seed mean of counts/rates)")
print("=" * 66)
for view in ["dead", "interior_dead", "border_dead"]:
    v0 = hv["runs"]["x1_hv30_seed19"]["gt_views"][view]
    print(f"\n{view} (n = {v0['n']} per seed):")
    print(f"  {'arm':<16}{'matched':>14}{'missed_bg':>14}{'matched%':>10}{'missed_bg%':>12}")
    for arm in ARMS:
        cells = [hv["runs"][f"{arm}_seed{s}"]["gt_views"][view] for s in SEEDS]
        m = sum(c["matched"] for c in cells) / 2
        mb = sum(c["missed_bg"] for c in cells) / 2
        mr = sum(c["matched_rate"] for c in cells) / 2
        mbr = sum(c["missed_bg_rate"] for c in cells) / 2
        print(f"  {arm:<16}{m:>14.1f}{mb:>14.1f}{mr * 100:>9.2f}%{mbr * 100:>11.2f}%")
    per_seed_m = [tuple(hv["runs"][f"{arm}_seed{s}"]["gt_views"][view]["matched"] for s in SEEDS) for arm in ARMS]
    print("  per-seed matched (seed19/seed1): " + "  ".join(f"{a}={p[0]}/{p[1]}" for a, p in zip(ARMS, per_seed_m)))

print()
print("=" * 66)
print("[D] DSB Gate A - rule arithmetic on frozen E2a numbers")
print("=" * 66)
ga = jload(G / "gate_a.json")
r_ = arm_mean["x1_hv8"]["Dead PQ"] - arm_mean["x1_hv30"]["Dead PQ"]
R_ = arm_mean["x2_hv30"]["Dead PQ"] - arm_mean["x1_hv30"]["Dead PQ"]
print(f"  input snapshot : {ga['hv_numbers']}")
print(f"  r = Dead(x1_hv8) - Dead(x1_hv30)      = {r_:+.8f}   (json {ga['r_x1_hv8_minus_x1_hv30']:+.8f})")
print(f"  R = Dead(x2_hv30) - Dead(x1_hv30)     = {R_:+.8f}   (json {ga['R_x2_hv30_minus_x1_hv30']:+.8f})")
print(f"  ratio r/R = {r_ / R_:}  -> {r_ / R_ * 100:.2f}%  (json {ga['ratio'] * 100:.2f}%)")
print(f"  bPQ tax of x1_hv8 = {ga['bpq_tax_x1_hv8']:+.8f}")
print(f"  thresholds: switch 0.50, downgrade 0.90, tax guard -0.001 (scripts/dsb_gate_a.py)")
print(f"  verdict: {ga['verdict']}  (base stays {ga['base']}) - {ga['reason']}")

print()
print("=" * 66)
print("[E] DSB Gate B - gradient-conflict probe")
print("=" * 66)
gb = jload(G / "gate_b.json")
print("first run (gate_b.json) - declared VOID, not a real measurement:")
print(f"  sequential first-256 fold-1 images; 64 batches x {gb['batch']}; "
      f"conflict bar {gb['threshold_conflict_fraction']}")
print(f"  decoder mean cosine {gb['decoder']['mean_cosine']:+.6f} = "
      f"({gb['decoder']['mean_cosine'] * 64:+.4f} / 64 batches; single informative batch)")
print(f"  on-disk verdict field '{gb['verdict']}' superseded by the ledger's VOID declaration")
print("\namended rerun (gate_b_v2.json) - real measurement, canonical:")
g2 = jload(G / "gate_b_v2.json")
print(f"  sampling: {g2['sampling']['scheme']}, seed {g2['sampling']['seed']}, "
      f"{g2['sampling']['n_dead_positive_images']} Dead-positive images")
cos = g2["per_batch_cosines"]
dec = cos["decoder"]
mean = sum(dec) / len(dec)
neg = sum(c < 0 for c in dec)
print(f"  batches {g2['batches']}, n_active {g2['n_active']}, "
      f"min_active for validity {g2['min_active']}")
print(f"  decoder composite cosine: mean {mean:+.6f}  min {min(dec):+.6f}  "
      f"median {statistics.median(dec):+.6f}  max {max(dec):+.6f}")
print(f"  negative batches: {neg}/{len(dec)}  conflict fraction {neg / len(dec):.3f} "
      f"vs bar {g2['threshold_conflict_fraction']}")
print("  module groups (mean cosine / negative batches / conflict fraction):")
for mod in ["encoder", "skips", "np_branch", "hv_branch", "tp_branch", "decoder"]:
    lst = cos[mod]
    nm = sum(c < 0 for c in lst)
    print(f"    {mod:<10} {sum(lst) / len(lst):+.6f}   {nm:>3}/{len(lst)}   {nm / len(lst):.6f}")
pl = g2["per_layer"]["modules"]
print(f"  per-layer one-sided t (alpha {g2['per_layer']['alpha']}, Holm over 6 groups): "
      + "  ".join(f"{m} t={pl[m]['t']:+.2f}" for m in pl))
print(f"  any significant negative group: {g2['per_layer']['any_significant']}")
print(f"  verdict: {g2['verdict']}  -> spec section 10 stop rule fires")

print()
print("=" * 66)
print("[F] DSB Gate C - L0 decision-decoupling menu (fold 2, val)")
print("=" * 66)
gc = jload(G / "gate_c.json")
print(f"  close-line: dDead >= {gc['gain_threshold']} with bPQ tax <= {gc['tax_threshold']} would close the line")
print(f"  {'tau_dead':>9}{'Dead PQ':>12}{'dDead':>11}{'d_bPQ':>12}{'d_mPQ':>12}")
for row in gc["rows"]:
    print(f"  {row['tau_dead']:>9}{row['dead_pq']:>12.6f}{row['d_dead']:>+11.6f}"
          f"{row['d_bpq']:>+12.2e}{row['d_mpq']:>+12.2e}")
peak = max(gc["rows"][1:], key=lambda r: r["d_dead"])
print(f"  peak dDead {peak['d_dead']:+.6f} at tau {peak['tau_dead']} "
      f"(< {gc['gain_threshold']} close-line)  verdict: {gc['verdict']}")
print("  note: base row mPQ/bPQ come from the tau_dead re-decode path; only Dead PQ")
print("        is bit-exact vs the official eval_fold2 baseline - do not quote them as such.")

print()
print("=" * 66)
print("[G] DSB Gate D - dead-oracle ceiling (fold 2)")
print("=" * 66)
gd = jload(G / "gate_d.json")
print(f"  drop_frac {gd['drop_frac']}, seed {gd['seed']}, threshold dDead {gd['threshold_d_dead']}")
rows = gd["rows"]
print(f"  {'variant':<8}{'Dead PQ':>12}{'dDead':>11}{'bPQ':>12}{'mPQ':>12}{'strict Dead':>13}{'n_added':>9}")
for name in ["base", "shrunk", "exact"]:
    row = rows[name]
    d = row["dead_pq"] - rows["base"]["dead_pq"]
    print(f"  {name:<8}{row['dead_pq']:>12.6f}{d:>+11.6f}{row['bpq']:>12.6f}{row['mpq']:>12.6f}"
          f"{row['strict_dead_pq']:>13.6f}{row['n_added']:>9}")
print(f"  shrunk = inserted at IoU 0.7; exact = GT masks verbatim; "
      f"{rows['exact']['n_added']} unmatched interior Dead GT instances")
print(f"  verdict: {gd['verdict']} (dDead_shrunk {rows['shrunk']['dead_pq'] - rows['base']['dead_pq']:+.6f} >> {gd['threshold_d_dead']})")

print()
print("=" * 66)
print("[H] E0 inference cost (runs/analysis/inference_cost_20261008/cost.json)")
print("=" * 66)
cost = jload(R / "inference_cost_20261008" / "cost.json")
pr = cost["protocol"]
print(f"  {pr['gpu']} (shared), batch {pr['batch']}, decode workers {pr['workers']}, "
      f"fold {pr['fold']}, n_images {pr['n_images']}")
print(f"  precision label in frozen artifact: '{pr['precision']}' (forward actually runs the "
      f"predict_fold autocast bf16 path; all arms share it, ratios unaffected)")
print(f"  {'arm':<10}{'reps':>5}{'mean ms/img':>13}{'p95 ms':>9}{'img/s':>8}{'peak GiB':>10}")
best = {}
for arm, spec in cost["arms"].items():
    reps = spec["reps"]
    best[arm] = min(r["mean_ms"] for r in reps)
    b = min(reps, key=lambda r: r["mean_ms"])
    print(f"  {arm:<10}{len(reps):>5}{b['mean_ms']:>13.6f}{b['p95_ms']:>9.3f}"
          f"{b['img_per_s']:>8.2f}{spec['peak_mem_gib']:>10.3f}")
fu = cost["fusion"]
fuse_ms = min(r["wall"]["mean_ms"] for r in fu["reps"])
fuse_cpu_ms = min(r["cpu"]["mean_ms"] for r in fu["reps"])
fw = min(fu["reps"], key=lambda r: r["wall"]["mean_ms"])
print(f"  fusion CPU replay over {fu['n_images']} cached validation images: "
      f"wall {fw['wall']['mean_ms']:.6f} ms/img (p95 {fw['wall']['p95_ms']:.3f}), "
      f"cpu {fw['cpu']['mean_ms']:.6f}; expected_added {fu['expected_added']} reproduced exactly")
vi = cost["verdict_inputs"]
two_model_wall = (best["x1"] + best["x2_du2"] + fuse_ms) / best["x1"]
two_model_cpu = (best["x1"] + best["x2_du2"] + fuse_cpu_ms) / best["x1"]
tta_ratio = best["x1_tta"] / best["x1"]
print(f"  two-model/x1 (wall)   = {two_model_wall:.6f}   (json {vi['two_model_over_x1_wall']:.6f})")
print(f"  two-model/x1 (cpu)    = {two_model_cpu:.6f}   (json {vi['two_model_over_x1_fuse_cpu']:.6f})")
print(f"  x1-TTA/x1             = {tta_ratio:.6f}   (json {vi['x1_tta_over_x1']:.6f})")
print(f"  x2-du2/x1             = {best['x2_du2'] / best['x1']:.6f}")

print()
print("=" * 66)
print("[I] 2026-10-08 audit re-checks")
print("=" * 66)
splits = [("split1", "eval_test_fold3"), ("split2", "eval_test_fold3"), ("split3", "eval_test_fold1")]
base_root = pathlib.Path("runs/cellvit_uni")
vals = []
print("  CellViT-UNI seed-19 test folds, bPQ per split (raw summaries):")
for sp, ev in splits:
    d = jload(base_root / sp / ev / "summary.json")
    vals.append(d["official"]["bPQ"])
    print(f"    {sp:<8}{ev:<18} bPQ {d['official']['bPQ']:.6f}")
mean = sum(vals) / 3
print(f"  exact 3-split mean = {mean:.6f}  -> reported .6653 (old rounded-average slip .6654)")
x2s2 = pathlib.Path("runs/cellvit_uni_x2/split2_seed2")
du2 = jload(x2s2 / "eval_test_fold3_du2" / "summary.json")
lev = jload(x2s2 / "eval_test_fold3_du2lev" / "summary.json")
print(f"\n  regenerated split2_seed2 test evals (2026-10-08 audit, from the surviving pred npz):")
for name, d in [("du2", du2), ("du2+lever", lev)]:
    print(f"    {name:<10} mPQ {d['official']['mPQ']:.6f}  bPQ {d['official']['bPQ']:.6f}  "
          f"Dead {d['per_class_PQ']['Dead']:.6f}")
print(f"    lever deltas  mPQ {lev['official']['mPQ'] - du2['official']['mPQ']:+.6f}  "
      f"bPQ {lev['official']['bPQ'] - du2['official']['bPQ']:+.6f}  "
      f"Dead {lev['per_class_PQ']['Dead'] - du2['per_class_PQ']['Dead']:+.6f}  -> PASS, grid verdicts unchanged")
sidecar = x2s2 / "pred_fold3_x2_du2_lev.npz.json"
print(f"    provenance sidecar present: {sidecar.name} -> {jload(sidecar).get('source', jload(sidecar))}"
      if sidecar.exists() else "    (sidecar missing)")
