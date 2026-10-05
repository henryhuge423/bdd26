# Existence-filtered P2 (EF-P2): pre-registered follow-up to the matched-seed round

Date: 2026-10-05. This document freezes the protocol BEFORE any EF sweep is run.
It executes the first of the recorded-only candidates from the 2026-10-03 typing
audit ("an existence-side filter for additions — the frontier is detection
precision, not type precision"), which the matched-seed verdict
(`runs/analysis/matched_seed_20261004/stats/stats.json`) made concrete: P2's Dead
gain is seed-robust (9/9 pairs, +.00742 ≥ 2.4× max seed std) but ΔbPQ −.00087
< −1× its max seed std (−.00064) fails the guard, so P2 stays a conditional
candidate. EF-P2 asks a single question: **can an existence-side filter on the
frozen P2 additions clear the bPQ guard while keeping the Dead arms alive?**

## Evidence this menu is built on (val folds ONLY — runs/analysis/existence_audit_20261005)

Descriptive audit of the 17,359 frozen-P2 val additions pooled over the 9 matched
pairs (43.7% match a GT instance at IoU>.5; the FP rate 56.3% reproduces the
typing audit's test-side 57–61%):

| axis | separation found |
|---|---|
| area | **the only strong axis**: <30 px matches 20% (n=2294), 30–60 px 44%, 60–100 px 49%, ≥100 px ~40% |
| dist to nearest base instance | none: 39–44% matched in every bin (the "FPs are fragments adjacent to detected nuclei" hypothesis is FALSE) |
| circularity / extent | weak (49–54% below .85 vs 35% above — inverted); Dead additions are round |
| pmax / margin | weak (37% → 42.5% across the range; partly inside the P2 menu already) |
| class | Dead additions match best (51%), Neo/Con worst (34%) — but Neo+Con carry 47% of all TPs, so class-dropping is not viable |
| gt_already_by_base | 0/17359 — an addition never re-detects a GT the base already matched (no duplicate channel) |

Also recorded: on val the frozen P2 selections show ΔbPQ −.0007…+.0020 (≈ 0, the
gate held) while the seed-mean TEST ΔbPQ was −.00087 — the tax is marginal and
partly a val→test transfer property. The audit fixes expectations: EF can only
trim the FP-richest slice (sub-30 px, 13% of additions, 80% FP, 26% of removed
TPs incl. 43/699 Dead-typed); the verdict may land near the boundary either way.

## Frozen protocol

1. **Arms.** For every (split, seed) pair of the matched-seed round (same 9
   pairs, same predictions, same P0 baselines — no new training, no new
   inference): stage 1 = the pair's frozen P2 selection, UNCHANGED
   (`matched_seed_20261004/p2/<pair>/selection.json`, bound by input sha256);
   stage 2 = one EF filter from the menu below applied to the stage-1 additions.
2. **EF menu (frozen).** Applied per candidate to the stage-1 additions:
   - `identity` — the P0 baseline row (reference for the gate; no additions);
   - `off` — stage 1 exactly (the matched-seed P2 arm reproduced);
   - `a30` — keep additions with area ≥ 30 px;
   - `a30d` — area ≥ 30 px for non-Dead additions, no floor for Dead;
   - `a60` — keep additions with area ≥ 60 px.
   No other filters, no per-class probability thresholds, no interaction terms.
   Dist/circularity/extent/pmax are excluded because the audit found no
   separation on them.
3. **Selection (frozen, identical to the P2 round).** On each pair's val fold:
   gate mPQ, bPQ ≥ identity −.002, strict Dead PQ ≥ identity, interior Dead
   matched count > identity; then max val mPQ, fewer additions, then name.
   `select_fusion` semantics, reference = the identity row of THIS menu.
4. **Runner binding.** The EF runner recomputes the `off` row through the same
   fuse/light path and HARD-ASSERTS it equals the frozen P2 selection row
   (mPQ/bPQ/strict Dead/interior matched/added, ≤1e-12 drift) — a stage-1
   reimplementation guard. Provenance: input sha256s must match the frozen P2
   selection; both arms bound to the declared pair seed (`--expect-seed`); code
   fingerprint includes the EF filter module.
5. **Test.** After ALL 9 EF val selections are frozen: apply each pair's frozen
   (stage-1 config + EF choice) once to the test fold, canonical
   `nucseg.metrics.pannuke_eval`, last checkpoint predictions.
6. **Statistics.** `scripts/matched_seed_stats.py --arm p2ef` on the EF round
   root (P0 evals reused via the same paths). Decision rule VERBATIM from the
   matched-seed round: EF-P2 is a robust improvement iff on 3-split seed-means
   ΔDead PQ ≥ 1× largest per-split seed std AND ≥7/9 pairs ΔDead ≥ 0 AND
   ΔmPQ, ΔbPQ ≥ −1× their largest per-split seed std. Secondary (attribution):
   per-pair EF-P2 − plain-P2 deltas on all six endpoints. Tissue-stratified
   paired-image bootstrap 2000, seed 20261004, as before.
7. **Interpretation boundaries (frozen).**
   - robust_improvement = true → EF-P2 replaces P2 as the reported candidate arm.
   - false → the P2 line CLOSES with the matched-seed verdict unchanged; no
     further menus on this arm.
   - Either way: no new menus, no test-driven re-selection, no endpoint
     substitution. The other two typing-audit candidates (map-vs-row abstention,
     per-class Dead thresholds) stay recorded-only and are NOT part of this round.
   - Documented limitation: this is the third pre-registered read of these test
     folds (v2 round, matched-seed round, EF round); each round's selection is
     val-only, but cross-round test reuse is a multiplicity caveat on any
     single-round significance claim, and is reported as such.

## Compute

CPU only (fusion sweeps + canonical evals), reusing existing predictions. ~5
configs × 9 val folds (~2–4 min each), then 9 test applies + evals (~4 min each),
run as lanes via `ops/scripts/matched_seed_phase.py` with a fresh command table.
