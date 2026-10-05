# Existence-filtered P2 (EF-P2): pre-registered follow-up to the matched-seed round

> Audit correction (2026-10-05): descriptive evidence and outcome wording below were
> corrected against archived records after execution (previously 43.7% matched and
> an ambiguous “26%” TP share). The frozen menu, selection, decision rule and archived
> outputs are unchanged. This annotation is not a new pre-registration.

Date: 2026-10-05. This document freezes the protocol BEFORE any EF sweep is run.
It executes the first of the recorded-only candidates from the 2026-10-03 typing
audit ("an existence-side filter for additions — the frontier is detection
precision, not type precision"), which the matched-seed verdict
(`runs/analysis/matched_seed_20261004/stats/stats.json`) made concrete: P2's Dead
gain is seed-robust (9/9 pairs, +.00742 ≈ 2.38× max within-split SD of paired deltas) but ΔbPQ −.00087
< −1× its max seed std (−.00064) fails the guard, so P2 stays a conditional
candidate. EF-P2 asks a single question: **can an existence-side filter on the
frozen P2 additions clear the bPQ guard while keeping the Dead arms alive?**

## Evidence this menu is built on (val folds ONLY — runs/analysis/existence_audit_20261005)

Descriptive audit of the 17,359 frozen-P2 val additions pooled over the 9 matched
pairs (7204/17359 = 41.50% match a GT instance at IoU>.5; the unmatched rate
is 58.50%, within the typing audit's test-side range of 57–61%):

| axis | separation found |
|---|---|
| area | **a low-precision small-object slice**: <30 px matches 20% (n=2294), 30–60 px 44%, 60–100 px 49%, ≥100 px ~40% |
| dist to nearest base instance | 39–44% matched across finite-distance bins; proximity alone is not supported as a filter |
| circularity / extent | circularity: 50–54% matched below .85 vs 35% above; extent: about 31% at .3–.7 vs 48% at ≥.7; these are marginal associations, not independent effects |
| pmax / margin | weak (37% → 42.5% across the range; partly inside the P2 menu already) |
| class | Inflammatory 60.5%, Dead 51.4%, Epithelial 46.0%, Neo/Con about 34%; Neo+Con carry 3563/7204 = 49.5% of matches, so dropping them would also lose many matches |
| gt_already_by_base | 0/17359 — an addition never re-detects a GT the base already matched (no duplicate channel) |

Also recorded: on val the frozen P2 selections show ΔbPQ −.0007…+.0020 (≈ 0, the
gate held) while the seed-mean TEST ΔbPQ was −.00087 — the tax is marginal and
partly a val→test transfer property. The audit fixes expectations: EF can only
trim the FP-richest slice (sub-30 px: 2294 candidates, 13.2% of additions,
80.2% unmatched; 454 matched candidates, 6.3% of all 7204 matches,
including 43 of the 699 correctly typed Dead matches); the verdict may land near the boundary either way.

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
   Dist/circularity/extent/pmax remain excluded from this frozen menu.
   Correction: the audit did show marginal shape/confidence associations; their
   independent usefulness was not established. This does not change the menu.
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

## Outcome 2026-10-05 ~10:10 (all stages executed to the frozen protocol)

9/9 EF val selections frozen before any test-fold read (7× a30, 1× a30d, 1× identity
for the pair whose P2 was identity); the runner's off-row guard reproduced the frozen
stage-1 rows bit-exact on all 9 pairs. **Pre-registered verdict
(`stats/stats.json`): robust_improvement = TRUE.** ΔDead PQ +.00699 ≥ 1× max per-split
seed std (.00337) with 9/9 pairs ≥ 0; ΔmPQ +.00068 and ΔbPQ −.0000091 pass their guards
(SD here means ddof=0 over within-split paired deltas). The mean bPQ loss
is reduced from −.00087 to −.0000091 while 94% of the Dead gain survives; split3's bPQ CI now straddles 0 (was excluding 0). Per the frozen
interpretation boundaries, EF-P2 replaces P2 as the reported candidate arm; the other
typing-audit candidates stay recorded-only. The 9/9 nonnegative Dead pairs include
one identity pair (zero). Passing the rule is not statistical proof of non-inferiority:
mPQ/bPQ intervals span zero, strict mPQ remains slightly lower, and dual-model
inference cost is unmeasured. Full numbers: docs/findings.md 2026-10-05.


Post-execution border-table correction: the original audit's border rows were invalid
(all zero due to an area-bin indexing error). The regenerated audit in
`runs/analysis/existence_audit_20261005_v2/` gives border 5273/9975 matched (52.86%)
and interior 1931/7384 (26.15%), with all legacy records and other tables unchanged.
These descriptive counts were obtained after the EF round and are not evidence that
was used to design its frozen menu. See findings.md for independent verification.
