# Matched-seed validation of the fair baseline / P2 (pre-registered)

Date: 2026-10-03. Executes the step recommended by `docs/P0_P3_RESULTS_2026-10-02.md` §7
("在 A100 预算恢复后做公平基线/P2 的匹配种子验证"). This document fixes the protocol BEFORE
any new result is seen; menus, selection rules and evaluator are frozen from the v2 execution.

## Goal

Replace the seed19-only point estimates of P0 (fair x1+M/S baseline) and P2 (gated conservative
additions) with matched-seed evidence: for each (split, seed) PAIR, both arms come from the same
training seed, so the P2-vs-baseline delta is paired by seed and its seed-noise can be scaled.

## Grid and missing runs

Matched pairs (x1 seed s, x2 seed s), splits × seeds {19,1,2}:

| split | x1 seeds present | x2 seeds present | new runs needed |
|---|---|---|---|
| 1 | 19, 1, 2 (`runs/cellvit_abl/split1_seed{1,2}`) | 19, 1, 2 | none |
| 2 | 19 | 19, 1, 1 (two seed-1 runs) | x1 seed1, x1 seed2, x2 seed2 |
| 3 | 19 | 19, 1, 2 | x1 seed1, x1 seed2 |

New runs (baseline config exactly = `runs/cellvit_uni/splitN` config.json, only seed/out differ;
x2 adds `upscale: 2` as in `runs/cellvit_uni_x2/*`):
- `runs/cellvit_uni/split3_seed1`, `runs/cellvit_uni/split3_seed2` (x1, 256)
- `runs/cellvit_uni/split2_seed1`, `runs/cellvit_uni/split2_seed2` (x1, 256)
- `runs/cellvit_uni_x2/split2_seed2` (x2, 512)

For the (split2, seed1) pair the canonical seed-1 x2 run is the one recorded in
`runs/analysis/p0p3_20261002_v2/remote_run_inventory.json` (first seed-1 entry); the duplicate
seed-1 run is not a third seed and is excluded from seed-mean statistics (it may be reported as
a run-level replicate, never as an independent seed).

## Protocol per (split, seed) pair — frozen from v2

1. Predictions: x1 val+test with probability tables (256, batch 2, no TTA); x2 val+test with
   tables via `--upscale 2 --decode-u 2` (du2). Inference machines: ugrad (sequential) or LM1
   when free — result-invariance by batch is already established.
2. P0: M/S menu {none,0,.25,.5,.75}×{0,20,40,80}; selection on that pair's val fold by max val
   mPQ with bPQ ≥ identity−.002, ties → less intervention. Fair baseline = selected x1+M/S.
3. P2: candidate menu {100,200}×{.5,.7,.9}×border, identity and unfiltered controls; gate =
   mPQ/bPQ ≥ reference−.002, strict Dead PQ ≥ reference, interior Dead matched increases; select
   max val mPQ, fewer additions, then name. NO_GO → identity.
4. Test: canonical `nucseg.metrics.pannuke_eval` on the test fold, last checkpoint. Report
   mPQ/bPQ/mPQ+/strict mPQ/Dead PQ/strict Dead PQ for baseline and P2 per pair.
5. Paired statistics: per-split seed-mean ± std of (P2 − baseline) on each endpoint; paired
   per-seed deltas with sign counts; tissue-stratified paired-image bootstrap (2000, seed fixed)
   for the 3-seed pooled delta per split (folds 3/3/1 are NOT pooled across splits).

## Pre-registered decision rule

- P2 is reported as a **robust improvement** only if, on 3-split seed-means: ΔDead PQ ≥ 1× the
  largest per-split seed std AND ≥ 7/9 pairs have ΔDead PQ ≥ 0, AND ΔmPQ, ΔbPQ ≥ −1× their
  largest per-split seed std.
- Otherwise P2 stays a conditional candidate, exactly as in the v2 verdict; no new menus, no
  test-driven re-selection, no post-hoc endpoint substitution.
- The M/S fair-baseline delta is reported with the same pairing (secondary endpoint).

## Compute and placement

Training on LM1 GPUs 2/3/4 only (user authorization 2026-10-03; LM2 GPU and ugrad training are
excluded). GPU3 (~34 GB free) starts first; GPU4 and GPU2 queues wait for their holders to
release memory (0%-util processes currently occupy them).
Update 2026-10-03 ~18:40 (new user authorization, superseding the placement above for the x1
chain only): with LM1 GPU4 held indefinitely, the split2 x1 pair moved to LM2 GPUs 1/2 in
parallel (~2 h; seed1 resumed from its verified last.pth). The x2 run stays on the LM1 GPU2
watcher. Same env (torch 2.5.1+cu124 A100); the protocol already tolerates RNG-stream changes
(resume precedent), so cross-machine training does not affect the matched-seed pairing.
Disk guard: a launch requires ≥ 10 GB
free on /data7 (currently ~21 GB; the disk is 100% full from other users); `last.pth` (4.4 GB,
resume-only) is deleted after each successful run, keeping `final.pth`, config, tb and logs.
~2.5–3 h per x1 run (70 s/epoch × 130); x2 longer (~512 compute). Predictions/evals afterwards.
Update 2026-10-04 06:26 (new user authorization, superseding the 18:40 decision for x2): the
LM1 GPU2 holder (77.7 GB, 0% util) had not moved for >23 h and the x2 run is the only remaining
critical path, so the split2 seed-2 x2 run moved to the idle LM2 GPU4 (fresh start, epoch 0;
~28.7 GB in use at 98% util). The LM2 copy of the v2-round duplicate was renamed
`split2_seed1_dup` first (same rename as LM1 2026-10-03) to free the output directory. The LM1
GPU2 watcher was stopped by PID and logged MIGRATED. Cross-machine move is already covered by
the resume/cross-machine precedent above. After completion: rsync back to LM1
(final.pth+config+log+tb+auto preds/eval, no last.pth), then prob-table val (fold 1) / test
(fold 3) predictions with `--upscale 2 --decode-u 2` per protocol.

## Boundaries

- Nothing in this round may read test-fold predictions before the val selections are frozen.
- The typing-audit candidates from 2026-10-03 (existence-side filter, map-vs-row abstention,
  per-class Dead thresholds) are NOT part of this round; they stay recorded-only.
- If fewer than 3 seeds land for a split (training slots never free), report the completed pairs
  and mark the split incomplete — never fill in with the duplicate seed-1 x2 run.
