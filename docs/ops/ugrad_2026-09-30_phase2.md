# 2026-09-30 evening/night — ugrad ops saga, batch A/B, lever M/S selection, seed grid start

Companion to the `docs/findings.md` entries of the same date; this file holds the operational
record. Verdict entry (lever endpoint) landed 2026-10-01 early: **PASS** — see the 2026-10-01
entry in docs/findings.md (the 00:53 pickup cron died with its session; pickup + verdict were
done manually).

## ugrad L4 hard limits — five run-killing modes, all diagnosed 2026-09-30

1. 16 GB host address space, mid-run: DefaultCPUAllocator death -> fixed by `predict_fold`
   preallocated output maps (9af4c99; constant host memory, batch-invariant by construction).
2. Same limit at the very end: `_ArrayMemoryError` at predict_cellvit.py save-time
   `inst.astype(np.int32)` — astype COPIES even when the dtype already matches (+663+158 MB
   at the worst moment) -> guarded conversion (70de7af).
3. ENFORCED per-process GPU cap ~3.6 GiB: CUDA OOM at "3.61 GiB in use, 18.44 GiB free"
   (batch 4 x2). The old CLAUDE.md "6.9 GB" was wrong — corrected in CLAUDE.md (5d11968).
   Batch 2 x2 sits at a stable 3.33 GiB; UGRAD_BS must stay 2. Batch size is
   result-invariant (see A/B below).
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is REJECTED by that cap: "CUDA driver
   error: out of memory" at model.cuda() even for 1 GiB, while plain cudaMalloc reaches
   3.5 GiB seconds earlier. Never set it on ugrad (297fe3e).
5. In-process eval on a 16 GB-VA host: eval decompresses gt_channels.npz (1.54 GiB uint16 per
   test fold) on top of the inference process's VA -> both preds died AFTER saving. Queue now
   runs `--no-eval` + a fresh-process `scripts/eval_pannuke.py` (3f8715d); evals are run
   centrally on LM2/LM1.

Ops gotcha that cost ~40 min: `pkill -f <pattern>` in the SAME ssh command as the launch line
kills the session (the whole command is one bash -c cmdline; the regex matches the unbracketed
launch text) — silent exit 255 that mimics network flapping. Bracket patterns or use two ssh
sessions; recorded in scripts/hosts_lm2.sh + memory.

LM2 -> ugrad direct ssh works via a dedicated keypair generated ON LM2
(scripts/hosts_lm2.sh); the LM1 private key never left LM1.

## Batch-size A/B (the user's "缩小 batch 不得改变结果" requirement)

runs/_ab_verdict.md: preallocation refactor bit-identical (0.00e+00 vs original at batch 32);
batch 2 vs batch 32 differs by <= 2.82e-05 on every rate metric (4 instances out of ~66k,
cudnn algorithm selection noise). UGRAD_BS=2 is result-invariant at reporting precision.

## ugrad delivery (VAL-fold x2 preds for the extended decode-menu sweeps)

- ugradv seq c (split1 fold2 du2+marker-u2) + seq d (split3 fold2 plain): BOTH preds saved,
  pulled to LM1/LM2 despite the eval-step death. ugradx seq a (split1 fold2 + split2 fold1
  plain): finished 08:39 EDT, both preds pulled to LM1+LM2.
- Central evals on LM2: split1 du2mk2, split3/split1/split2 plain-x2 VAL evals + the 3 du2
  TEST evals (they had never been computed — the lever endpoint needs them).
- Extended lever sweeps (LM1, Pool): split1 du2mk2 best = (frac .25, a_min 40) mPQ .4845
  bPQ .6486 — the SAME (0.25, 40) chosen on the du2 sweeps; split3 plain + split1/2 plain
  running.

## Lever M/S selection (VAL-only, pre-registered rule bPQ >= baseline-0.002, max mPQ)

du2 VAL sweeps (LM2): split1 .4836/.6455 -> (0.25,40) .4850/.6478; split2 .4959/.6585 ->
(0.25,40) .4971/.6603; split3 .4780/.6489 -> (0.25,40) .4798/.6514. ONE pair chosen for all
splits: **frac=0.25, a_min=40** (literal optimum on split1+split3, within .001 on split2;
bPQ increases on all three). Applied to the 3 du2 TEST preds -> *_lev.npz -> eval_test_fold
{3,3,1}_du2lev on LM2, rsynced to LM1.

du2lev TEST results so far (du2 side pending): mPQ .4881/.4872/.4978 (mean .4910), bPQ
.6592/.6592/.6641 (mean .6608), Dead PQ .1621/.1689/.2187, Dead F_c .3419/.3517/.4854,
Uterus-excluded Dead PQ .2199/.2342/.2256 (mean .2266), interior-Dead miss
.3429/.3268/.2372 (pooled 772/2539 = .3041). x1 CellViT-UNI reference: interior-Dead miss
pooled .3738, Uterus-excluded Dead PQ mean .2175 — levered x2+du2 beats x1 on both Dead
endpoints, still behind on mPQ (.4910 vs .4995) and bPQ (.6608 vs .6654).

## Seed grid (3 splits x seeds 19/1/2), split1 first

split1 TEST fold3, x2+du2 eval: seed19 = pending (the du2 test eval above), seed1 mPQ .4825
bPQ .6568 DeadPQ .1392, seed2 mPQ .4854 bPQ .6565 DeadPQ .1515 — seed spread ~.003 mPQ,
Dead PQ more seed-sensitive (.139-.152). Trainings at 23:43: split2_seed1 ep103, split3_seed1
ep121 (finish ~Oct 1 early morning), watcher-launched split2_seed2 ep63 (17:47), split3_seed2
ep23 (21:22); split1_seed1/seed2 chains DONE + evaled. Grid summation cron: 2026-10-02 10:07.
