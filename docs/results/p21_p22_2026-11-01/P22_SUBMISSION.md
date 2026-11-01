# P22 validation and formal submission — 1 November 2026

**Array 55469245 was submitted once at 2026-11-01T14:19:48.884174Z. Submission is not training completion.** Four fresh 130-epoch runs, mapping 0=C19 / 1=M19 / 2=C1 / 3=M1, concurrency 2; each task requests 1 V100, 4 CPU, 32 GiB and 16 hours on gpu-shared / gpu-shared-normal / jhu159 with the Lustre constraint and no requeue. The hard envelope is 64 GPU-hours; C runs stop themselves at their 12-hour runner budget and M runs at 16. No automatic expansion, Phase B, masks, fold2/fold3 evaluation, extra seeds, epochs, thresholds or rate/depth menus.

## Frozen question

Arm M is the original control C with encoder blocks 0..11 also trainable, so all 24 UNI blocks join the existing encoder AdamW group at 1e-5; patch embedding, positional embedding, class token and norm stay frozen. Everything else is identical: head 3e-4, cosine 130, weight decay .01, clip 1, BF16 autocast with FP32 focal, original focal numerator over the effective-batch object count, fold1 fit 2134 / audit 522 cohort, effective 16 / physical 4, last checkpoint, native decoder and matcher, original F1=C/T1=M admission conjunctively on seeds 19 and 1. Failure or an inconclusive result closes this exact full-depth recipe. See the [frozen design](../../superpowers/specs/2026-11-01-proposal-p22-design.md).

## Validation evidence

- LM1 tests: RED 22 failed with the module missing, then GREEN 52 passed across the P22, P21 and quota focused suites; full CPU suite **1002 passed, 3 skipped, 19073 warnings in 412.12s (0:06:52)**, exit 0, on commit a7a9efad677c with explicit worktree imports, PANNUKE_ROOT, empty CUDA, four BLAS threads and a fresh short TMPDIR. Only documentation commits followed.
- Real V100 validation **55468892 COMPLETED/0:0**, 991 s elapsed (within the 1800 s request and the 2100 s predeclared acceptance), stderr empty, 29 collected files verified by bytes and SHA256. Node checks: focused pytest; both-seed full initial state and RNG equality of P3, C and M; C versus P3 two-effective-step bitwise parity of trainable gradients, adapted parameters, AdamW state and RNG; M second-step finite nonzero gradients and updates beyond pure decay at blocks 0, 11, 12, 23 and the output convolution; trainable tensor counts C 186 / M 354 with 6 frozen encoder tensors; M trainable encoder parameters 302358528; actual BF16 autocast; exact optimizer membership of all 24 blocks; strict last/final reload; cross-arm configuration rejection; strict readiness PASS.
- Timing on the fixed 134-image fit-only set with tail 6: training 7.1776 / 10.0638 s, checkpoint 2.2941 / 3.7905 s (1853871572 / 3668184086 bytes), estimated 130-epoch 4.7108 / 6.4248 h and 1.5x safety 7.0662 / 9.6372 h against budgets 12 / 16 h; training peaks 4.0826 / 6.8034 GiB. Estimates are not completion guarantees.
- Resources at submission: allocation 1900 SU, spent 850, conservative account-wide full-walltime reservation of currently queued GPU jobs 401.00 SU, plus the 64-hour request within the allocation; quota headroom at least 100 GiB on scratch user, project user and project group; both filesystems at least 100 GiB free; `sbatch --test-only` admitted the array before the single submission.
- Storage admission inside every task uses `scripts/check_lustre_quota.py` (180 s per query, three attempts, fail-closed) before training and before preservation, adopted after the P21 Lustre incident.

- Independent review: a fresh reviewer read the runner, witness, tests, readiness validator and launchers against the original P3/P18/P19 code and found no blocking correctness problem. It noted that gradient-norm clipping at 1 spans all trainable parameters, so in M the clip coefficient applied to the shared blocks 12..23 and the head also depends on the gradients of blocks 0..11; this is inherent to the frozen recipe and is part of why M is a combined adaptation-depth/optimization recipe rather than a pure capacity test. Minor notes: the M runner budget equals the SLURM wall, so the 1.5x timing gate is the effective protection; the readiness validator is exercised end-to-end only by the smoke; JSON boolean identity comparisons rely on CPython singletons.
- Deployment note: the local SSH call of the once-only deployment timed out at 230 s while the remote sequence had already completed (snapshot, provenance, admission test, single sbatch); the exact state was inspected read-only (one smoke job by name, no remaining process) and the receipt was reconstructed from the remote `submission_receipt.json`; nothing was replayed.

## Provenance

Source commit 3af8cac4d025b7d97a1b548e3f3ef93863546bad; 463-file snapshot manifest 2ff1440699a7d23dc77b486087403896998dbeda7e2e55539714ac84cc54c492; project `/expanse/lustre/projects/jhu159/xjin10/bdd26/p22_20261101_01`; scratch `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p22_20261101_01`; validation `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p22_smoke_20261101_01`. Create-only `submission.started`, `job_id.txt` and `submission_receipt.json` exist on scratch with copies in ignored LM1 ops; never rerun the once-only deployment or submission scripts. Query the exact array ID with `sacct`/`squeue` on uncertainty. Each run saves 8 audit and 4 fit deterministic point examples at the final checkpoint.

## Queue at submission

```text
JOBID      STATE       TIME   TIME_LIMIT               NODELIST(REASON)
      55469245_[0-3%2]    PENDING       0:00     16:00:00                     (Priority)
```

These are reused fold1 point-development runs, not PQ or test performance.
