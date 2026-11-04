# P23 validation and formal submission — 4 November 2026

**Formal array 55538967 submitted once at 2026-11-04T23:31:24.262351Z. Submission is not completion.**

Two fresh 130-epoch trajectories: task 0 uses seed 19 and task 1 uses seed 1. Each yields final online C and fixed EMA E. Array 0-1%2 requests one V100, 4 CPUs, 32 GiB and 12 hours per task on gpu-shared/gpu-shared-normal/jhu159, with Lustre and no requeue. The formal envelope is 24 GPUhours, plus a separate 30-minute smoke. No automatic retry, extra seeds, epochs, parameter menu, masks, Phase B or fold2/3.

## Frozen science

The original last 12 UNI blocks are trainable; embeddings and norm remain frozen. Native input 256, effective batch 16 / physical batch 4, head LR 3e-4 / encoder LR 1e-5, AdamW (.9,.999), weight decay .01, clipping at 1, cosine schedule for 130 epochs, augmentation, focal/object normalization, decoder and matcher are unchanged. Fold1 contains 2134 fit and 522 audit images; seeds remain 19 and 1. FP32 trainable-only EMA is initialized before training, with fixed decay .999, no warm-up or bias correction, and one update after every successful optimizer step including the tail. Both C/E outputs are produced after 17420 updates at epoch 130; there is no best-epoch selection. Buffers, resume and existing output directories are rejected. E is the final state of an averaging algorithm, not an instantaneous final parameter vector. The original paired admission must pass in both seeds. This is repeated fold1 development, not PQ/test performance.

## Executed validation

- Baseline22passed;27new tests first failed with absent implementation;61 focused passed. Full CPU **1029 passed, 3 skipped, 19073 warnings in 502.90s (0:08:22)** on implementation1c5ce12,explicit worktree imports,emptyCUDA,fourBLASthreads,short`/tmp/bdd23_1104`. Subsequent production change was the isolated submission-storage adapter,covered by5 new RED→GREEN tests and17 combined LM1 storage/quota tests; scientific code unchanged.
- Fresh independent whole-branch review found no Critical/core-science blocker. Important finding: pre-submission quota was weaker than runtime admission. Corrected both once-only drivers to call the sealed hardened checker for scratch-user,project-user,project-group before sbatch;smoke before snapshot copy. Soft/hard block limits,inodes,inaccurate output and failed queries fail closed. No deferred minor findings. Actual hardware/resources were explicitly reserved for execution verification.
- V100 smoke **55536183 COMPLETED/0:0**,elapsed1280seconds,within30minute request/2100second acceptance;stderr empty;32collected files verified. Both seeds: initial whole-state/RNG equality and two-step parity of gradients, parameters, AdamW and RNG with originalP3 while EMA executes;analytical shadow equality,finite/changed/distinct shadows,clone and online/RNG isolation. Actual BF16,unchanged frozen weights,strict finalC/finalE/lastC/lastE reload predictions,9 steps including a 6-image tail,strict source/input/cohort/readinessPASS.
- Fixed 134-image fit-only smoke:training7.2172s,checkpoint2.8436s;130epoch estimate5.2634h,1.5x safety7.8951h<12h;training peak4.6232GiB<=24GiB. Includes EMA update/save overhead and3600s allowance for two final evaluations. Estimates do not guarantee completion.12 deterministic fit previews are preliminary;formal outputs will save48 final images total.
- Admission:allocation1900SU,spent932,conservative account-wide queuedGPU full-walltime reservation24.00SU plus24within allocation. Three quota scopes and actual filesystems passed100GiB/1000file admission. Each task repeats strict80GiB admission before training and preservation. `sbatch --test-only` passed before one create-only guarded submission.

## Provenance and operations

Source commit **9f2075b5a14e202c73fe804acf0ab4549aeee2a0**;471source files;manifest **830f3cdd44db1a17387a0e419137124257c0e719ff3a21fd9b037527e4172470**.

Project:`/expanse/lustre/projects/jhu159/xjin10/bdd26/p23_20261104_01`. Scratch:`/expanse/lustre/scratch/xjin10/temp_project/bdd26/p23_20261104_01`. Smoke:`/expanse/lustre/scratch/xjin10/temp_project/bdd26/p23_smoke_20261104_01`.

Create-only submission.started,job_id.txt,submission_receipt.json exist;copies and full validation logs are in ignored LM1`ops/p23_development_20261104`. Never replay submission on uncertainty:inspect exact job IDs and receipts. Last checkpoints include optimizer+EMA+count for provenance,not supported resume. Preserve other jobs/connections. Large checkpoint external archive remains outstanding.

Windows execution broker failed during development. The first controlled recovery authenticated but did not obtain a dispatcher response. Following explicit new user authorization,a single additional recovery succeeded. Unknown requests and files were checked absent before fresh deployment. No other user SSH was stopped,no proxy/identity/port changed. The root cause of the earlier channel failure remains unestablished.
