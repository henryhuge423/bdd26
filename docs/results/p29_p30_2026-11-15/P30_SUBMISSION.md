# P30: middle-twelve versus last-twelve UNI updates

**Status at 2026-11-16 02:48 UTC: formal experiment has not been submitted.** CPU-only pinned-runtime verification **55745506** is pending in the shared queue. No completed GPU readiness receipt exists. The two cancelled GPU attempts stopped before training.

## Frozen scientific comparison

P30 compares zero-based blocks 6–17 (arm M) with blocks 12–23 (arm C). Each updates 151,179,264 encoder parameters and 3,294,657 head parameters. The design was frozen before complete P29 seed-1 outcomes. Encoder/head learning rates 1e-5/3e-4, AdamW decay .01, the original cosine schedule, 130 epochs, seeds 19/1, cohort 2,134/522, input 256, effective batch 16/physical batch 4, focal objective, decoder, final-checkpoint rule and original two-seed decision remain unchanged. Scientific implementation: `7fecc38cff9c8eaa2456115d56204535a905c977`.

The latest LM1 focused checks passed **121 tests in 21.38 seconds**; the complete 352-file script/source/test inventory was sealed. Hardware acceptance requires exact C/P3 parity, identical starting tensors/RNG and parameter counts, actual nonzero gradients and updates beyond weight decay in middle blocks 6/17, invariant frozen suffix, BF16 forward with FP32 parameters/loss, last/final reload, source/data identity, memory and conservative timing. These requirements precede formal submission.

## Actual runtime and scheduler evidence

Acceptance **55744148** used 1,020 seconds and was cancelled before training after the original Python native-extension read stalled. Acceptance **55745221** used 655 seconds and was cancelled at 2026-11-16 02:13:35 UTC, also before training. Its node-local 3,657-source bootstrap passed; a job-local read-only CPU probe identified the blocked file as the original `torch/lib/libtorch.so`. Final scheduler state is CANCELLED/0:0. Preserve both jobs' logs, source snapshots and create-once submission records.

The recovery preserves all pinned versions: Python 3.10.21, torch 2.5.1+cu124, NumPy 1.23.5, SciPy 1.10.1, torchvision 0.20.1+cu124, timm 1.0.9 and Pillow 10.0.1. Scoped overlays contain 870 standard-library source files, 75 exact Python native extensions and 3,657 installed package sources. Package archive SHA256: `91cd738b293d79961f325931135a704907ce9aee6ef91d10e5f60f9f23071856`. Seven small Torch native files (1,199,887 bytes), including the blocked libtorch.so, match every original Expanse RECORD byte/hash/size. Native archive SHA256: `71228b2d4f874da819b8abc42b63797ce2acaa8809db303efb9df743c979af0f`; native manifest SHA256: `b92fe29c73d34fc6b191510beea51b87c5a17ced36f3f81948ee003e47107685`. Job-local extraction leaves the shared runtime intact.

The system-Python archive bootstrap removes Python-home/search/library variables before extraction. R5 also fixes missing hashlib/json imports in the formal preservation block; shell syntax, embedded Python compilation and global-name checks passed for every launcher block. Source/config identity remains required on the actual compute node.

CPU-only verification **55745506** was submitted once after fresh allocation, three-scope quota, capacity and fsync/readback checks: one node, four CPU cores, 8 GiB RAM, 15 minutes, zero GPUs, no requeue. It imports all pinned packages, verifies source hashes and executes the 121 focused tests before writing cpu_ready.json. The first two CPU preparation attempts failed before actual submission (wrong resource name; missing explicit node count); their records were preserved. A read-only test of moving the same pending CPU JobID to ind-shared was rejected because jhu159 lacks that resource allocation; the original job was preserved.

## Recovery within the original GPU budget

The revised external scheduling ledger stays within the original 48.5-GPU-hour package: 1,020 + 655 prior seconds, at most 1,800 recovered-acceptance seconds, and four formal tasks at 42,720 seconds each = 174,355 seconds (48.431944 GPU hours), below 174,600 seconds. Formal mapping remains C19/M19/C1/M1 with concurrency two, each one V100, four CPU cores and 32 GiB RAM. Each external scheduler limit is 11:52:00; scientific training still requires all 130 epochs and retains its original 12-hour logical budget. Both measured conservative safety estimates must be below 42,720 seconds. The dated pretraining scheduling addendum is in the frozen design document. No additional recovery GPU allocation is included.

## Next executable decision

Collect actual CPU COMPLETED/0:0 and cpu_ready PASS; seal and admit the corrected R5 source; execute one bounded V100 acceptance allocation; inspect the complete control, learning, frozen-suffix, reload, precision, identity, memory and timing evidence. Only then perform fresh storage/account/queue/readback and scheduler admission and submit the four-task formal request once. Existing P29 outputs and all prior failed attempts remain preserved.
