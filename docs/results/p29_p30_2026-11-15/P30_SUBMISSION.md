# P30: middle-twelve versus last-twelve UNI updates

**Formal array 55746187 was submitted once at 2026-11-16T03:54:47.093470Z (November 15, 22:54:47 EST).** At 04:22:29 UTC all four tasks were PENDING/Priority. Mapping is 0=C19, 1=M19, 2=C1, 3=M1, concurrency two. Each requests one V100, four CPU cores, 32 GiB RAM and 11:52:00. [Formal receipt](P30_FORMAL_SUBMISSION.json) and [scheduler snapshot](P30_FORMAL_STATUS_2026-11-16.txt) record the actual submission.

CPU acceptance **55745667** completed/0:0 in 106 seconds, with all pinned imports and **121 tests in 11.95 seconds**. V100 acceptance **55745707** completed/0:0 in 424 seconds and passed **121 node tests in 105.55 seconds**, followed by all original hardware gates. The same pending acceptance JobID moved legally from gpu-preempt to gpu-debug and ran on exp-7-60; source, resources, 30-minute bound and no-requeue were preserved. [CPU evidence](P30_CPU_ACCEPTANCE.json), [GPU receipt](P30_GPU_ACCEPTANCE_RECEIPT.json), [hardware review](P30_HARDWARE_REVIEW.json), and [complete original acceptance evidence with four preliminary PNGs](p30_smoke/README.md) provide provenance.

## Frozen scientific comparison

P30 compares zero-based blocks 6–17 (arm M) with blocks 12–23 (arm C). Each updates 151,179,264 encoder parameters and 3,294,657 head parameters. The design was frozen before complete P29 seed-1 outcomes. Encoder/head learning rates 1e-5/3e-4, AdamW decay .01, the original cosine schedule, 130 epochs, seeds 19/1, cohort 2,134/522, input 256, effective batch 16/physical batch 4, focal objective, decoder, last-checkpoint evaluation rule and original two-seed decision remain unchanged. Scientific implementation: `7fecc38cff9c8eaa2456115d56204535a905c977`.

The latest LM1 focused checks passed **121 tests in 21.38 seconds**; the complete 352-file script/source/test inventory was sealed. Hardware acceptance requires exact C/P3 parity, identical starting tensors/RNG and parameter counts, actual nonzero gradients and updates beyond weight decay in middle blocks 6/17, invariant frozen suffix, BF16 forward with FP32 parameters/loss, last/final reload, source/data identity, memory and conservative timing. These requirements precede formal submission.

## Actual runtime and scheduler evidence

Acceptance **55744148** used 1,020 seconds and was cancelled before training after the original Python native-extension read stalled. Acceptance **55745221** used 655 seconds and was cancelled at 2026-11-16 02:13:35 UTC, also before training. Its node-local 3,657-source bootstrap passed; a job-local read-only CPU probe identified the blocked file as the original `torch/lib/libtorch.so`. Final scheduler state is CANCELLED/0:0. Preserve both jobs' logs, source snapshots and create-once submission records.

The recovery preserves all pinned versions: Python 3.10.21, torch 2.5.1+cu124, NumPy 1.23.5, SciPy 1.10.1, torchvision 0.20.1+cu124, timm 1.0.9 and Pillow 10.0.1. Scoped overlays contain 870 standard-library source files, 75 exact Python native extensions and 3,657 installed package sources. Package archive SHA256: `91cd738b293d79961f325931135a704907ce9aee6ef91d10e5f60f9f23071856`. Seven small Torch native files (1,199,887 bytes), including the blocked libtorch.so, match every original Expanse RECORD byte/hash/size. Native archive SHA256: `71228b2d4f874da819b8abc42b63797ce2acaa8809db303efb9df743c979af0f`; native manifest SHA256: `b92fe29c73d34fc6b191510beea51b87c5a17ced36f3f81948ee003e47107685`. Job-local extraction leaves the shared runtime intact.

The system-Python archive bootstrap removes Python-home/search/library variables before extraction. R5 also fixes missing hashlib/json imports in the formal preservation block; shell syntax, embedded Python compilation and global-name checks passed for every launcher block. Source/config identity remains required on the actual compute node.

The first two CPU preparations failed before sbatch (wrong resource name; missing explicit node count). CPU job 55745506 then ran after a legal change of its pending scheduler route and exposed a conda-forge standard-library version-parser mismatch. Job 55745560 exposed a stalled NumPy random-library read and then Pillow's missing adjacent shared-library directory; job 55745652 reproduced the latter after exact NumPy-native recovery. All failed records remain preserved.

The successful CPU job **55745667** used one node, four CPU cores, 8 GiB RAM, a five-minute bound, zero GPUs and no requeue on the preempt CPU partition. It restored the exact original conda-forge platform.py (42,211 bytes), nine NumPy random native files (3,186,312 bytes) matching every original RECORD hash/size, and original adjacent Pillow/NumPy/SciPy/torchvision/NVIDIA library-directory references. NumPy archive SHA256: `90bc3624062a7852b91af675f324c1ed794686da2632f915edbca7ffcc3bc371`; manifest SHA256: `35d30b7f00af09e238fdbfc8247b0c851a1525a259b096e5ee8714eeadd98419`. All six pinned package versions imported with zero visible GPUs, the exact source inventory was verified, and 121 focused tests passed. No new GPU allocation was consumed during these CPU corrections.

Corrected R7 source: **514 files**, manifest `73a9d313d81525eaca316cd41f0a310ac3da7310d6742412181f2c812853c58e`, published source commit `232d7f44398be41dc8521e93cf52339597953df7`. Its script/source/test identity matches the accepted CPU snapshot (`e32af843b9558b48b8f23d53a38f296517d84993d14fd6f94037ca8d42d1babe`). Three-scope quota, 100-GiB filesystem headroom, OST43 plus 150-GiB target reserve, source hashes, exact platform bytes, fresh fsync/readback, allocation headroom including the formal budget and scheduler test-only all passed immediately before the one GPU acceptance submission.

## Actual hardware acceptance and budget

For both seeds, C/P3 states matched exactly after two effective optimizer steps, and C/M initial tensors and RNG were identical. P3, C and M each have 154,473,921 trainable parameters. Middle blocks 6 and 17 had positive data gradients and updates beyond pure AdamW decay in both seeds. All frozen encoder tensors remained unchanged; BF16 forward, FP32 logits, last/final checkpoint reload, optimizer membership and source/data/cohort identity passed. Complete measurements are preserved in `p30_smoke/ready.json` and `p30_smoke/control.json`.

| Arm | Conservative 130-epoch estimate | Peak training memory |
| --- | ---: | ---: |
| C, last twelve | 7.070071 h | 4.094307 GiB |
| M, middle twelve | 7.677061 h | 4.490509 GiB |

Both conservative estimates are below the 11:52:00 scheduler bound. Actual acceptance consumed 1,020 + 655 + 424 = **2,099 seconds**. Four formal tasks request 4 x 42,720 = **170,880 seconds (47.466667 GPU hours)**. The combined upper envelope is **172,979 seconds (48.049722 GPU hours)**, below the original 174,600-second/48.5-hour package. Scientific training still requires 130 epochs and keeps its original 12-hour logical budget; scheduler cleanup may affect actual accounting.

Immediately before the actual formal submission, fresh three-scope quota, filesystem and OST43 capacity, 150-GiB target reserve, fsync/readback, source hashes, account headroom including existing queued work, queue limits and scheduler test-only passed. The receipt preserves the measured admission summary.

Project source and preserved results: `/expanse/lustre/projects/jhu159/xjin10/bdd26/p30_20261115_07`. Formal training and logs: `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p30_20261115_07`. Acceptance: the sibling `p30_smoke_20261115_07` scratch directory. The formal receipt binds the deployed source commit and 514-file manifest; later documentation commits leave that sealed snapshot intact.

## Next result review

Monitor the existing array **55746187**. After each task finishes, verify 130 continuous epochs, last-checkpoint evaluation, original source/data/cohort/seed identity, complete metrics, fixed examples and preservation hashes. Analyze both seeds under the frozen decision rule. The current four PNGs are one-epoch preliminary training-set previews; final formal examples are collected after epoch 130. Preserve all earlier attempts and P29 data, and retain create-once markers so successful collectors and submissions are never replayed.
