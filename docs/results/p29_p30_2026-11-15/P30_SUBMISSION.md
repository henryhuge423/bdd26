# P30: middle-twelve versus last-twelve UNI updates

**Status following the 2026-11-16 01:26 UTC audit: formal experiment has not been submitted.** The design and CPU validation are complete. Expanse storage and hardware acceptance remain required before the four-run formal submission.

## Fixed scientific comparison

P30 compares zero-based blocks 6–17 (middle twelve, arm M) with blocks 12–23 (last twelve, arm C). Both update 151,179,264 encoder parameters and 3,294,657 head parameters. The design was frozen before full P29 seed-1 outcomes. It preserves the original encoder/head learning rates 1e-5/3e-4, AdamW decay .01, cosine schedule, 130 epochs, seeds 19/1, cohort 2,134/522, input 256, effective batch 16 and microbatch 4, focal objective, decoder, last-checkpoint rule and original two-seed decision.

The formal request remains C19/M19/C1/M1, concurrency two, each one V100, four CPU cores, 32 GiB RAM and twelve hours, for a 48 GPU-hour maximum. Hardware acceptance must show exact C/P3 parity, identical initial states and capacity, real middle-block gradients and updates beyond weight decay, unchanged frozen suffix, BF16 correctness, final reload, data/source identity, memory and timing. The scientific source commit is `7fecc38cff9c8eaa2456115d56204535a905c977`. The LM1 focused checks passed **121 tests in 17.81 seconds**; GPU checks have no completed PASS receipt yet.

## Runtime recovery and scheduler evidence

Initial smoke job **55744148** was cancelled after 1,020 seconds, before training began, after Python reads blocked in the original Lustre runtime. The scientific source and pinned dependency versions are unchanged. Scoped pure-standard-library files and 75 exact Python 3.10.21 native-extension bytes were staged; native hashes match the original conda installation metadata. The exact Python package SHA256 is `a0dbc57fe7f0ffaace6708cdb813b1c0d06b9e52dd796780e0ff989f1befa86d`.

A second scoped recovery package contains 3,657 Python source files from NumPy 1.23.5, torch 2.5.1+cu124, SciPy 1.10.1, torchvision 0.20.1+cu124, timm 1.0.9 and Pillow 10.0.1. Every copied source byte matches the original installed distribution RECORD. The 9,102,168-byte archive SHA256 is `91cd738b293d79961f325931135a704907ce9aee6ef91d10e5f60f9f23071856`. Nodes extract these sources into job-local temporary storage and link native libraries/data to the existing pinned runtime. The original shared environment is preserved. These startup checks establish file identity; the real GPU witness remains necessary.

The recovery smoke retains at most 780 seconds, so old plus new GPU acceptance stays within 1,800 seconds. A `gpu-debug` test-only request failed with “Requested node configuration is not available”; no second job was submitted by that attempt. The subsequently prepared `gpu-shared` acceptance retained the same GPU, CPU, RAM and 780-second limit. Its scratch-user quota admission timed out after 600 seconds; the driver and remote quota process subsequently exited. That attempt has no smoke receipt or formal JobID, and its logs and once-only marker are preserved.

At 2026-11-16 01:09 UTC, `sinfo` showed both debug nodes drained. Of 56 shared GPU nodes, 55 were failed, drained or draining; the reasons included “Bright: Lustre df is hung”, Lustre input/output errors and leftover-process cleanup. One shared node was mixed. The new storage query was blocked in `lfs quota`. This observed cluster state explains the remaining admission delay. At 01:26 UTC, the sole remaining shared GPU node was allocated; 55 of 56 shared nodes and both debug nodes remained unavailable. The current source, recovery attempts and failures are preserved in ignored operations records. There is no formal JobID or completed GPU readiness claim.

## Corrected bootstrap and next executable decision

The launcher review found that system Python 3.6 inherited the isolated Python 3.10 home and search path. An actual Expanse probe reproduced a startup failure (`str.isascii` unavailable, exit -6); removing interpreter-specific environment variables allowed system Python 3.6.8 to import tarfile, lzma and hashlib and exit 0. Both newly prepared launchers use this clean environment only for archive extraction. The scientific processes retain the exact pinned Python 3.10 runtime and dependency overlays. Two executed shell-environment checks, shell syntax and embedded-Python compilation passed. The new attempt is prepared but unstarted and refuses deployment while no shared GPU node is idle or mixed.

Reconcile the existing acceptance driver, source seal, quota request and submission markers before a further action. When fresh storage admission succeeds, run the bounded V100 acceptance, collect actual ready/control evidence and submit the original four-task formal request once. Preserve P29's completed model outputs and finish its outstanding scheduler/preservation audit. Report the formal receipt and scheduler state when available.
