# P29 validation and formal submission

Formal Expanse array **55735733** was submitted once at **2026-11-14 14:31:54 EST** (2026-11-14T19:31:54.144151Z). At **14:36:16 EST**, tasks0–3 were all PENDING/Priority, with concurrency2, zero runtime and no requeue. Four fresh130-epoch runs map to C19/B19/C1/B1. Control updates blocks12..23; B updates blocks0..5 and18..23. Original split/seeds/loss/learning rates/last checkpoint and two-seed decision remain fixed.

Each task requests one V100,4CPU,32GiB and12h, gpu-shared/gpu-shared-normal/jhu159, Lustre constraint and no requeue. Formal ceiling48GPUh. The new jobs are queued; final scientific results follow their completion.

## Real V100 acceptance

Validation **55735682** completed with exit0:0 in672 seconds. LM1's focused suite passed114 tests in12.57s; the same114 tests passed on the V100 node in232.18s. Real two-seed initial-state/RNG equality and original-control two-step parity passed. Blocks0/23 had finite positive data gradients and beyond-decay updates; all frozen encoder parameters remained unchanged. Optimizer membership, BF16 forwards/FP32 logits and loss, same-arm last/final reload and cross-arm configuration rejection passed. Final readiness rechecked source/data/cohort identities and fitting-only example selection.

| Acceptance measurement | Control | First6+last6 |
|---|---:|---:|
| Conservative130-epoch time, hours | 7.1760 | 8.3878 |
| Peak training allocated memory, GiB | 4.0943 | 4.8906 |

29 validation files, including four one-epoch training previews, were collected and SHA256 verified. These previews record validation behavior on training images. Each formal run preserves eight fixed development and four training examples at epoch130.

## Source and admission

Scientific commit `562ff1731a90fe60ace4c4c4aa489d8eb13fbf7c`;505-file sealed manifest `7be3f3064ac2223e387d1cd4d1636884e05c79b82f4441105b0e7b610c06d20e`. The snapshot inherits byte-identical reference files from the verified500-file P28 parent; source and running projects were preserved. The existing pinned runtime/data/weights are reused.

The current [official Expanse guide](https://www.sdsc.edu/systems/expanse/user_guide.html), partition/QoS, allocation and queue were checked. Immediately before sbatch, three-scope storage quotas, filesystem capacity, selectedOST43 capacity/reserve, fsync/readback, source/launcher/validation hashes and scheduler test-only checks passed. Fresh accounting reported1,900SU allocation and1,491 spent; our formal maximum is48SU. Other account jobs' full-walltime upper bound250.43SU was recorded separately. Shared-account spending can change while tasks wait, and aggregate accounting changed between queries.

Source/results: `/expanse/lustre/projects/jhu159/xjin10/bdd26/p29_20261114_01`; jobs/logs: `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p29_20261114_01`. Source snapshot is in snapshot/, final preserved outputs in results/, and each task log is A_<arm>_s<seed>.log with slurm-55735733_<task>.out/err. Operational driver and receipts remain in ignored LM1 ops/p29_development_20261114. Do not replay successful once-only submission scripts. The next check uses exact array55735733, verifies130epochs/last checkpoint/preservation, and recomputes the original decision.

LM1 and LM2 received no additional GPU training. LM1 had about16GiB project free; LM2 about40GiB project/2.7GiB root free at the resource snapshot. Existing user processes and all other account jobs were preserved. Windows/LM2 brokers and the existing Expanse master were reused.
