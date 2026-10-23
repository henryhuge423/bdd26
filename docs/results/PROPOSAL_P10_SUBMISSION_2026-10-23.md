# P10 restricted encoder adaptation submission — 2026-10-23

**Formal array 55227144 was submitted once at 2026-10-23T09:24:58.430563-04:00. Training is not yet complete.** Four fresh runs compare C/S with seeds 19/1 for 130 epochs, with at most two concurrent tasks. C adapts UNI blocks 12–23 (zero based); S adapts blocks 20–23. Both start from the same model and head. Data, loss, augmentation, optimizer, decoding and admission remain unchanged. S leaves earlier feature taps frozen, so this tests adaptation scope and its associated feature paths and parameter count. There is no P9 weighting, Phase B, mask stage or fold2/fold3 model evaluation.

V100 smoke 55226781 finished COMPLETED / 0:0. The node passed 119 focused tests. Verification covered identical initial states for both seeds; exact C/P3 gradients, adapted parameters and AdamW states after two effective optimizer steps; finite C/S gradients and updates; every frozen encoder tensor unchanged; exact optimizer membership; C block12 changing while S block12 stays fixed; internal BF16 and terminal FP32; exact predictions after same-arm last/final checkpoint reload; and rejection of wrong-arm configuration. Smoke used the first 256 fit images and made no audit predictions. The original timing formula with a 1.5 safety factor estimated C at 6.946 hours and S at 5.227 hours, below 12 hours. Training peaks were 4.071/2.278 GiB. Reload time was measured separately; runtime estimates are not guarantees.

Original admission maps F1=C and T1=S. Both seeds must have internal-Dead recall ≥.60, gain ≥.05, original bootstrap lower bound >0, precision ≥.75 and ≥C−.02, overall recall ≥C−.02, nondegeneracy ≥.95, and support of at least 10 images/100 objects. The cohort remains 2,134 fit and 522 audit images, with final checkpoints. A failed or inconclusive result closes this last-four-block recipe without searching other block counts or adding epochs/seeds. This is repeated fold1 image-level development, with 11 Dead-positive audit images/145 internal-Dead objects; it is not independent validation, PQ or test performance.

Training commit: `4163992bcd7387a96468015effd8f132d0a9f0d0`, branch `research/proposal-p10`. Source manifest: `53965cfd9c5b7b95e9c351a37c63ca9740bf6f81eb3599ac55e0a6b4b733fa16`. Strict readiness binds source, job scripts, inputs, cohort and weights; these were checked again before submission. Main, P3–P9 worktrees, existing research documents and other jobs were preserved. There was no merge/push, dependency upgrade, large transfer or cleanup. LM1 passed 119 focused tests in 13.17 seconds. Metrics/HoVer-Net were unchanged; five historical dense_votes failures remain disclosed, and no full-suite pass is claimed. Independent design/code reviews found no blockers. The S output-directory tags in the formal preflight were corrected before invocation. An unused expected-keyset variable remains in the sealed smoke script; effective checks cover frozen tensors, block coverage and optimizer identity.

Each task requests 1 V100, 4 CPUs, 32 GiB and 12 hours under gpu-shared / gpu-shared-normal / jhu159, array 0-3%2, Lustre and no-requeue: at most 48 GPU-hours. Current official guidance, allocation, quotas, disk, partitions, QoS and queue were checked. Source/results project: `/expanse/lustre/projects/jhu159/xjin10/bdd26/p10_20261023_01`. Run/log scratch: `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p10_20261023_01`. Smoke uses sibling `p10_smoke_20261023_01`. Existing compact data, UNI weights and pinned runtime are reused. Atomic-checkpoint, disk and project-quota guards remain. Lustre requires external archival backup.

Create-only guards, job IDs and receipts exist. Never rerun deployment or submission. Next inspect the exact array ID, confirm completed 130-epoch runs and 0:0 exits, verify preservation hashes and input/configuration identities, return small results and apply the original gate. Do not expand stages automatically.

Finalstateat2026-10-23T13:25:30.196099Z:
```
                 JOBID                                NAME      STATE       TIME NODELIST(REASON)
            55227144_3             bdd26-p10-encoder-scope    PENDING       0:00 (Priority)
            55227144_2             bdd26-p10-encoder-scope    PENDING       0:00 (Priority)
            55227144_1             bdd26-p10-encoder-scope    PENDING       0:00 (Priority)
            55227144_0             bdd26-p10-encoder-scope    PENDING       0:00 (Priority)
```

Official execution/storage reference: [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html). Detailedreceipts,reviewnotes,resourceevidence,smokelogsandconnectionrecovery stayinignoredops/p10_development_20261023.
