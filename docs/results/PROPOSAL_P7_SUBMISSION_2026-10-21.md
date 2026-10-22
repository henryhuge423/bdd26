# P7 RGB-gain control: verified and submitted

2026-10-21, America/New_York. Submitted at 2026-10-22T02:17:04.024767Z UTC. The user requested autonomous progress through the next substantial Expanse submission. Array **55176183** was submitted exactly once after smoke **55175430** reached COMPLETED / 0:0 and produced strict ready.json. Initial scheduler status is saved with the operational evidence. Submission does not establish training completion.

All four P6 array55148533 runs completed 130 epochs. Original admission failed in both seeds: W-C internal Dead recall +.020689655 each, CI[-.02151719,.06172840]/[0,.05309735], W precision .660346/.673455 and absolute Dead recall .489655. The bandwidth recipe closes. We verified 40 collected files, 36 preservation hashes and cohort/input/weight identities; see [P6 results](PROPOSAL_P6_RESULTS_2026-10-21.md). No automatic Phase B, masks or expansion.

## Frozen experiment

Independent C/J × seeds19/1 gives four fresh 130-epoch runs, using original fold1 fit2134/audit522 and center-only sigma1/radius3 targets. During training, J adds independent per-channel U[.8,1.2] gains after original D4, brightness and contrast transforms. Evaluation is unchanged. This tests the full augmentation recipe, including brightness variation; it makes no physical stain-normalization or causal claim. Weights, model, trainable blocks, optimizer, normalization, BF16, last-checkpoint rule, decode and original admission remain fixed. Admission maps F1=C/T1=J. Failure or an inconclusive result closes this recipe. The reused 11 Dead-positive images / 145 objects are exploratory development evidence. They do not establish independent validation, PQ, test or three-split performance. No fold2/fold3 model endpoint, tuning or automatic next stage. [Design](../superpowers/specs/2026-10-21-proposal-p7-design.md), [plan](../superpowers/plans/2026-10-21-proposal-p7.md).

## Validation

LM1 baseline: 51 passed in 10.02s. New tests first produced six expected missing-implementation failures; the final P3-P7 focused suite passed 57 tests in 12.68s. Pinned interpreter and source imports were verified. Node test output is retained with smoke evidence. Independent design and code/script reviews found no Critical or Important issues. Metrics and HoVer-Net were untouched. Focused tests follow user AGENTS instructions; the full suite was not rerun, and the five historical dense_votes failures remain disclosed.

Real V100 smoke verified equal initial states in both seeds; C/P3 trainable gradients, adapted parameters and AdamW tensor/scalar states matched bit-for-bit over two effective optimizer steps. Both arms produced finite positive encoder/output gradients, parameter updates, actual CUDA BF16 internal tensors and FP32 terminal bilinear output. Last and final checkpoint reload predictions were identical. No audit predictions were generated. Timing uses the first 256 fit images, includes optimizer startup and retains the original 1.5 safety multiplier and fixed 1800s evaluation reserve. Preliminary MEASURED output cannot authorize submission: both strict smoke PASS reports, ready.json and SLURM COMPLETED / 0:0 are required.

| Arm | Safety estimate130epochs | Training peakGiB |
|---|---:|---:|
|C|7.237h|4.059|
|J|7.036h|4.059|

Estimates do not guarantee completion. Reload duration lacks a separate timing field (deferred minor); equality is checked and reload is excluded from training timing. A Windows CRLF syntax failure was corrected to LF before sealing; bash-n and embedded Python AST checks passed. No scientific job failed and no scientific setting changed during that correction.

Smoke was originally queued in gpu-debug. Both nodes were occupied, with a predicted 17-minute wait. After checking eligible shared V100 capacity, the same PENDING job 55175430 moved to gpu-shared/gpu-shared-normal. Its 30-minute limit, 1GPU/4CPU/32GiB, Lustre and no-requeue settings stayed fixed. No cancellation, resubmission or source/script mutation occurred. The override receipt is preserved; the sealed script records the original debug request and the scheduler override records actual execution provenance.

## Submission and provenance

Array0-3%2 maps 0=C19, 1=J19, 2=C1, 3=J1. It uses gpu-shared/gpu-shared-normal/jhu159, each task 1GPU/4CPU/32GiB/12h, Lustre and no-requeue; maximum48 GPU-hours. Existing native13 jobs were preserved. A create-only marker preceded the single sbatch call; job_id and receipt were saved immediately. Unknown outcomes are investigated without replay. Current SDSC guidance, allocation, QoS, quota and queue were checked. The experiment reuses migration_20261020 compact data, UNI and the p3 runtime, without downloads, dependency upgrades, large transfers or deletion.

Training commit `73190915d67f658d4573e5f601b9cdc48a0f0079`, branch research/proposal-p7, source manifest `83947384b842c3989fda71e2c6663b2fb958742e567dc113057003e23f5a9a45`. Exact script, weight, data and cohort hashes are in ready/provenance. Immutable project snapshot: `/expanse/lustre/projects/jhu159/xjin10/bdd26/p7_20261021_01`. Scratch runs: `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p7_20261021_01`. Completed results copy to project/results/<tag> with a preservation manifest. Lustre is not archival backup.

Main research/hires-adapter-e19/a0044fc and its two existing untracked research documents were preserved. No new SSH, proxy change, MFA, merge, push or history rewrite. Initial LM1/LM2 broker SSH PIDs43232/2303516 had connection/TCP/authentication counters1; final comparison is recorded in ops. These PIDs become historical evidence at handoff and always require a fresh live check.

Operational evidence is in LM1 ignored ops/p7_development_20261021 and ops/review_20261021_evening. Query exact 55176183 tasks, receipts and logs before action; never rerun once-only deploy, submit or recovery scripts. Collect small results after all four runs complete 130 epochs. Apply original admission only, with no automatic next stage.
