# P19: validated effective-batch comparison and formal submission

P19 formal array **55326699** was submitted once on **27 October 2026 at 08:50 EDT** (2026-10-27T12:50:00.008112Z). Submission is not completed training or method admission. The original UNI point model compares C effective16 with B effective32, fresh seeds19/1,130epochs, last checkpoint and unchanged original two-seed admission.

The user requested larger-batch speed testing and continued research through the next large Expanse submission. Physical microbatch16 was selected from fitting-only measurements; both scientific arms share it. No audit predictions enter hardware selection. This hypothesis was frozen after P18 results became available; it is an independent recipe, not a claim of prospective independence from all previous development evidence.

## Measured physical-batch throughput

Real V100 validation job **55326654**, exp-7-59, COMPLETED/0:0. It passed342 node checks, original C/P3 full initialization and two effective-step gradient/parameter/AdamW/RNG identity for both seeds at micro4, both-seed micro4/8/16 numerical checks (whole model and encoder/head separately), frozen-tensor identity, actual C/B278-image smoke, real AMP/updates, correct tail6/22 and strict last/final reload. Twenty-five small evidence files were returned with SHA256/byte checks. Original inputs/cohort/UNI and418 sealed source files are bound to the manifest.

| Physical batch | Median seconds /256 fit images | Images/second | Training peak GiB |
|---|---:|---:|---:|
| 4 | 13.273405 | 19.287 | 4.047 |
| 8 | 12.542721 | 20.410 | 4.882 |
| 16 | 11.822740 | 21.653 | 5.972 |

Three trials per candidate followed orders4/8/16,16/8/4,4/8/16, with initialization/input ordering reset, fitting-only warmup and CUDA synchronization. Timed work includes target generation, transfers, forward/backward, clipping and AdamW; excludes model construction, checkpointing and final evaluation. At effective16, physical16 is **12.270% greater throughput**, or **10.929% less timed training duration**, than physical4. This is a short same-node measurement, not a guarantee for130 epochs, end-to-end I/O or another GPU.

Physical-batch comparisons require relative loss error<=.01, full and per-encoder/head gradient relative L2<=.05 and cosine>=.999 across two effective steps in both seeds, with identical-zero gradients explicitly accepted. Parameter-update differences are separately recorded relative to actual updates, not dominated by pretrained weight magnitudes. All candidates passed the finite numerical checks and memory ceiling. Different BF16 kernels/reduction orders still prevent promising bitwise130-epoch trajectories. Candidate OOM/tolerance failures are rejected; a catastrophic nonfinite assertion aborts validation rather than graceful fallback. This fail-closed limitation did not occur in the finite completed validation.

Actual C/B smoke training peaks are 6.004/6.580GiB. Safety-factor1.5 estimates are 6.305/6.267hours against12hour limits. These are estimates; effective32 did not halve smoke runtime.

## Scientific configuration and interpretation

Both arms use original UNI last12, Gaussian sigma1/radius3, original exact-center focal objective divided by the complete effective-batch object count (including actual tails and empty images), original deterministic augmentation/order, headLR3e-4/encoder1e-5, AdamW betas(.9,.999)/decay.01, cosine schedule and clip1. No learning-rate scaling. The fixed fold1 fitting/audit cohort remains2134/522; seeds19/1,130epochs, fixed decoder and last checkpoint remain unchanged. No fold2/3 model evaluation, masks or PhaseB.

C has134 optimizer updates/epoch with tail6; B has67 with tail22. Gradient noise, clipping history, Adam state and cumulative weight decay therefore change. This tests an optimization recipe. Physical-batch throughput and scientific effective-batch effects are distinct; neither establishes improved Dead/PQ performance before results exist.

The original gate maps F1=C/T1=B and must pass BOTH seeds without relaxation: support>=10positive images/100objects, Dead recall>=.60, gain>=.05, original bootstrap lowerbound>0, unique precision>=.75 and>=C-.02, overall recall>=C-.02, nondegeneracy>=.95. Failure/inconclusiveness closes this fixed recipe; success only motivates a separately designed independent validation. No extra seeds, epochs, LR/threshold menus or automatic stage expansion.

## Submission, validation and preservation

Array0-3%2 maps0=C19,1=B19,2=C1,3=B1. Each requests1V100/4CPU/32GiB/12hours in gpu-shared/gpu-shared-normal/accountjhu159 with Lustre/no-requeue, formal maximum48GPU-hours. Check exactjobID/receipt before action; never replay once-only submit scripts.

Snapshot01 validation55326648 was deliberately stopped after115seconds because identical initial zero encoder gradients were falsely rejected by the numeric comparator. Its sources/logs remain unchanged. A new regression failed before the fix and passed afterward; corrected immutable snapshot02/job55326654 used a28minute cap, and115+actual completed runtime was checked<=1800seconds before formal submission. No unrelated jobs were stopped. This correction is retained rather than relabeling the first job successful.

Training commit `f02ca218c8a34a66366d5cf33ebd7cf352a1e1e7`; source manifest `49217e27e937b7ae373d6abd5c43da162c3109ca99a7c08784c3751fa2777cd1`. Final CPU verification: **919 passed, 3 skipped, 19073 warnings in 458.12s (0:07:38)**. The earlier full suite918passed/3CUDA-skipped preceded the extra zero-gradient test; the final receipt above is the publication gate. No metrics/HoVer-Net or pinned dependency changes.

Project/scratch experiment directory names are `p19_20261027_02`, validation sibling `p19_smoke_20261027_02`. Existing migration data/UNI and p3 runtime are reused; no bulk transfer, deletion or new Expanse authentication. Small results will return to LM1 with preservation hashes; large checkpoints remain on Lustre and still require external archival. Machine ownership/routes/PIDs are kept in ignored ops notes.

Resource rules were checked against the [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html) on27October2026 and live allocation/QoS/quota/queues. Node free-memory guards run inside each allocation. Scientific metrics remain reused fold1 point-development evidence, not PQ/test or independent-patient validation.
