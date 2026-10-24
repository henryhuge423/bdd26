# P11 spatial positive focal redistribution: submission

Formal array **55244166** was submitted once at **2026-10-23T20:07:03.418066-04:00 (America/New_York)**. Submission does not mean completed training. Task0=C19,1=G19,2=C1,3=G1; four fresh130epoch runs, concurrency2. Each requests1V100/4CPU/32GiB/12hours, gpu-shared/gpu-shared-normal/jhu159, Lustre/no-requeue; formal ceiling48GPUhours. Existing unrelated jobs are preserved.

C is original exact-center focal. G redistributes each image's same positive coefficient mass over its original sigma1/radius3 Gaussian target, using y*count(y==1)/max(sum(y),1). Negative weights and true-nucleus accumulation denominator remain unchanged. Original model/head/last12 UNI adaptation/initialization/cohort/augmentation/optimizer/precision/last checkpoint/decode/admission stay frozen. Max-composed targets make this per-image redistribution, not per-object equal weighting. Conserving coefficient mass does not conserve loss or gradient magnitude.

Real V100 smoke **55243478 COMPLETED/0:0**, stderr0bytes. Node checks: ........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 143.82s (0:02:23). Two-seed full initial state equality; C/P3 two effective updates with bitwise equality of all trainable gradients, adapted parameters and AdamW states; independent G GPU loss/analytic-gradient agreement; unchanged negative gradients outside support; real augmented positive shoulders/mass conservation; finite updates/AMP/frozen tensors/optimizer membership/last+final reload/cross-arm rejection and strictready all passed. Safety-adjusted130epoch estimates: {'C': 7.197, 'G': 7.021} hours; training peaks: {'C': 4.072, 'G': 4.072} GiB. Estimates do not guarantee completion. CPU LM1 focused139passed12.21s. No metrics/HV edits or current full-suite claim; historical5dense_votesfailures remain disclosed.

The same pending smoke was moved from occupied gpu-debug nodes to permitted gpu-shared/gpu-shared-normal. Job ID, resources,30minute limit,no-requeue,source/script hashes unchanged; override receipt retained. No duplicate smoke or new authentication.

Training commit `3fa1d6550d884729c96c380b5b78a93d7ad28fbf`, source manifest `78cb80d7552091ade0630fd46f50fa3280aba7c09d037f5d3ae891d1c918a989`. Source and sbatch hashes reverified before formal submission and final handoff. Documentation commits do not modify sealed training code. No dependency upgrades, large transfers, cleanup, merge or push.

Original admission F1=C/T1=G requires both seeds: Dead>=.60, gain>=.05, frozen tissue-stratified bootstrap lower>0,precision>=.75 and>=C−.02,overallrecall>=C−.02,nondeg>=.95 and support>=10images/100objects. Failure/inconclusive closes this fixed recipe; no support/weight/threshold menu, extra seeds/epochs or automatic expansion.

This is repeatedly reused fold1 image development (fit2134/audit522;11Dead-positive images/145internalDead). It is not independent validation,PQ,test or three-split performance. No fold2/fold3 model evaluation,PhaseB or masks. P10 last4 recipe is closed; see PROPOSAL_P10_RESULTS_2026-10-23.md.

Project snapshot/results: /expanse/lustre/projects/jhu159/xjin10/bdd26/p11_20261023_01; scratchrun/logs: /expanse/lustre/scratch/xjin10/temp_project/bdd26/p11_20261023_01; smoke siblingp11_smoke_20261023_01. Existing migrationdata/UNI/p3runtime reused. Per-run status/complete/loss/log/launch files and project preservation manifests must be verified after all four tasks finish. Lustre is not an external archive; archival backup remains outstanding.

Current official [SDSCguide](https://www.sdsc.edu/systems/expanse/user_guide.html),liveallocation1900GPU SU available,queue/partitions/QoS,personal scratch and project quotas were checked. Small results/provenance are in ignored LM1 ops/p11_development_20261023 and ops/review_20261023_next.

Observed formal state:
```
JobID|State|ExitCode|Elapsed|NodeList
55244166_0|PENDING|0:0|00:00:00|None assigned
55244166_1|PENDING|0:0|00:00:00|None assigned
55244166_2|PENDING|0:0|00:00:00|None assigned
55244166_[0-3%2]|PENDING|0:0|00:00:00|None assigned
```
