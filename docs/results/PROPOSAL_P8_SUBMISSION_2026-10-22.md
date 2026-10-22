# P8 image-normalized center focal loss: formal submission

2026-10-22 America/New_York. **Array 55191961 was submitted once at 2026-10-22T12:21:30.982366Z UTC. Submission is not training completion.** Four fresh130epoch runs C19/I19/C1/I1, concurrency2,1V100/4CPU/32GiB/12h each, gpu-shared/gpu-shared-normal/jhu159,Lustre,no-requeue; maximum48GPU-hours. No automatic expansion.

P8 was frozen before reading P7 endpoint scores. [P7 partial review](PROPOSAL_P7_PROGRESS_2026-10-22.md) reports completed seed19 and explicitly leaves seed1 pending. P3-P6 closures and the original admission remain unchanged.

## Frozen question

Original focal numerator N_i and true instance count n_i. C retains sum_i N_i/max(1,sum_i n_i), preserving microbatch scalar reduction order. I uses mean_i N_i/max(1,n_i), including full empty-image negatives, actual effective tail B and counts from annotations rather than Gaussian peak counts. This changes the objective weighting and gradient scale, not sampling or class/tissue balancing. No inferred causal density mechanism. Same original D4/brightness/contrast; no P7 RGB gains.

Original model, sigma1/radius3 targets, exact fold1 fit2134/audit522, UNI weights, encoder blocks13-24, optimizer/schedule, effective16/micro4,BF16,last checkpoint, decoder and admission are fixed. No fold2/fold3 model evaluation, threshold/checkpoint selection, extra seeds/epochs, masks or automatic PhaseB. Original admission F1=C/T1=I: both seeds Dead>=.60,delta>=.05,original1000draw tissue-stratified paired CI lower>0,precision>=.75 and>=C-.02,overall recall>=C-.02,nondeg>=.95,support>=10images/100objects. Failure/inconclusive closes this exact recipe at its budget.

Fit-only count summary: 2134images,50968objects,93empty images; count percentiles[0,25,50,75,100]=[0.0, 12.0, 20.0, 31.0, 276.0]. Descriptive only, no changed sampling or tuning. Repeated audit11Dead-positive images/145objects is exploratory development, not independent validation/test/PQ or official three-split performance.

## Validation evidence

LM1 baseline57passed13.66s;18new tests first failed with missing-trainer assertions; P3-P8 focused75passed12.00s. Node75passed29.57s. Analytical loss/gradient tests cover counts[1,3,0],colliding centers,all-empty,unequal counts and actual tail3,exact C loss/gradient and full/micro numerical agreement. Bash/embeddedPython syntax and diff checks passed. Independent design/code review had no blocking issue; a minor pre-copy project capacity gap was fixed before sealing.

Real V100 smoke 55191831 COMPLETED/0:0,stderr empty. Both seeds' initial model states agree. C versus original P3 over two actual effective optimizer steps has bitwise equal all trainable gradients,adapted parameters and complete AdamW states. Both C/I train first256fit images with finite nonzero head/encoder gradients and updates, real BF16 internal operations/FP32 terminal bilinear, equal last/final reload predictions. Reload reconstruction/verification is timed separately, outside training. Preliminary measurements MEASURED; final PASS and strict ready bind all identities and script hashes.

| Arm | Original1.5x safety estimate for130epochs(hours) | Training peak(GiB) | Reload reconstruction/verification(seconds) |
|---|---:|---:|---:|
|C|6.995|4.059|5.526|
|I|7.031|4.059|5.594|

Each estimate is below12hours under the unchanged formula; it is not a completion guarantee. Resource checks use actual GPU free memory>=12GiB,scratch>=80GiB,atomic checkpoint20GiB+1.1tensorbytes; project filesystem/group quota rechecked before copying large checkpoints. Existing P7/native13 jobs and connections were preserved.

This change does not touch metrics/HoVer-Net and uses focused verification per user AGENTS. No new full-suite pass is claimed. Historical P4 full regression588passed/3skipped/5dense_votes failures remains disclosed: `tests/test_dense_votes.py::test_dense_targets_include_every_foreground_pixel_and_singletons`, `tests/test_dense_votes.py::test_foreground_offsets_vote_back_to_predicted_common_point`, `tests/test_dense_votes.py::test_single_pixel_and_empty_foreground_are_supported_without_area_filter`, `tests/test_dense_votes.py::test_dense_vote_decoder_has_no_gt_and_returns_classed_mask`, `tests/test_dense_votes.py::test_empty_gt_losses_and_dense_head_gradients_are_finite`.

## Provenance and recovery

Trainingcommit 0a62bf8d4cc38bbe44e9ce2e992fff8fb20e6dc9; source manifest 795fb49c4ce4d947f5e19c9481ae4d79901c3a4def92a468737812a3574c8864; branch research/proposal-p8. Cohort 55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22. Original data/UNI/pinned p3 runtime reused, no dependency changes or large transfers. Source hashes and strict ready were rechecked before the single formal sbatch; create-only submission.started,job_id.txt and durable receipt exist. Do not rerun once-only scripts or submit another copy.

Operational receipts,review,validation logs and collected evidence are in ignored LM1 ops/p8_development_20261022. Project/scratch paths are in its HANDOFF. Query exact job IDs; retain failed/unfinished outputs, inspect unknown outcomes without replay. Checkpoints/results are preserved on project with SHA256 manifests and scratch logs; Lustre is not archival backup.

Current [SDSC Expanse guide](https://www.sdsc.edu/systems/expanse/user_guide.html) was rechecked this round; allocation/partition/QoS/quota were verified live. LM1/LM2 did not receive new GPU workloads. Main checkout/teammate changes remain intact; no shared merge or push.
