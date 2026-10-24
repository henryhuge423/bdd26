# P13 fixed rare-image exposure — submission, 2026-10-24

P12 decoder dropout is closed: all four fresh130epoch runs completed, both original admission gates failed; R-C internal Dead recall changes -.034483/-.027586. The40small evidence files/36preservation payloads and exact gate arithmetic were independently rechecked. [P12 final result](PROPOSAL_P12_RESULTS_2026-10-24.md).

P13 tests one fixed image exposure recipe. C is original P3 T1. S starts with the same epoch permutation, chooses floor(N/4) distinct positions independently, and replaces them with iid uniform draws from fit images containing any Dead GT, including border. No P4-P12 intervention is combined. Full2134 exposures and134 effective updates including a6image tail per epoch,130epochs,last12 adaptation,model/loss/augmentation/optimizer/lastcheckpoint/decode/admission remain frozen. Original unique fit list is used for fit metrics.

There are52eligible fit images/780Dead objects;46images contain656internal Dead. Epoch0 C covers2134unique images/52Dead-positive exposures. S seed19/1 covers1608/1613unique images and578/573Dead-positive exposures, with533replacement positions. Repeated images receive the same original image-keyed augmentation within an epoch. The comparison changes complete image/background exposure and common-image coverage; it cannot isolate a class-balance mechanism. Historical CellViT balanced-sampling negatives remain relevant but differ from this point-model recipe.

## Verification

CPU baseline153passed18.48s; final P3-P13 focused regression195passed14.98s after explicit RED/GREEN. Independent whole-diff review covered runner,real verifier,sampler/tests and deployment/SLURM guards. No metrics/HoVer-Net changes. Historical five dense_votes failures and legacy misc import limitation are disclosed; no full-suite pass is claimed.

Real V100 smoke **55262502 COMPLETED/0:0**. Bothseed P3/C/S fullinitial states and C/P3 twoeffective-update gradients/adapted parameters/AdamW/CPU+CUDA RNG match exactly. S independent actual sampled order/positive exposures,nonzero updates,frozen tensors,optimizer membership,BF16/FP32 boundary,last/final reload,config rejection,source/input/cohort/script identities and strictready pass. Base256fit contains zeroDead; S uses the fixed fullfit pool and64positive exposures. No audit predictions in smoke. Safety estimates C/S=7.156/7.044hours with original1.5factor,below12hours; estimates do not guarantee completion. Training peaks C/S=4.071/4.071GiB.

## Formal submission

Array **55262695** submitted once at **2026-10-24T18:41:22.310662-04:00 America/New_York**. Tasks0=C19,1=S19,2=C1,3=S1; array0-3%2. Each1V100/4CPU/32GiB/12hours,gpu-shared/gpu-shared-normal/jhu159,Lustre,no-requeue; maximum48GPUhours. Submission is not completed training. Create-only guards,jobID and receipt exist; never replay deploy/submit scripts. Check exact jobID and existing artifacts before any recovery.

Training/source commit `caee42bcebf81d103370c03562edc4ddd38739a4`; source manifest `ecc5974d5340cf21694b76e3a10f0944f1014d54eadff36562b2de0719c54e9a`. Branch `research/proposal-p13`. Expanse project `/expanse/lustre/projects/jhu159/xjin10/bdd26/p13_20261024_01`, scratch `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p13_20261024_01`; smoke sibling `p13_smoke_20261024_01`. Existing migration data/UNI/p3runtime reused; no dependency upgrade,bulktransfer,cleanup or new authentication. Small results later return to LM1 with preservation hashes; checkpoints remain on Expanse and Lustre externalarchive remains outstanding.

## Decision and limitations

Use original F1=C/T1=S admission: both seeds Dead>=.60,gain>=.05,original1000draw tissue-stratified bootstrap lower>0 RNG20261018,precision>=.75 and>=C-.02,overallrecall>=C-.02,nondeg>=.95,support>=10images/100objects. Failure or inconclusive closes this replacement fraction/budget recipe,without extra seeds,epochs,thresholds,sampling menus,PhaseB or masks. No fold2/fold3 model evaluation.

All scores are repeated fold1 development point metrics,not PQ,independent validation or test. Audit has522images and only11Dead-positive images/145internal Dead;80%of objects are in five Lung/Colon images. Conditional imagebootstrap does not establish seed-population uncertainty. Passing a proposal gate would still not establish complete instance segmentation performance.

Frozen [design](../superpowers/specs/2026-10-24-proposal-p13-design.md) and [plan](../superpowers/plans/2026-10-24-proposal-p13.md). Current scheduling/storage rules were checked against [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and live account/partition/QoS/quota state before submission.
