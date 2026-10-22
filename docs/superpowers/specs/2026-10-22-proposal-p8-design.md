# P8: image versus object normalization of center focal loss

Frozen 2026-10-22 America/New_York, before reading P7 endpoint scores. The user explicitly authorizes autonomous research advancement through the next substantial Expanse submission. This bounded engineering experiment is the agent's choice within that authorization, not a claim that its hypothesis is established.

## Question and evidence

P3 memorized fit images without augmentation; its augmented trainable arm improved held-out recall but failed precision and Dead admission. P4 auxiliary foreground, P5 resolution refinement and P6 target bandwidth all failed their frozen gates. The common loss sums focal terms over an effective batch and divides by its total annotated object count. For nonempty images this weights their individually normalized losses in proportion to object count; empty images retain a nonzero negative term outside that interpretation. This arithmetic is not evidence that normalization caused the failures. P8 tests whether equal image weighting changes the held-out precision/recall tradeoff. P7 RGB augmentation is still running at freeze time; P8 does not use its scores or combine its augmentation.

Alternatives considered: further target/decoder tuning would reopen failed menus; longer training would change a closed budget; a mask architecture introduces a different research stage. Image normalization changes one explicit objective, retains architecture and inference, and can be checked analytically. It is not class balancing, tissue balancing or proof of a density mechanism. Correlations between density and Dead labels are descriptive only.

## Frozen objective

Use arms C (original control) and I (image normalized). Let N_i be the original sum of positive and negative center-focal terms for image i; n_i is its original annotation object count, even if centers collide. For actual effective batch size B (normally16, including the final partial batch):

- C: sum_i N_i / max(1,sum_i n_i). Preserve the original microbatch-wide scalar reduction exactly, including FP32 arithmetic order.
- I: (1/B) sum_i N_i/max(1,n_i). Each microbatch contributes its image terms divided by the actual whole effective B. Empty images keep the full negative focal sum with denominator1; do not drop them or normalize by pixels. This can increase empty-image influence and is an explicit part of the tested recipe.

Compute I numerator for each image with the unchanged original focal function, FP32 outside CUDA autocast. No target-based count estimate, class weights, added coefficients, clipping of per-image weights, or count thresholds. All-empty batches remain finite and learn to suppress background. Gradient accumulation must equal a full effective batch, including unequal counts, empty images and tail batches. Report loss objectives separately; their magnitudes are not quality comparisons. Invalid arm/count/batch inputs fail before silently changing the denominator.

## Protocol retained exactly

Four fresh 130-epoch runs C19/I19/C1/I1. Original fold1 fit2134/audit522, cohort SHA256 55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22, tissue SHA256 64746d127bc28e88be6fdc4639af4f21a91fb1fda00e68869ea5fe61a574f615. Original D4/brightness/contrast with SeedSequence([seed,epoch,image,987]); no P7 RGB gains. Exact original batch ordering, Gaussian sigma1/radius3 targets, UNI weights, encoder blocks13-24 trainable, encoder eval/dropout0, LayerNorm, AdamW head3e-4/encoder1e-5 betas(.9,.999),wd.01, cosine130 to10%, clip1, effective16/micro4, BF16 forward/FP32 parameters and loss. Last checkpoint only; no checkpoint/threshold/epoch/seed selection, resume or added folds. No fold2/fold3 model evaluation, automatic PhaseB, masks or three-split confirmation.

Inference unchanged: threshold.5,3x3 maxima,cap512, deterministic score/row-major order, max-cardinality unique matching within3pixels. Original admission function and bootstrap unchanged, mapping F1=C/T1=I: BOTH seeds I internalDead>=.60; I-C>=.05; original1000draw tissue-stratified paired95%CI lower>0, RNG20261018; I precision>=.75 and>=C-.02; overall recall>=C-.02; nondeg>=.95; support>=10images/100objects. Failed/inconclusive closes this exact objective at this budget. Repeated audit reuse (11images/145internalDead), image rather than patient splits and two seeds limit evidence. No PQ/test/independent-validation claim. Any pass requires a separately designed next stage, not automatic expansion.

Report fit/audit recall, unique precision, internalDead and smallDead<67, candidate counts/cap, loss/runtime/peak memory, and fit-only count distribution/empty-image fraction. Count summaries do not modify sampling, thresholds or the objective.

## Implementation and validation

New isolated remote worktree from P7 documentation HEAD acd3cc8754a0474007dcdff10349d8e3f6297fb3; main checkout and all prior changes remain. Add scripts/train_proposal_p8.py, scripts/verify_p8_control.py and tests/test_proposal_p8.py. Reuse original Proposal/Fold/evaluation/focal/admission functions; do not change metrics/HoVer-Net/shared dependencies. Add config normalization policy to manifest and checkpoints. The trainer is a scoped adaptation of P7 with original Fold for both arms; no RGB transformation path.

Tests before implementation: analytical unequal-count and empty-image values and gradients; C exact loss/gradient equality to original; I full versus micro accumulation, including3image tail and all-empty; input rejection; identical C/I original batches/evaluation; fixed arm/seed/fold/CLI/config. Run focused P3-P8 tests with explicit worktree PYTHONPATH and project-owned TMPDIR; user AGENTS permits focused checks for changes outside metrics/HoVer-Net. Disclose historical five dense_votes failures; do not claim full-suite pass.

Real V100 SLURM smoke, no audit predictions: exact initial model equality for both seeds, C versus original P3 T1 two actual effective optimizer steps, all trainable gradients/adapted parameters/AdamW state bitwise equal; both C/I oneepoch first256fit images; nonzero finite encoder/head gradients and updates, actual CUDA BF16 internal/FP32 terminal interpolation, last+final reload prediction equality and separately measured reload time. Bind data/weight/cohort/source/job-script hashes and pinned imports/versions. Training peak must be captured before reload. Each arm must satisfy unchanged 1.5*(130*training_seconds/256*2134+130*last_checkpoint_seconds+final_checkpoint_seconds+1800)<43200. Preliminary status MEASURED; smoke PASS only after budget checks; strict ready also binds parity/tests/identities. No relaxing settings or resealing old measurements after changes.

## Execution budget and provenance

Reuse migration_20261020 compact data/UNI and pinned p3 runtime; no large transfer, upgrade, deletion, synchronization or new Expanse authentication. Live-check official SDSC guide, allocation, Lustre quota, partitions/QoS and existing queue. Preserve all existing jobs. LM1 only small source/CPU/evidence, LM2 read-only (root nearly full).

New project/scratch p8_20261022_01 and p8_smoke_20261022_01. One <=30minute smoke on1V100/4CPU/32GiB via gpu-shared/gpu-shared-normal. One formal array0-3%2, C19/I19/C1/I1, each1V100/4CPU/32GiB/12h,Lustre,no-requeue: maximum48GPU-hours. Fresh launch freeGPU>=12GiB, scratch>=80GiB; atomic checkpoint disk floor20GiB+1.1tensor bytes. Strict smoke readiness before formal. Each sbatch called once after create-only guard; durable ID/receipt, inspect unknown outcomes without replay. Source/script changes invalidate readiness. Preserve complete outputs/checkpoints and SHA256 manifest in project, logs in scratch, return small artifacts with provenance to LM1; Lustre is not archival backup.

Independent design/build review and final code/script review precede sealed deployment. Commit portable code and research docs on research/proposal-p8; no merge/push. Record resources, actual test results, source commits/hashes, job IDs/paths/receipts and end connection counters. Current authorization covers this bounded choice; no extra approval stop is needed for the existing remote worktree/submit workflow.
