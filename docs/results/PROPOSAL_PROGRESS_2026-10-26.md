# Proposal research synthesis — 2026-10-26

No tested P3–P15 proposal recipe has passed its original two-seed admission. P15 is newly re-collected and independently verified in this task; older rows summarize committed closeouts, not a new independent rehash of every prior experiment. All rows below are repeatedly reused fold1 development POINT metrics, not maskIoU/PQ, test or official three-split performance. Compare each treatment only with its own fresh control, not across rounds or by pooling seeds into an independent cohort.

|Round|Fixed treatment|Internal Dead recall change, seed19 / seed1|Original admission|
|---|---|---|---|
|P3|Adaptation+augmentation T1 versus frozen+augmentation F1|+.068966 / +.013793|false|
|P4|Dense foreground auxiliary BCE|−.027586 / +.006897|false|
|P5|Matched-capacity native-grid refinement|.000000 / −.006897|false|
|P6|Wider negative-suppression Gaussian|+.020690 / +.020690|false|
|P7|Training RGB gains|+.034483 / −.013793|false|
|P8|Per-image object normalization|−.034483 / −.048276|false|
|P9|Dead-center positive focal weight4|+.006897 / +.048276|false|
|P10|Adapt last4 instead of last12 encoder blocks|+.027586 / −.013793|false|
|P11|Redistribute exact-center positive focal mass|+.041379 / .000000|false|
|P12|Post-stem channel dropout.2|−.034483 / −.027586|false|
|P13|Replace25%training positions by rare-positive fit images|−.137931 / −.048276|false|
|P14|Encoder initialLR1e-6 instead of1e-5|−.068966 / −.089655|false|
|P15|Training integer translation +/-16, white fill|−.193103 / −.089655|false|

The table is an exploratory chronology of hypotheses with shared audit data. It is not an unbiased hyperparameter ranking, an aggregate significance test or proof that any entire method family cannot work. Individual positive point estimates or a seed-specific positive CI do not override the pre-existing conjunction of absolute recall, gain/CI, precision, overall coverage, support and nondegeneracy requirements.

Three distinguishable failure profiles matter. First, original controls and P13 fit the training images extremely well, yet their audit reliability remains low. In P13, fit internalDead recall .998476/1 accompanies audit .351724/.358621. Extra rare-positive exposure alone did not transfer. Second, restricted adaptation (P10), dropout (P12) and smaller encoder updates (P14) do not provide a reliable generalization benefit; lower effective adaptation is not automatically better. Third, P15 substantially reduces even unaugmented FIT recall (~.978→.470 overall, ~.952→.45 internalDead). That is different from high-fit/low-audit overfitting: it is consistent with insufficient learning of the tested augmented task under the fixed training budget. No profile identifies a causal bottleneck on its own.

P15 shows why precision must be interpreted with coverage. Audit precision rises .692/.698→.865/.863, but candidates fall10469/10428→5498/5397 and unique matches7241/7280→4758/4657. Audit overall recall loses20.27/21.41percentagepoints; internalDead loses19.31/8.97points, with both original95%CIs below0. The unchanged512candidate cap never activates. White borders, partial nuclei, approximately9.94%removed training center targets per epoch and stronger invariance demands all change together in this recipe; do not assign the failure exclusively to position embeddings or conclude spatial augmentation in general is harmful. [P15 full closeout](PROPOSAL_P15_RESULTS_2026-10-26.md).

The absolute support remains only11Dead-positive audit images145internalDead, with116/145objects in Lung and Colon. Tissue-image bootstrap intervals describe uncertainty conditional on this reused image collection; they do not recover independent-patient evidence, protect against sequential hypothesis selection or quantify random-seed population uncertainty. Tiny tissue cells cannot support tissue-specific rescue claims or new selection rules. No methods in this proposal sequence have earned expansion to masks or the original Phase B.

P16 isolates a previously untested objective in this sequence: direct squared error to the EXISTING Gaussian heatmap. C retains original exact-center focal supervision; R minimizes coefficient1 FP32 summed squared sigmoid error divided by full effective-batch object count. In the original focal objective, a soft Gaussian halo only suppresses nearby negative loss. In R, below-target halo outputs receive an attractive gradient. P6 only changed the negative-suppression width and P11 redistributed exact-center positive mass. Thus P16 is a distinct objective test, while changing the whole loss also changes scale and background weighting. A result cannot identify halo supervision alone as a cause or establish calibration.

P16 keeps the existing architecture/initialization, last12trainable encoder blocks, encoderLR1e-5/head3e-4, original full fit permutation, D4/color augmentation,130epochs,BF16,last checkpoint, unaugmented audit/fit evaluation, decoding and original admission. C/R×19/1 are four fresh runs with their own controls. No P15 translation, loss-scale/sigma/threshold menu, extra seeds or automatic confirmation. Original F1=C/T1=R gate is BOTH-seed; failure/inconclusive closes coefficient1/sigma1 squared-error recipe. This is an agent-selected controlled exploratory experiment under the user's instruction to progress to the next Expanse submission, not evidence of expected benefit.

Validation before spending: LM1 baseline227passed; new objective tests RED→GREEN251; fresh review identified a parameter-update acceptance gap, then7regression tests RED→GREEN258. The fixed verifier records finite nonzero encoder/head parameter deltas after two updates in BOTH seeds, plus C/P3 full-gradient/parameter/AdamW/RNG parity and independent R actual-loss/analytic-logit-gradient checks. Real V100 frozen-tensor/AMP/last+final reload/timing/source/cohort readiness must pass before formal submission. A minor witness limitation remains: below-target halo sign selection is not explicitly asserted nonempty; independent full-gradient agreement and CPU sign regression are present. Actual smoke and formal receipt belong in the [submission record](PROPOSAL_P16_SUBMISSION_2026-10-26.md).

LM1 and LM2 have no observed project training; disks are constrained, so no new training/checkpoints there. Expanse has an authenticated reusable master and original data/runtime; compute is SLURM-only. The current [SDSC guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and live account/partition/QoS/quota/queue checks were consulted. Existing unrelated jobs and all connections/data/checkpoints are preserved. No dependency upgrade, bulk data transfer, cleanup, merge or push. No metrics/HV changes;258focused tests are not a full-repository pass. Historical dense_votes failures and legacy third-party import constraints remain unaddressed, and external Lustre archival remains pending.

Sources: [prior P3–P14 synthesis](PROPOSAL_PROGRESS_2026-10-25.md) and its linked original closeouts; [P15](PROPOSAL_P15_RESULTS_2026-10-26.md); [frozen P16 design](../superpowers/specs/2026-10-26-proposal-p16-design.md).
