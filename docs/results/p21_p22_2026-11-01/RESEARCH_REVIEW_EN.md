# P21 closeout and P22 rationale — 1 November 2026

All four P21 runs completed 130 epochs with SLURM `COMPLETED/0:0` and empty stderr on the completed tasks. The training-only five-class auxiliary center head (arm A) did not pass the original two-seed admission against the unchanged control (arm C). Internal-Dead matches changed 64→65 of 145 for seed 19 (+.006897, bootstrap 95% CI [−.018182, +.039604]) and 70→65 for seed 1 (−.034483, CI [−.125000, +.027273]). Absolute Dead recall (.448276 for both A runs) and unique precision (.699284/.696585) remain below the .60/.75 thresholds. The exact auxiliary recipe closes. P22 was chosen after reading these scores; it is an adaptive exploratory follow-up, not an independently proposed confirmation.

## Provenance and verification

Scheduler history. The original array 55416129 lost C19 after 37 s (`FAILED/5:0`) inside a transient Lustre quota query before any training; A19/C1/A1 were cancelled before start. Replacement array 55440148 completed A19 (7:26:54, including about 92 minutes before the training step during the Lustre incident) while C19/C1/A1 failed 4–7 minutes before training with `quota query exhausted; storage admission denied` (48-byte stderr each). After the quota helper was extended to 180 s per query, CPU gate 55450367 passed (`COMPLETED/0:0`, 330 s) and released array 55450368, which completed C19 (4:26:47), C1 (4:34:31) and A1 (4:46:03). Completed SLURM time sums to 21.24 GPU-hours; the four failed allocations add 961 s. The corrected envelope of 42.27 GPU-hours was never exceeded.

Payload checks. 70 small files were collected from the project results tree and both training roots (manifest, evaluation, fit, loss, status, summary, complete, log, launch, preservation manifest and the two coordinate JSON files per run, plus quota JSONL and launch records). Every file matched its preservation-manifest byte count and SHA256. All 438 sealed source files matched manifest `c750d7d059d8837e7dea17d9a37794e6c3f526c7d3aa13d17d5f220de2fe53a3`. The eight checkpoints matched their manifest sizes but were not independently rehashed. Identities agree across the four runs: UNI weights, fold1 input and tissue hashes, cohort `55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22` (fit 2134 / audit 522), 130 consecutive epochs with 134 updates each, effective 16 / physical 4, and run configurations equal to `run_config(arm, seed)`. The A head has exactly 325 more parameters than C. Every coordinate record carries `auxiliary_used_for_predictions=false`.

Gate reconstruction. The original `admission` function and an independent reconstruction of the 1000-draw, tissue-stratified, Dead-positive-image bootstrap (seed 20261018) agree on delta, CI and verdict for both seeds. `scripts/analyze_proposal_p21.py` reproduces the ops analysis exactly from the collected payloads.

## Results

| Run | Audit Dead /145 | Audit recall | Unique precision | Candidates | Matched | Small Dead /32 | Fit recall | Fit Dead /656 | Train h | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_19 | 64 (.441379) | .595918 | .692140 | 10547 | 7300 | 8 | .978653 | 626 | 4.0401 | 4.0474 |
| A_19 | 65 (.448276) | .582204 | .699284 | 10199 | 7132 | 7 | .979399 | 626 | 4.4055 | 4.0572 |
| C_1 | 70 (.482759) | .603184 | .695501 | 10624 | 7389 | 10 | .980399 | 640 | 4.0278 | 4.0474 |
| A_1 | 65 (.448276) | .591102 | .696585 | 10395 | 7241 | 5 | .985324 | 638 | 4.3458 | 4.0572 |

The audit set has 522 images with 12250 annotated objects, 145 interior Dead objects in 11 images and 32 small interior Dead. Fit values describe training coverage, not audit performance.

| Seed | A−C Dead recall | Paired 95% CI | Dead ≥ .60 | Gain ≥ .05 | CI lower > 0 | Precision ≥ .75 | Precision ≥ C−.02 | Recall ≥ C−.02 | Non-degeneracy | Pass |
|---|---:|---|---|---|---|---|---|---|---|---|
| 19 | +.006897 | [−.018182, +.039604] | no | no | no | no | yes (+.007144) | yes (−.013714) | yes | no |
| 1 | −.034483 | [−.125000, +.027273] | no | no | no | no | yes (+.001084) | yes (−.012082) | yes | no |

Both seeds fail the absolute Dead, gain, CI and absolute-precision clauses. The relative clauses pass, but only because A changed precision by less than a percentage point and lowered overall recall by about 1.2–1.4 points, inside the 2-point tolerance.

## Where the changes happened (descriptive only)

| Tissue | Interior Dead support (images) | C19→A19 | C1→A1 |
|---|---:|---:|---:|
| Breast | 1 (1) | 0→0 | 0→0 |
| Cervix | 15 (1) | 5→7 | 7→8 |
| Colon | 56 (2) | 17→17 | 16→18 |
| Esophagus | 8 (1) | 2→4 | 7→2 |
| Lung | 60 (3) | 38→36 | 38→35 |
| Stomach | 4 (2) | 1→0 | 1→1 |
| Uterus | 1 (1) | 1→1 | 1→1 |

Seed 19 gained two matches each in Cervix and Esophagus and lost two in Lung and one in Stomach; two Dead-positive images improved, two worsened, seven were unchanged. Seed 1 lost five of eight matches in the single Esophagus image (7→2) and four in one Lung image (11→7), while gaining one each in Colon, Cervix, Stomach and another Lung image. Dropping any one Dead-positive image moves the seed-19 net change between −.007692 and +.025000, and the seed-1 change between −.062500 and 0.000000 (the upper end is reached only by dropping the Esophagus image). These leave-one-image ranges are descriptive, not confidence intervals.

Small interior Dead did not improve: 8→7 (seed 19) and 10→5 (seed 1) of 32.

## Fitting, candidates and cost

A predicted fewer candidates on the audit set (10547→10199 and 10624→10395) and matched fewer objects (7300→7132 and 7389→7241), so precision moved by less than .008 while overall recall fell. At image level A matched more objects on 155/150 images, fewer on 209/208 and the same number on 158/164 of the 522. On the fit set A matched slightly more objects with fewer extras (49880→49918 with 84→60 extras; 49969→50220 with 79→54 extras): the auxiliary objective increased training-set coverage without transferring to held-out fold1 images.

A's reported loss sums six focal maps and is not on the same scale as C's single map; the last-ten-epoch means are .1041/.0984 (A) versus .0592/.0569 (C), and A first dropped below .2 at epochs 94/92 versus 74/73. Pure training time rose by 9.0% and 7.9% (training-second ratios 1.0904/1.0790); peak allocated memory rose from 4.0474 to 4.0572 GiB. Checkpoint writes were dominated by Lustre variability (per-epoch maxima 219.5 s for C19 and 90.3 s for A19) rather than by the arm.

## Interpretation and limits

Five extra class-specific Gaussian center maps supervised the same final 64-channel feature with coefficient 1 and the original object denominator. The auxiliary gradient reached the shared features (verified before submission), yet the main map's held-out Dead recall did not move beyond image-level noise, and the only clear changes are localized to one Esophagus and one Lung image in seed 1. We read this as: the bottleneck for interior-Dead centers on held-out fold1 images is not the absence of class information during training. This does not test larger coefficients, class balancing, inference with the auxiliary maps or different heads; those menus are closed by the protocol, and each would be a new recipe.

Support is sparse (11 Dead-positive audit images, 116 of 145 interior Dead in five Lung/Colon images), the bootstrap resamples images within tissue, and this remains repeated fold1 point development: not PQ, not a test-fold result, not patient-independent evidence. Nominal CIs are not corrected across the P3–P21 family. Large checkpoints remain on Lustre; external independent archival is still outstanding.

Operationally, the Lustre quota incident cost one failed original task and three failed replacement tasks before the 180-second helper and a CPU admission gate restored the workflow. P22 inherits that hardened admission.

## Visual examples

Each run saved eight audit and four fit point-example panels at the final checkpoint (GT-only stratified selection, radius-3 unique matching). On the Lung image 1025 (25 interior Dead, 38 objects) C19 matched 29 of 38 with 38 predictions, A19 27 with 39, C1 30 with 40, A1 29 with 42. On the Colon image 776 (7 interior Dead, 14 objects) all four runs matched 7–9 objects while predicting 17–21, illustrating the extra-candidate pattern on Dead-dense images. The two empty audit images (584, 812) received zero predictions from every run. Fit examples (Colon 782/784, Breast 0) were matched almost perfectly with no extras. Panels and coordinates are preserved with the results; a local copy accompanies this record.

## Why P22 and what it decides

P10 (adapting only the last 4 blocks) lowered overall audit recall by about .041 without a Dead gain; P14 (encoder LR 1e-6) raised precision but cut Dead recall by .069/.090. Both point in one direction: less encoder adaptation transfers less. The untested end of the axis is adapting all 24 UNI blocks at the original rates. P22 freezes arm M = C with blocks 0..11 also trainable (embeddings and norm frozen), everything else identical, C/M × seeds 19/1, four fresh 130-epoch runs, the same cohort, loss, decoder, last checkpoint and gate; M runs are budgeted at 16 hours and C at 12. Failure or an inconclusive result closes the full-depth recipe; it would not be rescued with layer-wise rates or drop-path inside this protocol.
