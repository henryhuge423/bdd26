# P14 fixed encoder learning rate: negative development result

All four fresh C/L runs with seeds 19 and 1 in array 55270432 completed 130 epochs with SLURM status COMPLETED/0:0. Total allocated one-GPU walltime was 17h06m38s. The original last-checkpoint evaluation and admission were recomputed from the preserved rows; both seeds failed. The fixed encoder LR 1e-6 recipe is closed. There are no additional LR, seed, epoch, threshold, Phase B, or mask trials in this result.

| Seed | C internal Dead | L internal Dead | L−C | Original paired 95% CI | C/L overall recall | C/L unique precision |
|---|---:|---:|---:|---|---|---|
| 19 | 74/145=.510345 | 64/145=.441379 | −.068966 | [−.114583,+.021517] | .583429/.495102 | .689067/.776569 |
| 1 | 69/145=.475862 | 56/145=.386207 | −.089655 | [−.154545,−.049451] | .595755/.475673 | .697172/.786264 |

L clears the absolute precision floor, but both seeds fail the Dead recall ≥.60, gain ≥.05, strictly positive bootstrap lower-bound, and relative overall recall ≥C−.02 requirements. The nondegenerate fraction is 1, and support of 11 positive images/145 objects passes. Seed 19's interval includes zero; seed 1's interval is entirely negative. The two-seed mean Dead change of −.079310 is descriptive, not a pooled significance estimate.

| Run | Fit recall | Fit Dead recall | Fit precision | Audit candidates | Matched/12250 | Small Dead hits/32 | Training hours | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C19 | .978045 | .957317 | .998638 | 10372 | 7147 | 6 | 4.0272 | 4.0474 |
| L19 | .662906 | .743902 | .970417 | 7810 | 6065 | 10 | 4.0973 | 4.0474 |
| C1 | .978398 | .963415 | .998258 | 10468 | 7298 | 9 | 4.0310 | 4.0474 |
| L1 | .630611 | .695122 | .973704 | 7411 | 5827 | 6 | 4.0974 | 4.0474 |

No candidate-cap saturation occurred. Under the fixed 130-epoch budget, lower adaptation strength produced fewer proposals and higher candidate precision while reducing fit and audit recall. The observations are compatible with insufficient adaptation at that budget; they do not prove optimizer convergence or identify a universal regularization mechanism. Do not extend this closed recipe to more epochs. Small-Dead changes disagree by seed (+4/−3), so they do not establish a stable improvement.

The descriptive tissue breakdown for internal Dead counts is: seed 19, Colon 23/56→10/56 and Lung 39/60→39/60; seed 1, Colon 17/56→12/56 and Lung 39/60→33/60. Other tissues have only 1–15 internal-Dead objects each. No tissue-specific selection or thresholds were used. The loss is not uniform across tissues, and this small, repeatedly reused image audit cannot identify a causal tissue mechanism.

Evidence includes 41 small files totaling 3,025,326 bytes returned to LM1; all 36 payloads matched the project preservation byte counts and SHA256. Each run has exactly 130 finite, nonnegative loss records. Cohort membership, evaluation ordering, configuration, UNI, and input identities were checked, and the original summary and admission were recomputed. Training source commit: `9a2533e1103ebd8db1ec79b2f8c2804a5588f2bb`. Source manifest: `630c58d842ee84a48f0b16bd92919412ebef68758a171ba6d102db59cbea943d`. Large checkpoints remain on Expanse; writer-side hashes are not independent rehashes. External Lustre archival remains outstanding.

This is point matching on repeatedly reused fold 1 development images, not mask PQ, test, independent validation, or official three-split performance. The audit contains 522 images, including 11 Dead-positive images and 145 internal-Dead objects; separation is by image, not proven patient/slide identity. The original 1000-draw tissue-image bootstrap is conditional on trained models and does not measure seed uncertainty or correct for sequential selection. Historical P3–P13 controls are not pooled with this round's fresh controls.

P15 was separately frozen as a training integer-translation hypothesis, not a low-LR continuation. C/J use the original encoder LR 1e-5, head LR 3e-4, full image coverage, model, and loss. J alone adds integer translations in [-16,16] with white fill before the original augmentations and uses the retained-object targets/count. The original decoder and admission remain unchanged. Prior recipes stay closed; validation and formal submission are reported separately.
