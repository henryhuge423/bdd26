# P13 fixed rare-positive image exposure: negative development result

All four fresh runs in array 55262695 completed 130 epochs with SLURM status COMPLETED/0:0. The experiment compared C and S with seeds 19 and 1, using the final checkpoints, original decoder, and original admission rule. The fixed 25% replacement recipe is closed; no additional sampling fraction, seed, epoch, Phase B, mask stage, or fold 2/fold 3 evaluation is authorized by this result.

| Seed | C fit Dead recall | S fit Dead recall | C audit Dead recall | S audit Dead recall | S−C | Original paired 95% CI | S precision | Gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| 19 | .957317 | .998476 | .489655 | .351724 | −.137931 | [−.212389,−.090395] | .684929 | false |
| 1 | .955793 | 1.000000 | .406897 | .358621 | −.048276 | [−.185185,+.051020] | .684297 | false |

S audit overall recall was .577633/.579837, compared with .591265/.588245 for C. S unique precision was .684929/.684297, compared with .690665/.690627 for C. Every run had a nondegenerate-image fraction of 1. Both seeds fail the unchanged admission rule, which requires Dead recall ≥.60, gain ≥.05, a positive lower confidence bound, precision ≥.75, and the relative guards. The bootstrap uses the original 1000 tissue-stratified image draws and is conditional on the trained models; it does not measure uncertainty over seed populations.

S replaced 533 of the 2134 image positions in each epoch by sampling from the 52 fit images containing any Dead objects. In epoch 0, S had 578/573 Dead-positive exposures across the two seeds, compared with 52 for C. The unique-image counts were 1608/1613 for S and 2134 for C. Duplicate occurrences of an image received the same augmentation within an epoch. Fit internal-Dead recall approached 1 for S while audit recall deteriorated. This is consistent with memorization or a changed exposure/coverage tradeoff, but does not isolate rarity, show that all balancing is harmful, or establish a generalization mechanism. Whole-image background/common-class exposure and image coverage also changed. No threshold was selected.

Evidence includes 45 transferred small files and 40 preservation payloads verified by bytes and SHA256. The complete 130-epoch finite histories, original cohort, evaluation membership, and frozen admission were checked or recomputed. Large checkpoints remain at source; writer-side preservation records are not independent rehashes. Source commit: `caee42bcebf81d103370c03562edc4ddd38739a4`. Manifest: `ecc5974d5340cf21694b76e3a10f0944f1014d54eadff36562b2de0719c54e9a`. Original files remain in scratch and project storage.

This is point detection on repeatedly reused fold 1 development data: 522 audit images, of which 11 contain Dead-positive labels covering 145 internal-Dead objects, concentrated in a few Lung/Colon images. These are not instance PQ, test results, independent validation, or official three-split confirmation. P3–P12 recipes remain closed. P14 is a separately frozen fixed encoder-learning-rate control, framed as an adaptation-strength investigation rather than a rescue or tuning menu for P13.
