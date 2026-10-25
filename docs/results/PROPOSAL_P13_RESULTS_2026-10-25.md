# P13 fixed rare-positive image exposure: negative development result

All four fresh runs in array55262695 completed130epochs, SLURM COMPLETED/0:0. Original C/S×seeds19/1, final checkpoints, original decoder and admission. Close the fixed25% replacement recipe; no extra sampling fraction, seeds, epochs, PhaseB, masks or fold2/fold3 evaluation.

| Seed | C fit Dead recall | S fit Dead recall | C audit Dead recall | S audit Dead recall | S−C | Original paired95%CI | S precision | Gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
|19|.957317|.998476|.489655|.351724|−.137931|[−.212389,−.090395]|.684929|false|
|1|.955793|1.000000|.406897|.358621|−.048276|[−.185185,+.051020]|.684297|false|

S audit overall recall .577633/.579837 versus C .591265/.588245; S unique precision .684929/.684297 versus C .690665/.690627. Nondegenerate image fraction1 in every run. Original Dead>=.60, delta>=.05, positive lower CI, precision>=.75 and relative guards unchanged; both seeds fail. Bootstrap is the original1000 tissue-stratified image draws, conditional on trained models. It is not uncertainty over seed populations.

S repeated the52 fit-any-Dead images at533 replacement positions per2134-exposure epoch. Epoch0 positive exposures578/573 versus C52; unique images1608/1613 versus C2134. Duplicate occurrences receive the same augmentation within an epoch. S fit internal-Dead recall approaches1 while audit deteriorates; this is consistent with memorization or altered exposure/coverage tradeoffs. It does not isolate rarity, prove that all balancing is harmful, or establish a generalization mechanism. Whole-image background/common-class exposure and image coverage also changed. No new threshold was selected.

Evidence:45 transferred small files,40 preservation payloads verified by bytes+SHA256; complete130-epoch finite histories, exact original cohort and evaluation membership, original frozen admission recomputed. Large checkpoints remain at source; their writer preservation records do not constitute independent rehashing. Source caee42bcebf81d103370c03562edc4ddd38739a4, manifest ecc5974d5340cf21694b76e3a10f0944f1014d54eadff36562b2de0719c54e9a. Scratch and project retain originals.

This is repeatedly reused fold1 development point detection, only522 audit images with11 Dead-positive images/145 internal Dead objects, predominantly a few Lung/Colon images. These are not instance PQ, test results, independent validation or official three-split confirmation. Earlier P3–P12 tested recipes remain closed. P14 is a separately frozen fixed encoder-learning-rate control, justified as adaptation-strength investigation, not a rescue or tuning menu for P13.
