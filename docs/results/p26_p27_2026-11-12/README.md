# PanNuke point development review and next experiment

- [English analysis](RESULTS_EN.md)
- [中文详报](RESULTS_ZH.md)
- [中文导读](READER_ZH.md)
- [P26 numerical results](p26_analysis.json), [object-level coordinates](fixed_objects.csv), and [artifact hashes](artifact_hashes.json)
- [P27 protocol](../../superpowers/specs/2026-11-12-proposal-p27-design.md) and [submission receipt](p27_submission.json)

All four P26 runs completed 130 epochs. The weight anchor recovered two additional interior Dead nuclei per seed and reduced overall recall and precision in both. The source and small artifacts were checked against their preservation hashes.

Expanse accepted P27 array **55702167** at 2026-11-12T13:52:41.996454Z. The four tasks compare updates to the first versus last 12 UNI blocks in seeds 19 and 1, each requesting one V100 for up to 12 hours with concurrency two. Node smoke **55702061** completed successfully in 444 seconds. Focused regression checks passed in LM1 and on the V100. The [training previews](smoke_training_previews/) show one epoch of smoke training. Formal results will follow training and development evaluation.
