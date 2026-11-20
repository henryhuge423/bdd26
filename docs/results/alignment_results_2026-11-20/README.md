# Alignment and low-score results — November 20, 2026

All eight P36 scientific computations completed. A preservation-only task recovered one failed save; all eight collections passed. The 242 collected preserved files, paired decisions and P36 analysis were independently checked. D37 adds an exclusive object-level diagnostic using the existing point records.

- [Detailed English analysis](REVIEW_EN.md), [detailed Chinese analysis](REVIEW_ZH.md), [Chinese reader](GUIDE_ZH.md).
- [Stage-nine LaTeX reports and sources](../../report/stage_report_9/README.md): English, Chinese and illustrated Chinese reader.
- [P36 results](P36_RESULTS.json), [D37 results and input hashes](D37_RESULTS.json), [P36 validation and collection receipts](P36_VALIDATION.json).
- [Fixed design for D37](D37_FROZEN_DESIGN.md), [case assignments and coordinates](CASES.csv), [measured case scores and positions](CASE_RESPONSES.csv).
- Prior rounds: [P34/P35 and frozen P36 design](../alignment_p34_p36_2026-11-19/README.md), [D32/D33](../center_diagnostics_recovery_2026-11-18/README.md).

The all-layer detector primarily benefits from changing alignment. The translation-trained detector has many low-score matches: at its training-selected threshold 0.2, it matches 101/99 of 145 interior Dead nuclei versus 44/44 at 0.5, with precision 66.04%/66.50%. The original two-seed quality criteria remain unmet. Statistical charts, eight fixed D37 images and full coordinates are saved; source records contain both fills and all sixteen alignments.

Scope: fold-1 training/development division, last checkpoints, frozen detectors, CPU single precision and one-to-one center matching within three pixels. No detector was retrained in this task. The full object rows remain with the preserved execution outputs; public case tables cover the four fixed images for two seed-19 detectors. The D37 JSON records hashes of every input used.

Portable replay:

```bash
PYTHONPATH=src:scripts python scripts/analyze_alignment_misses_d37.py --root <collected_P36_results> --out <new_output> --images <compact_fold1_images.npy>
PYTHONPATH=src:scripts python docs/results/alignment_results_2026-11-20/analyze_p36_results.py <collected_P36_results> <new_output>
PYTHONPATH=src:scripts python docs/results/alignment_results_2026-11-20/trace_cases.py <collected_P36_results> <new_case_responses.csv>
```

Use the pinned project environment. Original and failed outputs are preserved; output directories must be fresh. D37 checks original, selected-threshold and consensus counts image by image and preserves predictions in images with no annotated nuclei.
