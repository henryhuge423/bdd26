# P20–P30 stage review and P31 submission

P30 completed four 130-epoch runs. Middle-twelve encoder adaptation adds 7/15 interior Dead matches and 648/516 all-nucleus matches, removing 215/254 unmatched predictions. Both original decisions are false. P31 tests fixed group normalization in the point decoder; formal Expanse array **55792463** was submitted once after V100 acceptance **55792375** passed.

- [Detailed English report, 20 pages](reports/RESULTS_EN.pdf) and [Markdown](RESULTS_EN.md).
- [Detailed Chinese report, 19 pages](reports/RESULTS_ZH.pdf) and [Markdown](RESULTS_ZH.md).
- [Chinese reader, 7 pages](reports/READER_ZH.pdf) and [Markdown](READER_ZH.md).
- [P31 scientific protocol and submission](P31_SUBMISSION.md), [actual receipt](p31_submission_receipt.json), [hardware evidence](p31_smoke_evidence.json).
- [P30 verified analysis](p30_verified_analysis.json), [collection checks](collection_result.json), [stage comparisons](stage_pairs.csv).
- [P30 object records](fixed_objects.csv), [P22/P25 object records](early_fixed_objects.csv), [early image provenance](EARLY_FIGURE_PROVENANCE.json).

The reports cover all eleven comparisons and analyze large real-image panels from P22, P25, P27, P29 and P30. `p30_files/` retains 117 collected small artifacts, including 48 final PNGs. `p31_validation/` retains source hashes, control checks and four preliminary one-epoch training previews. `early_files/` preserves P22/P25 coordinates; `figures/` contains final and analysis overlays. Large model checkpoints remain at their preserved server locations; independent rehashing and external archival are pending.

## Rebuild the reports

From this directory, in the pinned project environment with existing Pandoc and XeLaTeX:

```bash
python build_latex_reports.py --root .
python qa_latex_reports.py --root .
```

The builder generates seven TikZ/PGFPlots statistical figures and compiles each TeX report twice with XeLaTeX. `reports/` retains the final PDFs, TeX sources, build Markdown, figure TeX and build hashes. The page checker renders every page, checks text/glyphs and layout warnings, and writes its evidence to a sibling QA directory. Final visual review is recorded in `REPORT_QA.json`.

To reconstruct the large early-round overlays from the preserved coordinates and shared native image pixels:

```bash
python ../../../scripts/plot_proposal_p30_stage_examples.py --root .
```

P30 analysis plots can be rebuilt from the original stage table in the preceding report directory:

```bash
python ../../../scripts/plot_proposal_p30.py --root . --prior-stage-pairs ../p29_p30_2026-11-15/stage_pairs.csv
```

These are repeated fold-1 center-detection development comparisons. Each paired contrast uses its own freshly trained control; the project records instance segmentation with the separate three-split evaluation protocol.

[Recorded P31 training progress](P31_RUNNING_STATUS_2026-11-16.json): C19 completed 53 epochs and G19 51 epochs at 2026-11-17 04:58 UTC; C1/G1 were waiting for the array concurrency limit.
