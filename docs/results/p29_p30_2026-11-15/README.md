# P29 results and P30 preparation — November 15, 2026

P29's four models completed 130 epochs and development evaluation. The first-six plus last-six intervention adds 16/13 interior Dead matches and 283/165 total matches, while removing 65/64 unmatched predictions. Both original seed-level decisions are false. Experimental seed 1 reached the scheduler time limit during preservation after all model evaluation and final images completed; its original preservation manifest remains unavailable.

- [English PDF](reports/RESULTS_EN.pdf), [中文 PDF](reports/RESULTS_ZH.pdf), [中文导读 PDF](reports/READER_ZH.pdf)
- [LaTeX sources, build instructions and page validation](reports/README.md)
- [Detailed English stage analysis](RESULTS_EN.md)
- [详细中文阶段分析](RESULTS_ZH.md)
- [中文导读](READER_ZH.md)
- [P30 protocol, acceptance and actual submission status](P30_SUBMISSION.md)
- [Verified numerical analysis](p29_verified_analysis.json)
- [Collection and recovery evidence](collection_result.json)
- [Fixed-object coordinates and changes](fixed_objects.csv)
- [Object summary and image sensitivity](object_summary.json)
- [P20–P29 paired comparison table](stage_pairs.csv)

The report includes 48 epoch-130 model PNGs and 13 large analysis PNGs. Original images, GT centers, predictions and one-to-one assignments are in `p29_files/`. Figures and all results use the frozen fold-1 development cohort. The 505-source-file verification comes from the earlier verified seed-19 collection; 92 collected small payloads have SHA256 verification. Experimental seed-1 data were recovered directly from scratch after application completion. Six checkpoint sizes match original preservation records; two experimental seed-1 checkpoint sizes were recorded directly. Independent checkpoint rehashing and external archival are pending.

Reproduce numerical analysis and figures from the repository root:

```bash
PYTHONPATH=src python scripts/analyze_proposal_p29.py --root docs/results/p29_p30_2026-11-15 --out /tmp/p29_replay_new.json
python scripts/plot_proposal_p29.py --root docs/results/p29_p30_2026-11-15 --prior-stage-pairs docs/results/p28_p29_2026-11-14/stage_pairs.csv
```

Use a new output JSON path for the create-only numerical replay. The plot command regenerates the figure PNGs, fixed-object table and descriptive sensitivity summary from the collected coordinates.
