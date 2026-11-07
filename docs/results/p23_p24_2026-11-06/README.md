# P23 / D24 / P24 — 2026-11-06

P24 V100 validation55571695 passed; formal submission awaits fresh storage admission. See [current status](P24_SUBMISSION.md) and [storage recovery](P24_STORAGE_RECOVERY.md).

- [English analysis](RESULTS_EN.md)
- [详细中文分析](RESULTS_ZH.md)
- [中文导读](READER_ZH.md)
- [P24 validation and submission status](P24_SUBMISSION.md)
- [Recomputed raw gate, fit/audit metrics, costs and hashes](raw_verification.json)
- [Recovered P23 raw results and eight final examples](p23_recovered)
- [D24 six-model compact outputs](evidence/d24_recovered_six.jsonl)
- [P23 metrics](figures/p23_metrics.png), [D24 miss reasons](figures/d24_miss_reasons.png), [paired object response](figures/d24_paired_response.png)

The early compact-evidence analysis is retained in `figures/analysis.json`; its lack-of-raw-retrieval limitation was superseded by `raw_verification.json` after storage recovered. D24 P23 seed1 object-level outputs remain unavailable because the diagnostic job failed before creating them. P23 final seed1 evaluation is present and independently recomputed. No results are imputed.

All metrics are repeated fold1 point-development evidence, not PQ or independent test performance. Original scientific snapshots and failed infrastructure attempts are retained. No checkpoint is included here.

To regenerate the initial descriptive plots, run the pinned project Python with `scripts/analyze_p23_d24.py docs/results/p23_p24_2026-11-06/evidence OUTPUT_DIRECTORY`. Full recovery verification invokes unchanged `proposal_p3_utils.admission` on the retained C/E `evaluation.json` files; the command record is in LM1 ignored `ops/logs/remote_session/review20261106_rawverify47`.
