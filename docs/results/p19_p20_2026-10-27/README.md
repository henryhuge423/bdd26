# P19 closeout / P20 encoder-scale study — 27 October 2026

- [Detailed English analysis](RESEARCH_REVIEW_EN.md)
- [详细中文分析](RESEARCH_REVIEW_ZH.md)
- [中文导读](RESEARCH_READER_ZH.md)
- [Verified scientific numbers](P19_NUMBERS.json)
- [P20 validation and submission](P20_SUBMISSION.md)
- [Frozen P20 specification](../../superpowers/specs/2026-10-27-proposal-p20-design.md)

Reproduce using pinned project Python and `scripts/analyze_proposal_p19.py --collection P19_COLLECTION.json --out NEW_ANALYSIS.json`. The 42-file original collection is retained in the ignored handoff and local delivery, with hashes listed in the scientific numbers. Full accounting, machine resources, operational scripts and logs stay ignored. Stage6 PDFs retain their original snapshots. These are reused fold1 development point results, not PQ/test/patient-independent evidence.

A [standalone NumPy reproduction](REPRODUCE_P19.py) independently checks all42small-file hashes, preservation, aggregates and original1000draw bootstrap without torch, inference or the original admission import. Place it beside the delivered P19_COLLECTION.json and P19_NUMBERS.json and run `python REPRODUCE_P19.py`. It passed in the pinned NumPy1.23.5 environment.
