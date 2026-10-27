# P18 closeout / P19 batch experiment — 27 October 2026

- [Detailed English results](P18_RESULTS_EN.md)
- [详细中文分析](RESEARCH_REVIEW_ZH.md)
- [中文导读](RESEARCH_READER_ZH.md)
- [P19 measured validation and submission](P19_SUBMISSION.md)
- [P18 independently reproduced numbers](p18_independent_analysis.json), [original verified record](p18_verified_final.json), [P19 validation](P19_VALIDATION.json)

P18 failed both original gates; P19 array55326699 was submitted once after V100 validation. Submission is not completion. This closeout supplements the existing Stage6 report without replacing its PDFs or its historical cutoff.

Reproduce with the pinned project environment (NumPy1.23.5):

```bash
PYTHONPATH=src:scripts ~/.conda/envs/nuclei/bin/python scripts/analyze_proposal_p18.py --root ops/p19_development_20261027 --out ops/p19_development_20261027_r2/new_analysis.json
```

The ignored artifact root must contain p18_collection_final.json and p18_verified_final.json; use a new output filename. The script independently hashes36smallpayloads, validates preservation/config/epochs/cohort identities, aggregates rows and reproduces the original1000-draw stratified image bootstrap without importing admission. Upstream source verification413files is recorded; large checkpoints were not independently transferred/rehashed. The raw artifact path is operational and not a Git dataset publication.
