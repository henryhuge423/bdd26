# P21 closeout / P22 development — 1 November 2026

- [Detailed English analysis](RESEARCH_REVIEW_EN.md)
- [详细中文分析](RESEARCH_REVIEW_ZH.md)
- [中文导读](RESEARCH_READER_ZH.md)
- [Verified P21 numerical snapshot](P21_NUMBERS.json)
- [Descriptive per-image, leave-one-image and cost decomposition](P21_DIAGNOSTICS.json)
- [Frozen P22 design](../../superpowers/specs/2026-11-01-proposal-p22-design.md)

P21 failed both original gates (Dead A−C +.006897/−.034483; absolute Dead .448276, precision .699284/.696585); the fixed five-class auxiliary recipe is closed. P22 (full-depth UNI adaptation, arm M versus the original last-12 control C) is an adaptive exploratory follow-up chosen after reading P21; its validation and submission are recorded here when verified. These are reused fold1 point-development results, not PQ/test. Deterministic 8 audit + 4 fit point examples per run are preserved with the results on Expanse; a local copy accompanies the handoff.

P21 operations: original array 55416129 (C19 failed in a Lustre quota query), replacement 55440148 (A19 completed; C19/C1/A1 failed before training), CPU gate 55450367 and recovery array 55450368 (C19/C1/A1 completed). See [startup recovery](../p20_p21_2026-10-30/P21_RECOVERY_2026-10-30.md) and [storage investigation](../p20_p21_2026-10-30/P21_STORAGE_RECOVERY_2026-10-31.md).
