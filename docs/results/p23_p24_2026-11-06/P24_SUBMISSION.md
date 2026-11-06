# P24 validation and submission — work in progress

Scientific design: [frozen specification](../../superpowers/specs/2026-11-06-proposal-p24-design.md). Four fresh C19/L19/C1/L1 runs, original last12, encoder1e-5/3e-5,130epochs;0-3%2,1V100/4CPU/32GiB/12h each,48GPUh formal maximum. Separate30minute fit-only smoke. No training result is claimed here.

Portable source sealed for deployment from d4f1593 (design8b234af, implementation5588bbb, merge of teammate presentation1eab38e). Full CPU1055passed/3skipped/19073warnings in535.96seconds; subsequent submission-readback/source-identity and relevant experiment tests71passed in14.25seconds. Baseline source and three new P24 files were checked against the base Git tree and the pre-suite review bundle; additional readback tests passed separately. Tests exercise fsync errors, corrupt readback, stale source and changed inventories. Independent review found no runner/verifier blockers; its two deployment findings (fresh readback, tested-source binding) were implemented and covered by the failure-then-pass checks.

At17:37UTC all three strict quota scopes passed. Subsequent scratch/project64-KiB fsync/readback passed. The once-only deployment was launched and is sealing a new source snapshot; source copying showed Lustre delay. **No smoke or formal job receipt was available at this publication checkpoint. Do not replay deployment.** The existing deployment process and exclusive markers must be inspected first.

Planned paths: project `/expanse/lustre/projects/jhu159/xjin10/bdd26/p24_20261106_01`; scratch `/expanse/lustre/scratch/xjin10/temp_project/bdd26/p24_20261106_01`; smoke sibling `p24_smoke_20261106_01`. LM1 ignored orchestration/evidence: `ops/p24_development_20261106`. No new LM1/LM2 GPU computation was launched.

Before formal submission require COMPLETED/0:0 smoke, actual C/P3 two-step state/gradient/AdamW/RNG equality, L finite gradients and changes beyond decay, frozen hashes, BF16, original9-step/tail6 fitting smoke, strict last/final reload, source/identity-bound readiness, <=24GiB training peak,1.5x safety estimate <12h for each arm, fresh quotas/readback and live account reservation. New input/GT rules are not selected from smoke audit scores. No audit predictions during smoke.

No automatic retry or speculative GPU submission is authorized by a partial source copy. Infrastructure failure records remain preserved. Later receipt updates must distinguish submission from completion.
