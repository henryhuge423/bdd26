# P6 target-bandwidth control: formal submission, 2026-10-21

The current user requested a three-machine review, result analysis/documentation and advancement through the next substantial Expanse submission. P5 four130epoch runs completed and failed original admission; see [P5 closeout](PROPOSAL_P5_RESULTS_2026-10-21.md). Independent P6 compares center-target negative-suppression bandwidth: original sigma1/radius3 C versus sigma2/radius6 W. Original P3 model and all other protocol details are unchanged. Gaussian halos are not positive centers. Closed recipes and original PhaseB remain closed.

## Submitted work

Array **55148533**, submitted once at2026-10-21 **09:56:52 EDT** (2026-10-21T13:56:52.649509Z). Tasks0=C/seed19,1=W/seed19,2=C/seed1,3=W/seed1. Four fresh130epoch runs, original fold1 exact fit2134/audit522, final checkpoint only. InitialPENDING, not completion or a positive scientific result.

Resources: gpu-shared/gpu-shared-normal/jhu159, array0-3%2, each1V100/4CPU/32GiB/12hours, Lustre/no-requeue; maximum48GPU-hours. Existing other-project native13 jobs preserved. Reused migrated data/UNI/pinned runtime; no dependency upgrades, large transfers, source deletion or new Expanse authentication.

## Actual validation

LM1 baseline40passed10.33s; new feature11missing-feature failures3.79s; final P3–P6 focused51passed11.67s, syntax/CLI and sbatch/once-script bash syntax passed. Independent design and whole-change code reviews found no blocking issues. Metrics/HoVer-Net untouched; no full-suite green claim, historical five dense_votes failures disclosed.

Real V100 smoke **55148203 COMPLETED/0:0**,10m26s, exp-7-59, empty stderr. Node focused51passed96.54s. Both seeds had exact initial C/W state equality; seed19 C and original P3 were equal after two real effective-batch optimizer steps for all trainable gradients and adapted parameters, using identical smoke membership/order. Both arms had finite positive encoder/output gradients and parameter changes, active CUDA BF16 internal encoder/output operations and original FP32 terminal bilinear, exact last/final reload predictions on the same GPU. No P6 audit scores read.

|Arm|256fit-image training seconds|130epoch estimate incl writes/eval|With1.5safety|Training peakGiB|
|---|---:|---:|---:|---:|
|C|13.4709|4.633h|6.950h|4.059|
|W|13.4566|4.624h|6.936h|4.059|

Readiness extrapolations are not actual completion times. Original12hour time gate passed without changing scientific settings. Source/script/data/weight/cohort checked in smoke, before submission and again at formal launch. Final quality gates pending all four complete runs.

## Frozen decision

Call unchanged proposal_p3_utils.admission mapping F1=C/T1=W. BOTH seeds require W internalDead recall>=.60, W-C>=.05 with original tissue-stratified1000draw95%CI lower>0; precision>=.75 and>=C-.02; recall>=C-.02; nondeg>=.95; support>=10images/100objects. Reused11positiveimages/145objects are development data, not independent validation/PQ/test/three-split performance. No threshold search, extra epochs/seeds, automatic PhaseB/masks or fold2/fold3 model evaluation. Failure/inconclusive closes this bandwidth recipe. Success requires a separate next-stage design.

## Provenance and limitations

Branch research/proposal-p6; training commit `c1b3bc9b7d1c942b2cd1d0b06c19eaa4fc47054e`. Docs-only follow-up does not alter deployed source. Source manifest `f41c9868070894a9d93dc5028da8d74d2abbdd059992e77d97aa6463558c103e`. [Frozen design](../superpowers/specs/2026-10-21-proposal-p6-design.md), [plan](../superpowers/plans/2026-10-21-proposal-p6.md). Operational scripts/receipt/guard/readiness/evidence/ledger/handoff in LM1 ignored ops/p6_development_20261021/. Project/scratch paths are in receipt. Results preserve last/final and SHA256manifest in project; Lustre still requires external archive.

Deferred minor: preliminary smoke_measurements.json has PASS label before the time assertion. It is diagnostic only; authoritative smoke.json/ready.json exist only after that assertion, and formal requires COMPLETED/0:0 plus readiness. No gate bypass in this successful smoke.
