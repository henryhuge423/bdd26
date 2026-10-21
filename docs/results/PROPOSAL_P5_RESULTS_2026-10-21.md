# P5 native-grid refinement closeout — 2026-10-21

All four fresh130epoch runs (L/R × seed19/1) completed; SLURM array55128210 tasks0-3 are COMPLETED/0:0. Source training commit cebd200d22625f0ac86bbc29c6e9e11904a50c2b, source manifest8d696c09adc35fa35ff15ae167182d8b79efb9c8e6fbebf19dce7db59d1a427f. Last checkpoints, unchanged fold1 fit2134/audit522, original threshold.5/cap512/unique3pixel scoring. No fold2/fold3 predictions or PQ scores.

40smallfiles (2,977,018bytes) returned through the existing master; all transport hashes and36payload hashes match original preservation manifests. Checkpoint hashes remain in those manifests, checkpoints were not downloaded or rehashed in this review. All four saved cohorts match the frozen cohort digest and identical weight/input identities.

| Arm/seed | Audit recall | Unique precision | InternalDead recall | SmallDead hit/total | Fit recall | Fit precision | Hours | Saturated images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_L_s19 | 0.607020 | 0.687373 | 0.489655 | 7/32 | 0.983048 | 0.998386 | 4.468 | 0 |
| A_R_s19 | 0.600408 | 0.685397 | 0.489655 | 10/32 | 0.980851 | 0.998243 | 5.184 | 0 |
| A_L_s1 | 0.598041 | 0.697117 | 0.468966 | 7/32 | 0.983990 | 0.998248 | 4.526 | 0 |
| A_R_s1 | 0.596327 | 0.691238 | 0.462069 | 7/32 | 0.983970 | 0.998348 | 5.148 | 0 |

## Frozen decision

Seed19: R-L internalDead recall +0.000000; original paired tissue-stratified1000draw95%CI [-0.032680, +0.053846]; pass=false.

Seed1: R-L internalDead recall -0.006897; original paired tissue-stratified1000draw95%CI [-0.031579, +0.013514]; pass=false.

Original proposal_p3_utils.admission mapping F1=L/T1=R returns false. Both seeds fail absolute Dead recall>=.60, required improvement>=.05/positiveCI, and precision>=.75. Overall recall/relative precision floors, support11positiveimages/145objects and nondegeneracy1.0 do not rescue a conjunctive gate. Close this matched-capacity native-grid refinement recipe at this budget. No extra seeds/epochs, retuning, PhaseB or mask expansion.

## Interpretation and next hypothesis

Native refinement does not improve the held-out deficit at this budget; fit recall>.98 and precision>.998 still do not transfer. This does not establish a unique cause, rule out all high-resolution processing, or show that more parameters alone are the issue. No cap saturation was observed. P4 foregroundBCE and P3 PhaseA gates remain closed, along with earlier query/E19 recipes. The remaining question is supervision/generalization of actual inferred centers.

The independent P6 compares sigma1/radius3 against sigma2/radius6 center-target negative-suppression bandwidth using the original P3 architecture and a fresh matched control. Focal positives stay at exact GT centers, so widening is not additional positive-label supervision. Decode/matching/gates remain unchanged. A wider halo could increase duplicate peaks; precision remains a hard guard. This is a hypothesis, not an established diagnosis.

The11Dead-positive audit images/145objects have been reused for development. Intervals resample images within represented tissues, not independent patients, and do not correct sequential hypothesis selection. These are development point metrics, not official PQ, independent test evidence or three-split performance. Total focal losses across bandwidth settings will not be treated as directly comparable model-quality scores.

## Evidence

LM1 ignored ops/review_20261021/{p5_collected,p5_collection_manifest.json,p5_gate.json,p5_analysis.json}; Expanse project bdd26/p5_20261020_03/results retains last/final checkpoints and original preservation manifests. Training memory at last epoch: L4.05GiB, R4.18GiB; elapsed training+evaluation around4.5/5.2hours per arm. Lustre copies still require external archival backup.
