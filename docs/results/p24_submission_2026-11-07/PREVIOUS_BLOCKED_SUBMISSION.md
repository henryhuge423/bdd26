# P24: GPU validation passed; formal submission blocked by live quota timeout

Updated 7 November2026. V100 smoke **55571695** completed with exit0:0 in **858seconds**. Actual readiness is **PASS**. The four formal tasks were **not submitted**. The project-user quota subprocess timed out after600seconds; the orchestration exited before sbatch. At13:54:40UTC on7November, the formal driver and remote submit parent were absent, all remote submission/job/receipt markers were absent, and the user queue was empty. [Reconciled failure evidence](p24_validation/formal_admission_failure.json).

Full CPU1055passed/3skipped/19073warnings in535.96s; subsequent readback/source-identity and relevant tests71passed. On V100,61focused tests passed in254.34s. Both seeds passed exact initial P3/C/L identity, actual two-step P3/C gradient/parameter/AdamW/RNG equality, L learning beyond pure decay, original order and BF16 checks. C/L smoke passed134fit images,9updates including tail6,frozen tensors,optimizer membership,last/final checkpoint reload and bound source/script identity. No audit predictions were made during smoke.

1.5x safety estimates C/L **7.6991/7.0353hours** are below12hours; training peak is **4.0943GiB** in both arms, below24GiB. These are smoke measurements, not guaranteed completion times. Four one-epoch fit preview PNGs and coordinate records are retained under [p24_validation](p24_validation); they are not final performance examples or a basis for parameter selection.

Training source `4f467331de7a43e4855dd02c22345e2f4a59a9ee`;479files,manifest `5efef9d479aab158fe8f1a88c333dd449924576b1dd854864b52c916a315a75c`. New project/scratch `bdd26/p24_20261106_02`,smoke sibling `p24_smoke_20261106_02`. Only these directories use OST43 with150GiB target reserve; prior _01 source-copy failure remains preserved. Existing runtime `timm` cache on OST5 also delayed smoke imports before recovering. These observations do not establish the number of failed storage servers or whole-Lustre recovery. [Storage rationale](P24_STORAGE_RECOVERY.md).

Formal design remains C/L encoder1e-5/3e-5,original last12/head3e-4/effective16/physical4,seeds19/1,fresh130epochs,lastcheckpoint,original gate. Mapping C19/L19/C1/L1,array0-3%2,each1V100/4CPU/32GiB/12h,max48GPUh. LR changes decoupled decay too. Fresh strict quota,capacity,fsync/readback and account reservation are required immediately before sbatch; no failed storage guard is bypassed. No new LM1/LM2 training was launched.

[Smoke evidence](p24_validation/smoke_evidence.json),[strict readiness](p24_validation/ready.json),[control witness](p24_validation/control.json). LM1 ignored records: `ops/p24_development_20261106_r2`. Reusedfold1 point development,notPQ/test; no extra menu,PhaseB,masks or fold2/3.

## Continuation after storage recovery

Keep the passed smoke and sealed scientific source. Preserve the local exclusive `formal_submission.started` and failure logs; never blindly rerun the once-only script or remove its marker. A future controlled attempt must first reconcile all old markers, processes and receipts, then obtain fresh accurate quota/capacity/fsync/readback and allocation reservation. Use a new auditable submission-attempt record after confirming no previous job exists; do not retrain the already passed smoke or change science to bypass a storage failure. The old blocked lfs process may remain in a kernel wait after its parent exits; it cannot submit a job, and must not be targeted using historical PID alone.

LM1 and LM2 were rechecked at13:44UTC: data7 had about91GiB free and data6 about2TiB, with existing GPU allocations preserved. No new local training. Windows execution SSH was recovered once under the user request; LM2 and the existing Expanse master were reused.
