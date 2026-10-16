# E6 set-mask development: implementation and resource admission

**Outcome: implementation verified; resource admission failed; 0/4 formal runs launched.** No validation predictions, quality endpoints, or end-to-end inference cost results exist for this round.

## Frozen scope and source

Approved B/Q × seeds19/1; UNI/native256; fold1 training → fold2 development validation only;130 epochs, unfreeze25, batch16, workers8. No fold3 access, dependency upgrade, publishing, extra seeds, or recipe search.

Training source: `research/set-mask-e6`, commit `365b07a391b06acb387edda1e74e702e0c00b01a`. Isolated LM2 snapshot: `/data6/jinxinhao/bdd26/ops/staging/setmask_e6_20261015`. Archive SHA256: `f7f240a4af5e946516073096c11d67c0ffba7f8a57aabbecaaee9dcd35cbe140`;217 scoped source/config files,40 pinned vendor Python files. Existing worker data/weights/vendor paths were linked after identity checks. Main checkout changes belonging to other work were preserved.

## Verification

- LM1 complete frozen CPU suite:471 passed,3 skipped,19073 warnings,362.49s. E6 focused:54 passed. LM2 E6 focused:54 passed,19 warnings,26.77s.
- Fresh whole-change review:3 Important findings fixed through6 RED→GREEN tests: official GT/evaluator provenance, checkpoint/log interruption reconciliation, and validated idle-GPU latency comparability. No deferred minors.
- Operational child-lifecycle checks:4 passed (success,nonzero exit,timeout,ownership isolation). Queue remains unlaunched.
- Both actual runtimes match torch2.5.1+cu124,CUDA12.4,torch Git build `a8d6afb511a69687bbb2b7e88a3cf67917e1697e`. Other checked pinned versions match. Only distribution metadata differs (`2.5.1` vs `2.5.1+cu124`); environment admission receipt records this explicitly.
- Fold1/2 input hashes match, including official fold2 `gt_channels.npz` SHA256 `abcf44369d8e9be459d43070903b9b33af1c3b210c2034e806380562a9dfc7f3`.

## GPU smoke measurements

One bounded smoke ran on an initially empty A100-SXM4-80GB,physical GPU3. GPU6 was refused before child creation after foreign Ray jobs appeared. No existing job was stopped. GPU2 stayed excluded. Both models consumed the same16-image augmented batch; digest `d0b5e9d5d0aa1ab0ee1b2ec6a73bf641352eefe26e44603573f1d21f6a0f178c`.

One warmup and one measured training step were run for each frozen/unfrozen state, followed by one no-grad validation batch and real durable full-state/final serialization. Losses and all present gradients were finite. Gradient-check overhead was measured separately from step throughput.

| Measurement | B | Q |
|---|---:|---:|
| Parameters |362,934,749|337,441,241|
| Frozen measured training step |0.202s|0.379s|
| Unfrozen measured training step |0.252s|0.442s|
| Maximum reserved GPU memory |15.16GiB|13.18GiB|
| Validation batch |0.067s|0.283s|
| Complete last checkpoint |4.356GB|4.050GB|
| Durable last save |84.73s|80.80s|
| Final model file |1.452GB|1.350GB|
|130-epoch train-step extrapolation |1.45h|2.58h|
| Validation extrapolation |0.08h|0.32h|
|130 complete saves extrapolation |3.06h|2.92h|
| Per-run total extrapolation |**4.59h**|**5.82h**|

Smoke body256.61s,outer process264.84s; exit2 explicitly means admission failed,not a crash or timeout. Initial GPU free78.74GiB; measured peak+4GiB fits. Estimated maximum retained outputs plus atomic temporary checkpoint and prediction allowance28.93GiB fits32GiB. The **time gate fails**: each arm exceeds4h; four-run extrapolation20.82h exceeds16h.

These are extrapolations from one paired batch,not full-run timings. They omit some loader/epoch transitions,final serialization and downstream evaluation/benchmark costs; they cannot establish that a marginally passing run would fit. Training-step ratios are not inference-cost measurements and do not satisfy the≤2× end-to-end inference gate.

## Interpretation and next experiment design

The immediate measured bottleneck is durable checkpoint I/O: approximately50–51MB/s on `/data6`,despite~2TiB free. The filesystem was97% used. `/tmp` shares the root filesystem,which had only773MiB free; it cannot host the approved output budget. `/dev/shm` had154GiB free but is volatile and is not adopted as the sole checkpoint store.

The original fold1/2 capacity audit found maxima276/289 instances across2656/2523 images,zero images over512,and no invalid original targets. Current geometric augmentations do not introduce new IDs,so original target capacity is not the observed blocker. This does not prove boundary quality or geometric recall.

Both architectures are numerically executable at batch16 within measured memory. No evidence yet supports or rejects Q's segmentation benefit. Q also has~7% fewer parameters; any eventual benefit must remain an experiment-package conclusion.

Preferred follow-up: resolve durable checkpoint throughput while retaining the approved per-epoch recoverable checkpoint policy. Q would need complete-save time below~30.4s just to meet4h under this limited extrapolation (about133MB/s for the measured4.05GB payload,before additional margin). This is a diagnostic target,not a measured faster storage result.

An alternative requiring an explicit policy amendment is complete checkpoints every5 epochs,including epoch130,with every-epoch sample digests/logs and tested rollback/replay from the last committed epoch. At current measured times,26 saves would extrapolate to B2.14h/Q3.48h and four-run11.25h,excluding other overhead. This alternative was **not implemented or run**. It needs recovery tests for logs ahead of a checkpoint and a new resource admission; the current fail-stop rule is preserved.

## Handoff

LM1 evidence: `ops/setmask_e6_20261015/{progress.md,full_review_suite.log,smoke.json,smoke_receipt.json,environment_admission.json,capacity_audit.json,archive.json}`.

LM2 evidence: `ops/setmask_e6_20261015/{smoke.log,smoke_receipt.json,smoke/smoke.json,capacity_audit.json}`;source/environment/test receipts in snapshot `e6_ops/`. Smoke launcher PID2870810 and child2870812 have exited. GPU3 returned to0MiB used in the post-run inventory. Formal output root `runs/setmask_e6_20261015` does not exist; no queue PID was created. Do not replay launch requests or infer authorization for a retry from this report.

Statistical and quality gates remain unexecuted. Fold2 reuse and two fixed seeds preclude held-out test claims and seed-population inference. Scientific split/seed/selection rules and historical metrics/models remain unchanged.
