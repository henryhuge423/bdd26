# HV threshold controls implementation and experiment plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Isolate working resolution from the minimum instance area receiving nonzero HV targets, without changing historical defaults or reading new test results.

**Architecture:** A local, attributed adaptation of the pinned official HV generator accepts a positive integer working-pixel area cutoff. Thread `hv_min_size` through CellViT configuration and both training/validation datasets; add a train-only CLI boundary and immutable scientific resume configuration. Launch fresh split1 controls under the same original training recipe, then infer/evaluate validation fold2 with frozen scale-aware decoding.

**Tech Stack:** Python 3.10; torch 2.5.1+cu124; numpy 1.23.5; existing pinned scipy/skimage/albumentations; pytest.

**Spec:** `docs/RESEARCH_NEXT_2026-10-06.md`, §2.6 and E2a. User approved implementation, experiment launch and Git commit on 2026-10-06. This file freezes the first experimental round before training.

## Global Constraints

- No changes to official PanNuke split mapping, metric implementation or released-weight policy.
- Own models use final checkpoints. First-round development uses split1 train fold1 / validation fold2; fold3 is not inferred or evaluated.
- Preserve default `hv_min_size=30` in working pixels. It is not auto-scaled by `upscale`.
- NP/TP labels remain unchanged; instances excluded from HV target generation receive zero HV, not an ignored loss.
- No edits to ignored `third_party` sources; preserve the upstream MIT notice with locally adapted code.
- Keep dependency pins unchanged. Run `python -m pytest -q` after this HoVer-Net plumbing change.
- New reusable code is in `src/` or `scripts/`; machine launchers and resource records remain ignored under `ops/`.
- Work on `research/hv-target-threshold-controls`, not master. Commit portable code/tests/docs only. Initial implementation did not authorize a push; on2026-10-06 the user subsequently authorized merging into master and pushing to GitHub **after experiment completion and analysis**.

## Review Focus

- Tiny, border-touching, disconnected and sparse-ID instances: threshold selection must preserve official remapping, center rounding and normalization; default output bit-identical.
- Cutoff inclusion: area equal to the positive integer threshold is retained; NP/TP are never filtered.
- Train vs validation: the same working-pixel cutoff reaches both datasets, without accidental multiplication by upscale².
- Resume: changed scientific settings must be refused before overwriting config or loading a checkpoint; legacy configs missing `hv_min_size` mean30.
- CLI: `--train-only` must not construct a test dataset, test predictor or inference model; invalid cutoffs fail before writes.

## Frozen experimental design

Four arms, each with seeds **19 and1**, split1 only; fresh directories rather than mixing historical baselines:

| arm | upscale | hv_min_size (working px²) | native integer area support |
|---|---:|---:|---:|
| x1_hv30 | 1 | 30 | >=30 |
| x1_hv8 | 1 | 8 | >=8 |
| x2_hv30 | 2 | 30 | >=8 after exact nearest x2 |
| x2_hv120 | 2 | 120 | >=30 after exact nearest x2 |

Shared: 130epochs, final checkpoint, batch16, unfreeze epoch25, lr3e−4, ExponentialLR gamma.85, sampler gamma.85, original augmentations, no added NP-WCE/copy-paste/synthesis/TTA. Fresh runs use namespace `runs/hv_threshold_controls_20261006/<arm>_seed<seed>`.

Validation: use `scripts/predict_cellvit.py --fold 2 --no-eval`, decode-u1 for x1 and decode-u2 for x2, marker-u1, no M/S or EF additions. Evaluate in a separate process with `scripts/eval_pannuke.py`; preserve raw probabilities/maps when feasible. No threshold search. Report mPQ/bPQ/Dead/strict counterparts and internal/border diagnostics separately. No claim of an independent new test benchmark or statistical proof from two seeds.

Contrasts: within x1, hv8−hv30; within x2, hv30−hv120; scale comparison at matched native support (x2_hv120−x1_hv30, x2_hv30−x1_hv8). Two-seed development results decide whether more seeds or a schedule control is informative; do not pick the best seed. All arms are reported, including losses.

## Verification and execution record

Implementation completed on2026-10-06; source commit **`c6e7e92`**. The35 new tests were first observed failing before their corresponding production changes, then passed. Target+legacy HoVer tests:26 passed. Full CPU suite:189 passed,3 CUDA-dependent skipped (62 existing dependency warnings). Independent review found no blocking defects; its stale single-variable docstring finding was corrected.

Worker verification initially stopped during collection because optional MONAI was absent. After providing pinned `monai==1.3.2` in an isolated test-only directory (no shared-environment upgrade), the full CUDA-enabled suite reported **191 passed,1 skipped**. The remaining skip is the report PDF/font prerequisite; that test passed in the CPU environment. Torch2.5.1+cu124 and numpy1.23.5 remained unchanged. This is not a claim that a single suite invocation had192 passing tests.

**Launch verified on2026-10-06:** both queues use an immutable snapshot of `c6e7e92`. The first active runs are `x1_hv30_seed19` and `x2_hv120_seed1`; at the startup check, they had completed3 and1 epochs respectively, with finite losses and saved configs matching the frozen recipe. Six other runs remain queued. All training and subsequent inference/evaluation use fold1/fold2 only; no new fold3 predictions or performance claims. Machine paths, queue PIDs and operational commands are retained only in ignored local records.

The checklist below records implementation/launch completion, **not completion of the eight experiments**. Scientific recipe and contrasts were fixed before launch.

### Completion analysis and publication (authorized, pending experiments)

On2026-10-06 the user requested follow-through to final analysis, documentation, local master integration and GitHub publication. The analysis program is `scripts/hv_control_numbers.py`; its fixed analysis settings are2000 tissue-stratified paired-image bootstrap draws, RNG seed20261006, the same image resampling across both trained seeds and all four predefined contrasts. These are conditional image-sampling intervals, not independent-patient or training-seed significance guarantees. Strict and pooled endpoints have no persisted per-image arrays and remain evaluator-summary point estimates.

The collector refuses missing runs/checkpoints/evaluations, changed source/configuration, incomplete epoch logs and any fold3 artifact. It verifies image/class/area/tissue row alignment before attaching native-scale boundary flags, rather than truncating mismatched GT records. It records input hashes, checkpoint file sizes and duplicate epoch records. Report all eight runs and both within-scale and matched-support scale contrasts. GT views cover internal/border Dead and native area<30,30–59,>=60; these are mechanism diagnostics, not a new filter menu.

**Inference-provenance audit, fixed before reading results:** review identified that training provenance and internally consistent summaries alone do not certify checkpoint/TTA/decode settings. After training, `scripts/hv_inference_audit.py` explicitly reruns the unchanged prediction/evaluation CLIs on fold2, final.pth, no TTA, decode-u=upscale, marker-u1, batch8. It hashes the checkpoint before/after, records executed commands and source/prediction/evaluation hashes, and writes `audit_val_fold2` only after success. Original queue predictions/evaluations remain untouched. The collector checks these recorded settings and hashes; all arms use the audited rerun as the primary analysis, and audit−original differences are reported rather than selecting the higher score. This adds validation inference, not another training recipe or test-fold evaluation.

- [ ] All eight final checkpoints and validation evaluations complete and verified.
- [ ] Produce audited result tables and mechanism interpretation, including negative findings and limitations.
- [ ] Update research documents, verify tests, commit analysis/results.
- [ ] Merge the feature branch into local master and push the reviewed project content while preserving existing remote presentation content; no force-push or credential upload.

## Task 1: Parameterized HV generation, default parity (complete)

**Files:** create `src/nucseg/hovernet/targets.py`, `tests/test_hv_threshold.py`; modify `src/nucseg/hovernet/data.py`, `src/nucseg/hovernet/official.py`.

**Interfaces:** `hv_targets(inst: np.ndarray, min_size: int = 30) -> np.ndarray` remains importable from `nucseg.hovernet.data`; `validate_hv_min_size(value) -> int` rejects bool, non-integral and nonpositive values.

- [x] Write tests before implementation. Literal support fixtures:
  ```python
  inst = np.zeros((64, 64), np.int32)
  inst[20:24, 20:24] = 7
  assert not hv_targets(inst, min_size=30).any()
  assert hv_targets(inst, min_size=8).any()
  up = np.repeat(np.repeat(inst, 2, 0), 2, 1)
  assert hv_targets(up, min_size=30).any()
  assert not hv_targets(up, min_size=120).any()
  ```
  Also compare default output with the unmodified official generator on zero-padded random/sparse/border/disconnected masks; check30px inclusion, nonmutation and invalid cutoffs.
- [x] Run `python -m pytest -q tests/test_hv_threshold.py`; expected RED for missing cutoff support.
- [x] Adapt the official full-image HV generator locally, preserving label remapping, mass-center rounding, ±2 bounding box and signed normalization; replace only area selection with the validated argument. Include MIT notice. Keep the4px zero border and public wrapper.
- [x] Rerun target and existing HoVer tests; expected GREEN with exact parity.

## Task 2: Training/CLI wiring and safe experiment identity (complete)

**Files:** modify `src/nucseg/cellvit/data.py`, `src/nucseg/cellvit/engine.py`, `scripts/train_cellvit.py`; create `tests/test_hv_training.py`.

**Interfaces:** `TrainConfig.hv_min_size=30`; `PanNukeCellViT(..., hv_min_size=30)`; CLI `--hv-min-size` inherits an existing run value when omitted, defaults30 for legacy/new runs; `--train-only` stops after training. Extract `build_parser()` and `main(argv=None)` so the real CLI boundary can be tested without GPU training.

- [x] Write tests using small memory-mapped temporary fold data or narrow dataset stubs: train/validation dataset targets receive the specified working cutoff, NP/TP unchanged, x2 is not automatically scaled. Test CLI train-only with a stubbed expensive trainer and forbidden test dataset/model boundaries.
- [x] Add resume tests: existing config cutoff8 is inherited; explicit30 cannot silently resume it; legacy missing field means30; mismatch leaves original config bytes unchanged. Scientific config keys are fixed; only out_dir/workers/val_every may differ.
- [x] Run the new tests; expected RED.
- [x] Thread the field into `train_ds` and `val_dl`; validate before run-directory/config writes. Compare normalized JSON scientific settings before replacing config. Refactor CLI minimally, guard test inference behind train-only, reject incompatible skip-train/train-only flags.
- [x] Run new tests and full suite. CUDA-dependent tests require a usable device; CPU-only run must explicitly report skips, not a full CUDA pass.

## Task 3: Review, commit and launch (complete; experiments running)

**Files:** update `README.md`, `docs/RESEARCH_PLAN.md`, `docs/README.md`, this plan and the research proposal; runtime machine notes/launchers remain `ops/` only.

- [x] Preserve previous evidence-review changes in the user-requested commit; mark HV controls approved/implemented without claiming training results.
- [x] Independently review the complete diff for default parity, provenance, validation-only execution and resume safety; fix substantive findings with reproducing tests.
- [x] Run `git diff --check`, full pytest and CLI help/invalid-input smoke tests; expected clean diff and accurately reported test results.
- [x] Commit the exact portable file allowlist with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- [x] Check storage/free GPU capacity using local runbooks; stage code via an existing safe sync or explicit allowlist, excluding all credentials. Launch sequential resumable queues without killing existing jobs, preserving batch16. Do not reduce batch to squeeze into busy devices; wait for suitable capacity.
- [x] Verify a running training process, saved config and epoch/log progress. A queue alone is reported as queued, not training. Record commit, arm, seed, hardware/batch and process IDs in ignored operational records. No new test-fold evaluation.
