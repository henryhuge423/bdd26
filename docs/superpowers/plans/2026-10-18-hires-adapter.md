# E19 Hi-res adapter 鈥?implementation plan (TDD)

Spec: docs/superpowers/specs/2026-10-18-hires-adapter-design.md (frozen).

## Constraints
- Additive only: do not change `CellViTUNI` defaults or any historical run's numerics.
- One mechanism: `skip0(x) + hires(x)`; `hires` output zero-initialised (identity at init).
- New flag `TrainConfig.hires_adapter: bool = False`; `build_model` selects `HiResCellViTUNI`.
- Guards refuse mixing with other scientific flags; fold3 rejected before data/model access.
- Pinned env; no dependency changes.

## Task 1 鈥?model (RED then GREEN)
Files: `src/nucseg/cellvit/model.py` (add `HiResAdapter`, `HiResCellViTUNI`), `tests/test_hires_adapter.py`.
Tests:
- `HiResCellViTUNI()` at init: `hires` output is exactly 0 -> forward equals `CellViTUNI` forward on the same weights/inputs (bit-identical logits) when hires weights are zero.
- shared modules (encoder, skip0..skip3, branches, heads, tissue_head) have identical parameter shapes and identical initial values as `CellViTUNI` built with the same seed.
- `hires` output spatial shape 256x256 and 64 channels; channel-compatible with `skip0`.
- after setting a nonzero value in the last 1x1 conv, forward output changes (adapter is live).
- `freeze_encoder` still only touches the encoder.
Commit.

## Task 2 鈥?config/engine wiring
Files: `src/nucseg/cellvit/engine.py` (`TrainConfig.hires_adapter`, `build_model`, guards), `scripts/train_cellvit.py` (`--hires-adapter`), `tests/test_hires_engine.py`.
Tests:
- `TrainConfig(split=1,out_dir=...,hires_adapter=True)` accepted; mixing with `fusion!="plain"`, `dead_expert`, `widen`, `upscale!=1`, `cp_prob`, `synth`, `np_wce`, `dead_w`, `small_w`, `visibility_arm` raises ValueError.
- `build_model` returns `HiResCellViTUNI` iff `hires_adapter`.
- `prepare_train_config` rejects a config change (hires_adapter toggled) in an existing dir.
Commit.

## Task 3 鈥?CPU regression
- Focused tests, then full `~/.conda/envs/nuclei/bin/python -m pytest -q` with a project-owned TMPDIR/basetemp on a partition with space. Report real pass/skip/fail.

## Task 4 鈥?LM2 deploy, smoke, launch
- Scoped snapshot to LM2 `ops/staging/hires_adapter_20261018/`; symlink existing data/weights.
- Recheck LM2 disk >=52 GiB and idle GPUs >=2; exclude GPU2.
- CPU tests on LM2 staging; single-GPU smoke (batch16, finite loss/grad, params, peak, step time).
- Detached 2-run queue (H seed19, H seed1) plus matched baseline B0 if budget allows; per-run <=10h.
- Small JSON/CSV returned over the same broker; checkpoints stay on LM2.

## Task 5 鈥?analysis and handoff
- Predict fold2 (last checkpoint), evaluate with `nucseg.metrics.pannuke_eval`; frozen gates from the spec.
- Result doc `docs/results/HIRES_ADAPTER_E19_DEV_2026-10-18.md`; update README/RESEARCH_PLAN; commit/push.
