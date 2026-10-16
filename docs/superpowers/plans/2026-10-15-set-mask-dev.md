# E6 Set-mask Dev Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user selected native work in this chat; do not create separate chats.

**Goal:** Execute the approved four-run B/Q development comparison with matched recoverable data, original-resolution query masks, frozen gates and durable LM2 provenance.

**Architecture:** A new `nucseg.setmask` package shares existing UNI/decoder components without changing historical model/training defaults. A new dev-only engine trains both B and Q on the same stateless sample stream. Deployment uses an isolated LM2 source snapshot and the already authorized persistent connection.

**Tech Stack:** Pinned Python3.10, torch2.5.1+cu124, NumPy1.23.5, SciPy1.10.1, existing albumentations/timm/evaluator. No dependency installation or upgrade.

**Spec:** `docs/superpowers/specs/2026-10-15-set-mask-dev-design.md`, approved in this conversation, committed as `b5b42c6`.

## Global Constraints

- Only split1: fold1 training/fold2 validation; fold3 rejected before data/model access.
- B/Q × seeds19/1,130epochs,unfreeze25,batch16,workers8,lr3e-4,AdamW betas(.85,.95),weight_decay1e-4,ExponentialLR gamma.85,sampling_gamma.85,native256,noTTA,final checkpoint.
-512queries,3pre-norm decoder layers,dim256,8heads,FFN1024,dropout.1,maskdim128;6query classes with0no-object.
- Matching costs2class+5BCE+5Dice. Loss2CE+5full-nativeBCE+5full-nativeDice+.1tissueCE; no-object weight.1,positive class weights1.
- Query class threshold.5,mask threshold.5,16-query streaming chunks,deterministic lower-query-ID ties,no area filter/NMS/HV inQ.
- Main gate two positive seed deltas,mean≥.02,paired1000-bootstrap95%lower>0;DeadPQ bothnonnegative/mean≥.005;mPQ/bPQ/strictDead floors−.002,strictmPQ eachnonnegative,merge≤+.002;latency≤2×B.
- Max4formal runs,4GPUh/run,16GPUh total;smoke≤20minutes;output≤32GiB,initial free≥52GiB with20GiB reserve. Failure stops; no auto retry/reconnect or expanded recipe.

## Review Focus

1. Tiny/empty masks must contribute valid training behavior without vanished positives or NaN losses; Task2 tests this directly.
2. Different model RNG consumption must not alter sample/augmentation streams; Task1 compares hashes across workers/model activity and epoch resume.
3. Query collisions/overlaps/exact ties must decode reproducibly without loading all512native masks; Task3 checks streamed/full small-tensor equivalence.
4. Partial checkpoints or changed config/source/data must fail closed rather than overwrite or silently resume; Task4 exercises immutable manifests and atomic save.
5. A good DeadPQ with unchanged geometric recall must still fail admission; Task5 separates matched/miss and uses explicit bad-primary fixtures.

## Task1: Stateless paired data and checkpoint state

**Files:** Create `src/nucseg/setmask/__init__.py`, `data.py`, `state.py`; create `tests/test_setmask_data.py`.

**Interfaces:** `sample_keys(weights,seed,epoch)->list[tuple[index,epoch,draw]]`; `PairedDataset(fold,train,seed,root=None)` with `__getitem__(key)->dict` including img,inst_map,np_map,hv_map,tp_map,small_map,tissue,index,sample_key; `capture_rng()->dict`, `restore_rng(state)->None`, `atomic_save(path,**state)`.

- [ ] RED: verify module absence, then add tests for same epoch sequence regardless of global torch/NumPy/random activity, independently seeded replacement draws, workers0/2 identical augmented tensor hashes, no target access to fold3, and epoch-boundary restart.

```python
def test_sample_stream_ignores_model_rng():
    weights=torch.tensor([1.,2.,3.])
    first=sample_keys(weights,19,4)
    torch.rand(100);np.random.rand(100);random.random()
    assert sample_keys(weights,19,4)==first
    assert sample_keys(weights,19,5)!=first
```

- [ ] Implement `sample_keys` using an independent torch.Generator per epoch and `torch.multinomial(...,replacement=True)`; derive epoch generator seed from fixed BLAKE2b, without Python hash().
- [ ] Implement PairedDataset from PanNukeFold and existing normalization/augmentation/HV30 functions. Preserve random/NumPy state around each seeded augmentation. Construct every B/Q target from the same augmented inst/type maps; reject synth/copy-paste/visibility and unsupported folds.
- [ ] Save/restore Python/NumPy/torchCPU/CUDA RNG plus optimizer/scheduler and progress. Use atomic temp+replace for owned new-run files, never mutate older runs.
- [ ] GREEN focused tests, including a dropout toy model resumed across epoch boundary with identical final weights/optimizer. Commit only these new files.

## Task2: Exact targets, matching and full-resolution mask loss

**Files:** Create `src/nucseg/setmask/matching.py`; create `tests/test_setmask_matching.py`.

**Interfaces:** `make_targets(inst,typ,capacity=512)->list[dict]` (ids,classes,inst map,legal yx points); `match_queries(class_logits,mask_embed,mask_features,targets)->list[(query_ids,target_ids)]`; `setmask_loss(pred,batch)->(loss,terms)`.

- [ ] RED with synthetic empty map, IDs2/17, single pixel, disconnected/nonconvex target, exactly512vs513 objects, permuted target order and perfect-vs-wrong predicted masks.

```python
def test_single_pixel_is_in_matching_samples():
    inst=torch.zeros(1,32,32,dtype=torch.long);inst[0,7,11]=17
    typ=(inst>0).long()*4
    t=make_targets(inst,typ)[0]
    assert t['classes'].tolist()==[4]
    assert t['points'].tolist()==[[7,11]]
```

- [ ] Implement nearest-visible-centroid legal points, deterministic row-major ties, majority type with0ignored, capacity and class validation before allocation.
- [ ] Sample shared fixed grid plus legal target points, then compute pairwise BCE as `(sum(softplus(logit))-logit@target.T)/P`, Dice by matrix products and class cost from probabilities; call SciPy Hungarian on detached float32 costs.
- [ ] Use matched query classes for weighted CE and full-native per-instance BCE/Dice. Stream matched mask construction in chunks16. Empty target masks return graph-connected zero; backward must stay finite.
- [ ] GREEN: exact assignments on unique fixtures, loss invariance under target permutation, gradient reaches both query vector and native features; no dense Q×GT×P intermediate. Commit.

## Task3: Query head, shared initialization and streamed decoding

**Files:** Create `src/nucseg/setmask/model.py`, `decode.py`; create `tests/test_setmask_model.py`, `test_setmask_decode.py`.

**Interfaces:** `QueryMaskHead(channels=64,dim=256,mask_dim=128,queries=512,layers=3,heads=8)`; `SetMaskUNI(uni_ckpt=None)` with `freeze_encoder`; output keys class_logits,mask_embed,mask_features,tissue. `decode_set(pred,class_threshold=.5,mask_threshold=.5,chunk_size=16)->(inst,type)`.

- [ ] RED: small head tests on B2×64×32×32 features, full native output spatial shape, shared B/Q initialization, exact tie behavior, no-object rejection, one-pixel survival, batch/streaming equivalence.

```python
def test_decode_chunks_do_not_change_prediction(example_prediction):
    one=decode_set(example_prediction,chunk_size=1)
    full=decode_set(example_prediction,chunk_size=16)
    assert np.array_equal(one[0],full[0])
    assert np.array_equal(one[1],full[1])
```

- [ ] Construct normal CellViTUNI first to preserve public-module RNG order; reuse encoder,skip paths,tp-like native branch,tissue head; discard NP/HV branches and pixel type head. Do not alter CellViTUNI defaults.
- [ ] Implement 32×32 memory pooling/projection/2D sinusoidal positions; three specified attention blocks and query class/mask heads. Use native64→128 mask projection and sqrt128-scaled dot product.
- [ ] Decode accepted queries in native chunks16 with max-score/lower-index winner semantics; discard empty masks only. Return int32 instances anduint8 pixel types; handle no accepted queries without allocating full mask stack.
- [ ] GREEN focused tests and exact initialized common parameter comparisons. Record actual parameter counts; commit.

## Task4: Dev-only training/prediction CLI with frozen configuration

**Files:** Create `src/nucseg/setmask/engine.py`, `scripts/train_setmask_dev.py`, `scripts/predict_setmask_dev.py`; create `tests/test_setmask_engine.py`.

**Interfaces:** immutable `SetMaskConfig(arm,seed,out_dir,uni_ckpt,workers=8)`; `train_dev(cfg,device='cuda')`; `predict_dev(run,fold=2,device='cuda',batch_size=16)`; baseline uses existing `cellvit_loss` with unchanged TrainConfig defaults.

- [ ] RED: reject fold3/seed changes/arm changes/missing provenance/partial checkpoint, preserve existing output on refusal, compare uninterrupted/resumed tiny engine using the Task1 stream.

```python
def test_dev_cli_rejects_fold3_before_loading():
    with pytest.raises((ValueError,SystemExit)):
        predict_dev(Path('missing'),fold=3,device='cpu')
```

- [ ] Build paired datasets once per epoch with separate loader generator, replace persistent sampler state with deterministic epoch keys, hash emitted keys/augmented target sequence into logs. Both arms use identical data policy.
- [ ] Train under bf16 forward/float32 losses, freeze/unfreeze encoder at25, same optimizer/scheduler for130epochs. Validation loss every5epochs, final only for scientific evaluation. Use separate new output/config and capture complete epoch checkpoint state.
- [ ] Predict only fold2 final checkpoints. B uses existing predict_fold; Q uses native streamed decode. Save compressed predictions and invoke existing `scripts/eval_pannuke.py` in a separate process. Never import/load fold3 through these CLIs.
- [ ] GREEN focused tests, then full CPU suite with CUDA disabled; report real pass/skip/fail counts. Commit.

## Task5: Frozen analysis and end-to-end benchmark

**Files:** Create `scripts/analyze_setmask_dev.py`, `scripts/benchmark_setmask_dev.py`; create `tests/test_setmask_analysis.py`.

**Interfaces:** `analyze_round(root,out)->dict`; `admission(comparison,endpoints,cost)->dict`; benchmark writes JSON with config/source/input/checkpoint hashes, device,load snapshots,repetitions and per-image latency.

- [ ] RED: positive DeadPQ/failed recall must fail; strictmPQ−epsilon must fail; matched and miss endpoints must differ on a maxIoU.3 object; changing source/fold/seed fails before analysis.

```python
def test_mid_iou_is_neither_matched_nor_missed():
    row={'max_iou':.3}
    assert not (row['max_iou']>.5)
    assert not (row['max_iou']<.1)
```

- [ ] Reuse existing immutable geometric helpers and official summaries; do not modify analyze_fusion_arms or metrics. Bind741interiorDead and paired object keys. Use one shared tissue-stratified image draw across arms/seeds,1000draws/seed20261015; separately calculate actual miss.
- [ ] Implement all specified per-seed floors and cost gate exactly; no subgroup or threshold search. Output complete endpoints,criteria,hashes,limitations.
- [ ] Benchmark fixed66fold1images,batch1,16warmup,3repeats; include preprocess/H2D/forward/native decoding with CUDA synchronization. B/Q same device/process settings, run serially; no accuracy-based benchmark selection.
- [ ] GREEN synthetic gate and pairing tests, full suite, commit.

## Task6: Isolated LM2 deployment and GPU smoke

**Files:** New ignored `ops/setmask_e6_20261015/{manifest.json,smoke.py,run_queue.py,HANDOFF.md}`; LM2 new `ops/staging/setmask_e6_20261015/` and new `runs/setmask_e6_20261015/`.

- [ ] Build a scoped portable source archive (src/scripts/tests/configs/env/project metadata and approved docs only), explicitly excluding Git internals/credentials/ops/data/weights/old runs. Compute every source SHA and archive SHA.
- [ ] Through worker broker status, require livePID1786103 and connection/TCP/auth1/1/1; current main broker also must satisfy its guard. Use this session for all transfer/exec; never invoke legacy SSH/rsync sync scripts.
- [ ] Recheck disk≥52GiB and actual free GPU memory/jobs. Current LM2 had~2TiB disk free;GPU7 was nearly empty but must be rechecked. ExcludeGPU2 and preserve all other jobs. Verify fold1/fold2 prepared inputs, UNI checkpoint bytes hash, pinned environment and imported source paths.
- [ ] Create fresh staging directory exclusively; symlink existing LM2 data/weights/pinned third-party paths without overwriting anything. PYTHONPATH points to frozen staging src. Run tests there before GPU use.
- [ ] Smoke B/Q frozen and unfrozen,batch16; finite losses/gradients,actual params,peak memory/step time/checkpoint sizes and stream hashes. If runtime/storage/memory budget cannot hold, stop before formal training and report measurements. No architecture tuning via smoke.

## Task7: Launch bounded runs, evaluate, hand off

- [ ] Freeze final source/config/input/weight hashes and smoke receipt. Allocate one currently available admitted GPU; queueB19,Q19,B1,Q1 serially by default. No user/teammate process termination, no implicit parallel allocation.
- [ ] Detached queue closes inherited stdin/stdout/stderr, writes owned PID/events/logs; every run has unique output. Use per-run4h limit and total16h formal GPU budget. Failure terminates only owned exact process group and stops queue; no retry or silent resume.
- [ ] After each final checkpoint, separate prediction/evaluation processes; verify final weights finite, retain last/final onLM2. After all runs, benchmark and run frozen analysis. Do not start extra seeds, capacity controls, confirmation or testfold.
- [ ] Return small JSON/CSV and provenance over same connection, compare hashes; checkpoints stayLM2. Update research log/result report with actual outcomes, cost and limitations, preserving pre-existing edits.
- [ ] Fresh whole-change review using the available code-review capability; fix blockers within spec and recheck affected tests. Final handoff records exact branch/commit/files/tests/PIDs/logs/outputs and session counters. If formal runs remain active, report their true state and recovery entry without claiming results.

## Plan self-review

All approved structure,loss,inference,data/restart,capacity,quality/cost and deployment rules map to tasks1–7. Source boundaries preserve historical defaults. No fold3 path or automatic recovery is introduced. Resource figures are admission budgets; current readings are not future reservations. Execution remains native in this chat; the written plan requires user review before code implementation under the writing-plans workflow.
