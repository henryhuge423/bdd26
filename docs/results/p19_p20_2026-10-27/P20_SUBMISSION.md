# P20 validation and formal submission

Two once-only arrays were submitted at 2026-10-28T15:14:17.622329Z UTC: **C 55357457**, indices0/2, seeds19/1,12hours/task; **U 55357476**, indices1/3, seeds19/1,36hours/task. Each array has concurrency1; combined maximum2. Four fresh130epoch runs each request1V100/4CPU/32GiB, gpu-shared/gpu-shared-normal/jhu159, Lustre/no-requeue. Total requested formal walltime remains **96GPUhours**; actual SLURM accounting may include cleanup after a requested limit. Submission does not imply completion or scientific admission.

## Frozen scientific recipe
Protocol4b141ce predates inspection of P19 scores. C encodes256; U bilinearly enlarges to512 after original augmentation and before normalization. Four encoder tap grids are16²/32²; both PointHeads output native256 logits with identical parameters. Original cohort, targets, losses, augmentation, last12 blocks, optimizer/LR/schedule,130epochs, last checkpoint, decoding and two-seed gate remain fixed. Effective16/physical4 both arms. This is a whole working-scale recipe including attention/position interpolation/internal decoder size; it adds no optical information. PhaseB/masks/fold2/3/rescue menus remain closed. [Protocol and resource amendment](../../superpowers/specs/2026-10-27-proposal-p20-design.md).

## Failed original readiness and resource-only amendment
Original smoke **55343640 FAILED/1:0**, elapsed992seconds. All367 proposal tests, dual-seed full control and U learning witnesses, C134-image fitting/reloads, and U134-image fitting/last+final reloads had completed. The stop was the original U24hour safety estimate:25.760018hours; C7.069196hours. This was a real readiness failure and is preserved.

Before another allocation, commit04d187e redistributed the same96GPUhour formal budget to C12/U36 hours. Scientific choices were unchanged; no P20 audit score was read. Exact reverse transformation proves the main runner differs only in resource guards; all inherited computational dependencies retain their hashes. Six new resource tests failed before implementation and passed afterward. The second validation55356965 requested800seconds but TIMEOUT accounting was849seconds, including scheduler cleanup. Its six resource checks completed (step153seconds); the U step was cancelled after325seconds without completing U134/reloads. Original992 plus849 equals1841seconds, exceeding the original1800second/.5GPUhour budget by41seconds. This failure is preserved and is not called a U validation pass.

Before any third allocation, docs-only commit08ae279 froze one additional25minute/1500second validation, increasing total actual-accounting acceptance to3600seconds/1GPUhour. Existing1841 leaves1759seconds actual headroom;1500requested reserves259seconds for cleanup but cannot guarantee cleanup duration. Formal96GPUhour allocation and all scientific choices remain unchanged. All302computational/test files match the02seal byte-for-byte, so the actual950pass CPU run is reused for this documentation-only source change. No audit score was read. Exceeding3600seconds or failing any readiness check blocks formal submission.

## Actual successful validation
Third validation smoke **55357273 COMPLETED/0:0**. Actual cumulative elapsed across all three GPU validations **2260seconds**. It ran six new resource tests and fresh U134-image training with AMP, gradient/update and frozen-parameter checks and strict last/final reloads. C and dual-seed controls were inherited from the original55343640 run with nine evidence-file SHA checks against both source and copied files; the367 earlier node checks were inherited, not rerun on this allocation. The sealed parent425-file inventory and exact resource-only code transformation are revalidated by strict readiness. No audit predictions or scores enter readiness.

Both seeds' original control witness establishes C/P3 initial states/RNG and two-effective-step complete trainable gradients, parameters, AdamW states and CPU/CUDA RNG equality. U starts with identical parameter states/RNG, actually encodes512 with four1025token taps, outputs256, and shows second-step block12/block23/output gradients and updates beyond pure weight decay. Frozen encoder tensors stay unchanged. Exact optimizer membership, native targets,134fit images/nine effective steps/tail6, cross-arm config rejection and source/input/UNI/cohort identities are checked.

| Arm | Training peak GiB | 1.5 safety estimate h | Task limit h |
|---|---:|---:|---:|
| C256 (inherited) | 4.0709 | 7.0692 | 12 |
| U512 (new run) | 9.0193 | 29.9832 | 36 |

Estimates include measured training,130epoch saves, final save and1800seconds overhead before multiplying1.5. They are estimates, not completion guarantees. Memory limit24GiB. All 27 collected validation files passed byte/SHA checks. Full CPU suite: **950 passed, 3 skipped, 19073 warnings in 646.58s (0:10:46)**, wrapper649.55seconds, exit0. Three skips are legacy CUDA-only checks. The initial r2 TMPDIR generated a109-byte multiprocessing AF_UNIX listener path, which directly reproduced OSError: AF_UNIX path too long in the pinned Python. A shorter path bound successfully; the original pre-amendment path was three characters shorter. Four incomplete full-process attempts, the global-spawn diagnostic (four existing metric failures/299passed/3skipped before interruption), and a full isolated-test attempt (949passed/1failed/3skipped, child timeout) are retained honestly. Earlier thread/context hypotheses did not fix this path problem; gdb attachment was unavailable and security settings were not changed. With only TMPDIR shortened, all15 focused original DataLoader/metric tests passed using explicit PyTorch threads2 and default fork. The final full python -m pytest uses default fork, OMP/OpenBLAS/MKL4, all original tests and no custom plugin, process isolation or science/test-source changes. No tests were omitted. Initial missing-vendor collection errors were fixed by linking the authoritative existing vendor root, without upgrades or removing tests. All failed logs remain preserved.

### Reproducible CPU environment correction
Actual command: `/data7/jinxinhao/.conda/envs/nuclei/bin/python -m pytest -q -rs --basetemp=/data7/jinxinhao/bdd26/ops/p20_development_20261027_r2/cpu_suite_verified/basetemp`; TMPDIR `/data7/jinxinhao/bdd26/ops/t20-l6xqn3o9`. PYTHONPATH points to the existing worktree src/scripts, PANNUKE_ROOT to the authoritative compact data, CUDA_VISIBLE_DEVICES empty, OMP/OpenBLAS/MKL4. No test plugin or changed multiprocessing context is used. The long109-byte socket probe fails explicitly; the short-path probe and original15 focused cases pass. Temporary directories and all failed attempts remain in ignored operations evidence.

## Provenance and live status
Source commit **08ae279dbfd8f4fba71404d5f5af7872f25f5b64**; **426files**, manifest **1bd6e680e1b4c008f11b97ee50c165c21c4776209f718a915aca101333228e58**. Existing pinned runtime/data/UNI reused. No bulk data transfer, cleanup or unrelated cancellation. Checkpoints copy/hash within Lustre; no independent checkpoint transfer/rehash/external archive is claimed. [Machine-readable receipt](P20_RECEIPT.json).

```text
JOBID    PARTITION                 NAME      STATE       TIME               NODELIST(REASON)
            55357457_2   gpu-shared          bdd26-p20-C    PENDING       0:00                     (Priority)
            55357457_0   gpu-shared          bdd26-p20-C    PENDING       0:00                     (Priority)
            55357476_3   gpu-shared          bdd26-p20-U    PENDING       0:00                     (Priority)
            55357476_1   gpu-shared          bdd26-p20-U    PENDING       0:00                     (Priority)
JobIDRaw|State|ExitCode|Elapsed|NodeList
55357458|PENDING|0:0|00:00:00|None assigned
55357457|PENDING|0:0|00:00:00|None assigned
55357477|PENDING|0:0|00:00:00|None assigned
55357476|PENDING|0:0|00:00:00|None assigned
```

After all four runs complete130epochs and preservation, verify source/payload/input/config/cohort then apply the original F1=C/T1=U conjunction in both seeds. Failure/inconclusive closes the exact512 recipe without new scales/seeds/epochs/LR/threshold menus. Reused fold1 point development evidence is not PQ/test/patient-independent confirmation. Do not replay once-only deployment/submission scripts.
