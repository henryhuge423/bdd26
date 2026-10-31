# P21 Lustre timeout investigation and resource amendment

Recorded 2026-10-31, America/New_York. This is an operational amendment; scientific source `911a3718c157ceb51b2924feb18c50a3165a1a73`, the 438-file seal, cohort, seeds, 130 epochs, losses, last-checkpoint evaluation, fixed point examples and original two-seed decision rule remain unchanged. No formal P21 score is available yet.

## Findings and limits

Replacement array 55440148: C19/C1/A1 failed before training, in 420/267/237 seconds; A19 continues on exp-10-57. Their quota logs distinguish query failure from a capacity denial. C1's first query returned exit5, bracketed/inaccurate usage and a warning about unavailable/deactivated devices; subsequent queries timed out. C19's scratch query passed, then project-user queries timed out. A1's scratch queries timed out. The three 45-second attempts were insufficient during this incident.

A read-only login-node strace located the wait in the Lustre quota ioctl, after mount lookup; a 50-second probe was terminated at 52 seconds including kill grace. A separate numeric-UID query completed successfully after 33.243 seconds inside the same ioctl. Numeric identity did not avoid the storage call. At 12:51 UTC, 118 of 144 OSC client instances had CONNECTING transitions within 30 minutes, with 91 currently FULL and 53 IDLE. FULL here is a connection state, not a full disk. The two mounts are distinct Lustre subdirectory mounts named expanse/scratch and expanse/projects; similar totals are not evidence of independent physical capacity.

This establishes intermittent Lustre client/target connection recovery and quota RPC availability as the observed failure layer. It does not identify which server, fabric component or administrative change caused it. We cannot repair the cluster fabric or quota service as a normal user. The repair below makes our admission workflow tolerate longer recovery and prevents GPU allocation if an independent CPU admission fails. It does not establish that the underlying storage service is healthy.

Accurate A19 startup records showed approximately 8.33 TB of user headroom, 1.896 million free quota inodes and 52.7 TB of group headroom; the later successful numeric-UID query remained far below its limit. This is not an observed quota-exceeded event. The 80/100 GiB values in our guards are minimum-headroom thresholds, not reserved storage requests. Lowering them would not make a blocked quota ioctl faster. `lctl` also reported a shared-data creation ENOSPC error; login /tmp had 6.9 GiB and /dev/shm 125 GiB available, so that tool error alone cannot establish full project storage. No system objects or other users' files were removed.

## Implemented mitigation and validation

Portable helper commit `cfa0c176dc3d795241a193d99b4bf7bfcb81f950` changes default query timeout from 45 to 180 seconds, supports an explicit maximum of 300 seconds, retains three attempts, and records UTC start, observed elapsed time and partial stdout/stderr on timeout. Invalid/inaccurate data, nonzero statuses and verified insufficient capacity still cannot pass. Longer waits are a bounded mitigation for measured recovery latency, not a guarantee against a sustained outage. Kernel waits and process cleanup can exceed a nominal subprocess timeout; elapsed diagnostics expose that difference.

Two new behavioral regressions failed against the old helper (unsupported long-query budget and missing partial timeout diagnostics), then the quota/P21 focused checks reported **29 passed in 16.95s**. These are focused storage/CLI tests; the full suite was not rerun because no metrics, HoVer-Net or model/loss behavior changed. Earlier full-suite results remain historical. Both new launchers passed `bash -n` and `sbatch --test-only` before submission. Preparation-only quoting and Python3.6 `text=` compatibility errors were corrected before actual submission; receipts and queue were checked before continuing. No duplicate GPU allocation resulted.

CPU gate **55450367**: debug/debug-normal, one CPU, 2 GiB, 30 minutes, no GPU. It queries scratch user, project user and project group, each requiring 100 GiB and 1000 free quota inodes, then performs small exclusive-create/fsync/readback probes. Only successful completion releases GPU array **55450368** through `afterok:55450367`; invalid dependency is automatically cancelled without GPU execution. Each GPU task also performs current fail-closed quota checks before training, so a gate passing hours before scheduler placement is not a stale-admission bypass.

Array 55450368 contains only indices **0=C19, 2=C1, 3=A1**, concurrency **1**, each **1 V100 / 4 CPUs / 16 GiB / 10 hours**, no requeue. A19 **55440148_1** is neither cancelled nor restarted; its original launcher/helper and running allocation remain intact. While A19 runs, at most two P21 GPUs can be active. Failed-run logs and original directories remain preserved; recovery training uses a fresh output root. The same original scientific snapshot, readiness evidence and source hashes are checked.

## Resource sizing and A19 estimate

At 12:51 UTC A19 had 56 completed epochs, using 1-based epoch numbering. Its recent 20-epoch mean was 134.374 seconds, median 123.267; training-only mean 122.857 seconds, checkpoint mean 11.481 seconds, checkpoint maximum 87.242 seconds. Peak recorded GPU allocation was 4,356,422,144 bytes (4.06 GiB), and SLURM observed step MaxRSS 1,874,876 KiB (1.79 GiB). Prior P20 256-input controls completed with step MaxRSS below 2 GiB. Sixteen GiB host RAM gives substantial margin for model loading and final evaluation; 32 GiB was generous. Host RAM and GPU VRAM are different resources; requesting one V100 still allocates a whole GPU.

A19 reached 58 completed epochs at 12:54 UTC. Remaining training at the recent mean is about 2.69 hours; allowing final evaluation, point examples, quota checks and preservation gives approximately **3–4 more hours**, around **12:00–13:00 EDT**, conditional on storage behavior. GPU allocation had already run 4h54, including a measured 92m38 before the srun training step. Additional initialization occurs inside training; its individual causes were not traced. Avoid treating all startup time as quota latency, or extrapolating the entire allocation time per epoch. A19 retains its 12-hour limit because storage stalls and final preservation remain uncertain. New tasks use a 10-hour cap instead of 12 hours; this is not an assurance of completion under a prolonged outage.

The corrected conservative P21 envelope is **12 + 3×10 + (37+420+267+237)/3600 = 42.266944 GPU-hours**, including A19's full cap, the three new caps and all known failed allocations. Scheduler cleanup or rounding may add time. The earlier recovery script requested 11:59:50, but SLURM reports an effective 12:00:00; use actual scheduler limits, not a claimed ten-second saving. The new envelope still remains below the original 48 GPU-hour formal envelope. CPU gate charges are separate (one CPU/2 GiB/0.5h request; minimum job charge may apply).

Current `gpu-shared` billing weights were GPU=3600, CPU=360/core and memory=37.40/GiB. One GPU dominates both the previous 4-CPU/32-GiB request and the new 4-CPU/16-GiB request. Reducing RAM helps packing and avoids excess reservation; it does not halve GPU SU charging. A lower walltime may improve backfill eligibility; actual queue delay is not predictable. See the [SDSC job charging and storage guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and [Lustre quota documentation](https://www.lustre.software/repos/master/?path=Documentation%2Fman1%2Flfs-quota.1).

## Handoff and remaining risks

At the initial post-submission check, gate55450367 was RUNNING, array55450368 PENDING/Dependency, and A19 RUNNING; submission does not mean training resumed. The final status below supersedes this snapshot. If the CPU gate fails, new GPU tasks do not start. Inspect the exact gate logs rather than replaying submission or weakening the quota check.

A19's original post-training preservation still uses the old 45-second helper. If it fails after `complete.json` records 130 epochs, preserve scratch and recover the completed artifacts through a separate CPU copy with accurate quota admission; do not retrain or alter the checkpoint. External independent checkpoint archival remains outstanding.

Private launchers, traces, gate JSONL, scheduler logs, submission receipts and recovery paths are retained in LM1 ignored `ops/p21_storage_recovery_20261031/`. Future collection must include A19 from the original root and C19/C1/A1 from the fresh recovery root, along with each launch provenance, fixed point PNG/coordinate JSON and recursively verified preservation manifests. No fold2/fold3/test, masks or automatic Phase B was added.

## Verified final admission, 09:00 EDT

CPU gate **55450367 COMPLETED/0:0**, elapsed **330 seconds**, with an exclusive PASS receipt at **2026-10-31T12:59:37.431326Z**. All three checks and all fsync/readback probes passed. Scratch-user query took **47.741s**, project-user **26.238s**, project-group **27.241s**, each on its first attempt. The 47.741-second successful query directly demonstrates why a 45-second cutoff could reject healthy-but-slow responses; the extended timeout has now admitted a real response beyond that old bound. User headroom remained **8.328 TB**, group headroom **52.733 TB**, with **1.895 million** user inodes available. The group inode quota is unlimited. The first helper start timestamp was 12:57:48 UTC, about 3m42 after allocation began; this portion precedes the measured query and must not be labelled ioctl time without further tracing.

The GPU dependency has cleared: array **55450368** is now **PENDING/Priority**, rather than Dependency. A19 remains RUNNING. This proves successful admission on a real CPU node; it does not yet prove the replacement GPU tasks have started or the cluster-wide instability has ended. [Sanitized gate receipt](P21_STORAGE_GATE_2026-10-31.json).

### 中文说明

问题有两个层次：Lustre 目标连接反复恢复，导致 quota ioctl 时延波动甚至返回不完整结果；原检查只等45秒，把部分最终能够成功的慢查询判为失败。我们修复了可控制的检查时限与诊断，并用不占GPU的CPU作业先验收。真实查询47.741秒成功，用户/组空间充足，三项quota和fsync回读均通过；这不是把容量不足的错误忽略掉，也不能据此声称集群存储服务已经修好。

三个失败任务已改为每项10小时、16GiB主存、串行，占用至多1张V100；A19继续原进程。32GiB主存偏宽松，但显存与主存不同，一张GPU仍是最小GPU申请单位。当前单GPU主导计费，缩小主存不会把GPU费用减半。80/100GiB是最低剩余空间门槛，并非向调度器申请磁盘。A19按实测速度还需约3–4小时，12小时是截止上限，不代表必然运行或计费12小时。若存储继续阻塞，预计时间会延后。总GPU上限按实际调度时限与失败消耗修正为42.266944小时，低于原48小时。
