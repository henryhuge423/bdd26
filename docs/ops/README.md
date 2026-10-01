# Ops runbook — machines, sync, ugrad constraints

Internal operations reference (split out of the top-level README). Dated incident notes live
alongside this file (e.g. `ugrad_2026-09-30_phase2.md`). Hard rules that bite during
experiments are also duplicated in `CLAUDE.md`.

## Machines

| machine | GPUs | storage | role |
|---|---|---|---|
| LM1 | 8x A100-80G, shared | `/data7/jinxinhao/bdd26` | source of truth, training |
| LM2 | 8x A100-80G, shared | `/data6/jinxinhao/bdd26` | training (`scripts/push_lm2.sh`) |
| ugradx / ugradv | 2x L4 each | local `/tmp/cgf2604` | eval, inference, CPU jobs (32 cores) |

## Code / data sync

```bash
scripts/push_ugrad.sh [--no-data] [--weights] [host]    # code (+data) -> ugrad /tmp, bootstrap venv
scripts/pull_ugrad.sh [--weights] [host]                # runs/ (+weights/) back to LM1
scripts/push_lm2.sh [--no-data] [--weights]             # code (+data) -> LM2, ensure env
# on ugrad:  source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh
```

## ugrad (L4) constraints and how they are handled

- The NFS home (shared by both machines) has a 15 GB quota -> **nothing big goes in home**. Code, venv,
  data, caches and weights live on each machine's local `/tmp/cgf2604` (~70 GB free each).
  `scripts/ugrad_env.sh` redirects every cache (uv, pip, HF, torch, wandb, triton...) there.
- `/tmp` is purged after 10 days without access -> `ugrad_bootstrap.sh` installs a daily cron that
  touches `/tmp/cgf2604`; `scripts/push_ugrad.sh` restores everything idempotently from LM1.
- **Hard 16 GB address-space limit per process** (`limits.conf: @users hard as`). torch+CUDA use ~9 GB of
  it; host-side memory (model copies, memory-mapped folds, eval arrays) eats into the same budget, so
  `predict_fold` preallocates its output maps (constant host memory, batch-invariant) and evaluation
  runs as a separate process (`scripts/eval_pannuke.py`), never inside the inference process. CONCH
  nucleus prior: 2.6 GB peak. There is also a per-process CPU time limit (~7 days); long jobs must be
  resumable.
- **Per-process GPU memory is ENFORCED at ~3.6 GiB** on the L4s (2026-09-30: CUDA OOM at 3.61 GiB in use
  with 18.4 GiB free; the pre-2026-09-30 "6.9 GB" figure was wrong). x2 inference at batch 2 sits at a
  stable 3.33 GiB -> keep UGRAD_BS=2; batch size is result-invariant (<= 2.8e-05 on every rate metric,
  `runs/_ab_verdict.md`). The cap behaves **per-user across GPUs**: two concurrent CUDA processes of
  ours (one per L4) die silently mid-run (SIGKILL, no traceback) while each alone runs fine — run ugrad
  GPU jobs strictly sequentially, one process at a time. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is REJECTED by the cap — never set it on ugrad; right after a cap-OOM crash, wait a few minutes before
  relaunching (driver reclaim is slow). Training needs 20-36 GB -> all training on the A100s; ugrad is
  inference / CONCH / eval only.
- Downloads from huggingface.co / Google Drive are much faster on ugrad than LM1; download there and
  `scripts/pull_ugrad.sh --weights`. hf-mirror.com (LM1 default) does not serve gated repos.

## Tokens / gated weights

HF token: `hg-token.txt` (git-ignored). Gated weights (CONCH, UNI, UNI2-h, PixCell-256 +
Cell-ControlNet) are in `weights/` (`scripts/download_weights.sh`).

## Known gotchas (details in the dated notes)

- **`pkill -f` session suicide**: never put `pkill -f <pattern>` and the same script's launch
  line in ONE ssh command — the whole command is one `bash -c` cmdline, so the regex matches the
  unbracketed launch text and kills the session (silent exit 255, looks like network flapping).
  Bracket every occurrence (`'queu[e]'`), make sure the bracketed word does not appear as a plain
  word anywhere else in the same cmdline (log *paths* count), or split clean/launch into two ssh
  sessions. Safest: pgrep first (bracketed), kill by PID list.
- **x2 saved preds have non-contiguous instance ids** (nearest downsampling drops small-instance
  ids; e.g. 39 unique ids with max id 40). Any per-instance array must be dense over
  `1..max_id`, not compact over unique ids (fix 6b62a66).
- **LKCell construction peaks on GPU**: the vendored `CellViT()` builds itself on device
  (`.to()` inside `__init__`), so construction counts toward the cap; at 256² use
  `--batch-size 3` (~2.7 GiB; batch 8 ≈ 4.2 GiB is over cap) — predict-script defaults were
  tuned for CellViT, not LKCell.
- LM2 -> ugrad direct ssh works via a keypair generated on LM2 (`scripts/hosts_lm2.sh`); the
  LM1 private key never left LM1.
