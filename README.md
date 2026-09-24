# bdd26 — PanNuke nuclei instance segmentation (course project 5)

Research plan: [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Course brief: `our_project5_nuclei_segmentation.pdf`.

## Layout
```
src/nucseg/            our package (pip install -e .)
  constants.py         class / tissue names, official 3-split protocol
  data/prepare.py      stream-convert official float64 zips -> compact uint8/uint16 (37 GB -> 3 GB)
  data/pannuke.py      memory-mapped fold access (PANNUKE_ROOT overrides the location)
  metrics/instance.py  fast PQ / AJI / AJI+ / centroid pairing (tested == official)
  metrics/errors.py    merge / split / missed / FP taxonomy, touching-neighbour counts
  metrics/pannuke_eval.py  official mPQ/bPQ + strict mPQ, mPQ+, per tissue/class, F_d/F_c, errors
  hovernet/            official HoVer-Net (third_party) + our full-patch PanNuke plumbing, TTA
scripts/               CLIs + machine sync (see below)
tests/                 pytest: metrics vs official code, HoVer-Net targets/TTA
third_party/           pinned upstream repos (scripts/fetch_third_party.sh)
data/ weights/ runs/ logs/   large, git-ignored; LM1 is the source of truth
```

## Common commands (LM1, env `~/.conda/envs/nuclei`)
```bash
python scripts/prepare_pannuke.py                       # data/raw/fold_{1,2,3}.zip -> data/pannuke
python -m pytest -q                                     # metric + plumbing tests
python scripts/train_hovernet.py --split 1 --out runs/hovernet/split1 --tta
python scripts/eval_pannuke.py --pred X.npz --fold 3 --out runs/.../eval
python scripts/predict_hovernet.py --ckpt CKPT --fold 3 --out runs/...
```
Official splits (train/val/test): 1 = 1/2/3, 2 = 2/1/3, 3 = 3/2/1. Always test the LAST checkpoint.

## Machines
| machine | GPUs | storage | role |
|---|---|---|---|
| LM1 (here) | 8x A100-80G, shared | `/data7/jinxinhao/bdd26` | source of truth, training |
| LM2 | 8x A100-80G, shared | `/data6/jinxinhao/bdd26` | training (`scripts/push_lm2.sh`) |
| ugradx / ugradv | 2x L4 each | local `/tmp/cgf2604` | eval, inference, CPU jobs (32 cores) |

ugrad constraints and how they are handled:
- The NFS home (shared by both machines) has a 15 GB quota -> **nothing big goes in home**. Code, venv,
  data, caches and weights live on each machine's local `/tmp/cgf2604` (~70 GB free each).
  `scripts/ugrad_env.sh` redirects every cache (uv, pip, HF, torch, wandb, triton...) there.
- `/tmp` is purged after 10 days without access -> `ugrad_bootstrap.sh` installs a daily cron that
  touches `/tmp/cgf2604`; `scripts/push_ugrad.sh` restores everything idempotently from LM1.
- **Hard 16 GB address-space limit per process** (`limits.conf: @users hard as`). torch+CUDA use ~9 GB of
  it, leaving **~6.9 GB usable GPU memory per process** on the L4s (with the env tweaks in
  `ugrad_env.sh`). HoVer-Net training needs 20-36 GB -> train on A100s. There is also a per-process CPU
  time limit (~7 days); long jobs must be resumable.
- Downloads from huggingface.co / Google Drive are much faster on ugrad than LM1; download there and
  `scripts/pull_ugrad.sh --weights`. hf-mirror.com (LM1 default) does not serve gated repos.

```bash
scripts/push_ugrad.sh [--no-data] [--weights] [host]    # code (+data) -> ugrad /tmp, bootstrap venv
scripts/pull_ugrad.sh [--weights] [host]                # runs/ (+weights/) back to LM1
scripts/push_lm2.sh [--no-data] [--weights]             # code (+data) -> LM2, ensure env
# on ugrad:  source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh
```

HF token: `hg-token.txt` (git-ignored). Gated weights (CONCH, UNI, UNI2-h, PixCell-256 + Cell-ControlNet)
are in `weights/` (`scripts/download_weights.sh`).
