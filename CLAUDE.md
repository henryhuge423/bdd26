# CLAUDE.md

Read README.md (layout, machines, commands) and RESEARCH_PLAN.md (goals) first.

Hard rules for this project:
- Never unpack PanNuke float64 zips to disk; use `scripts/prepare_pannuke.py` (streams from zip).
- Never evaluate released "all-data" checkpoints (official HoVer-Net, CellViT, NuLite, BiomedParse)
  as test results — they saw every fold. Use them only as pipeline sanity checks.
- Evaluate with `nucseg.metrics.pannuke_eval` (tested bit-exact vs PanNuke-metrics run.py; the official
  `class_stats.csv` has a column bug). Test on the last checkpoint, report 3-split mean.
- ugrad (L4) machines: nothing big in the NFS home (15 GB shared quota); work in `/tmp/cgf2604` after
  `source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh`. <=6.9 GB usable GPU memory per process (16 GB
  address-space hard limit; ~4.2 GB for CellViT-UNI) -> inference / CONCH / eval only, no training.
- Env on LM1/LM2: `~/.conda/envs/nuclei` (py3.10, torch 2.5.1+cu124, numpy 1.23.5 for imgaug/CellViT).
- Run `python -m pytest -q` after touching metrics or HoVer-Net plumbing.
