# CLAUDE.md

Read README.md (layout, commands) and docs/RESEARCH_PLAN.md (goals) first.
Machine-specific instructions, when present, live in the untracked `CLAUDE.local.md`.

Hard rules for this project:
- Never unpack PanNuke float64 zips to disk; use `scripts/prepare_pannuke.py` (streams from zip).
- Never evaluate released "all-data" checkpoints (official HoVer-Net, CellViT, NuLite, BiomedParse)
  as test results — they saw every fold. Use them only as pipeline sanity checks.
- Evaluate with `nucseg.metrics.pannuke_eval` (tested bit-exact vs PanNuke-metrics run.py; the official
  `class_stats.csv` has a column bug). Test on the last checkpoint, report 3-split mean.
- Reference environment: Python 3.10, torch 2.5.1+cu124, numpy 1.23.5 for imgaug/CellViT;
  see `env/requirements.txt`. Do not implicitly upgrade pinned dependencies.
- Run `python -m pytest -q` after touching metrics or HoVer-Net plumbing.
- Keep reusable code in `src/` or `scripts/`, never in the ignored `runs/` artifact tree.
- Git tracks portable project code, tests, research documentation and reports only.
  Keep machine configuration, orchestration, credentials and operational notes in ignored
  local files (`ops/`, `CLAUDE.local.md`); do not force-add them or include them in uploads.
