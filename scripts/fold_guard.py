"""Dev/probe CLI guard: refuse a run split's held-out TEST fold (audit 2026-10-08).

The DSB/E0 probe scripts operate on dev folds only (train/val). Their recorded runs never
read a test fold; this guard makes that a machine-checked invariant instead of a comment.
Usage inside a script's main(), right after parse_args:

    from fold_guard import ensure_dev_fold, run_split
    ensure_dev_fold(a.fold, run_split(a.run))          # run-dir based (config.json split)
    ensure_dev_fold(a.fold, a.split)                   # explicit --split (maps dumps)
"""

import json
from pathlib import Path

from nucseg.constants import SPLITS


def run_split(run: Path) -> int:
    """Load the split id from a run dir's config.json; refuse if it cannot be verified."""
    cfg = run / "config.json"
    if not cfg.exists():
        raise SystemExit(f"{cfg} missing — cannot verify the fold is not a test fold; refusing")
    split = json.loads(cfg.read_text()).get("split")
    if split not in SPLITS:
        raise SystemExit(f"{cfg} has no valid 'split' — cannot verify the fold is not a test fold; refusing")
    return split


def ensure_dev_fold(fold: int, split: int) -> None:
    if fold == SPLITS[split][2]:
        raise SystemExit(f"--fold {fold} is split {split}'s TEST fold; dev/probe tools never read it")
