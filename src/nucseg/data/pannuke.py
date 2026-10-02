"""Memory-mapped access to the compact PanNuke format written by `prepare.py`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..constants import SPLITS


def data_root() -> Path:
    """PanNuke root; override the default data/pannuke location with $PANNUKE_ROOT."""
    default = Path(__file__).resolve().parents[3] / "data" / "pannuke"
    return Path(os.environ.get("PANNUKE_ROOT", default))


@dataclass
class PanNukeFold:
    fold: int
    root: Path | None = None

    @property
    def dir(self) -> Path:
        return (self.root or data_root()) / f"fold{self.fold}"

    @cached_property
    def images(self) -> np.ndarray:
        return np.load(self.dir / "images.npy", mmap_mode="r")

    @cached_property
    def inst(self) -> np.ndarray:
        return np.load(self.dir / "inst.npy", mmap_mode="r")

    @cached_property
    def type(self) -> np.ndarray:
        return np.load(self.dir / "type.npy", mmap_mode="r")

    @cached_property
    def tissue(self) -> np.ndarray:
        return np.load(self.dir / "tissue.npy")

    @cached_property
    def gt_channels(self) -> np.ndarray:
        """(N, 256, 256, 5) original per-class instance channels (loads ~1.8 GB into RAM)."""
        return np.load(self.dir / "gt_channels.npz")["gt"]

    def __len__(self) -> int:
        return len(self.tissue)


def split_folds(split: int) -> tuple[int, int, int]:
    """(train, val, test) fold ids for official split 1/2/3."""
    return SPLITS[split]
