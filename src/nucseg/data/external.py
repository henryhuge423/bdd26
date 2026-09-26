"""Access to external cross-domain datasets converted by `scripts/prepare_external.py`.

Mirrors the `PanNukeFold` interface (images / inst / type / tissue memmaps) so the shared
prediction and evaluation plumbing works unchanged. Class ids follow PanNuke (1..5 in
`constants.CLASS_NAMES` order); classes the source dataset lacks simply never occur.
`gt_channels` is built lazily per image (the evaluate() hot loop indexes it one image at a
time), keeping host RAM flat — required on the address-space-limited ugrad machines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..constants import NUM_CLASSES


def external_root() -> Path:
    """External-data root; override with $EXTERNAL_ROOT (kept off /data7, which is full)."""
    default = Path(__file__).resolve().parents[3] / "data" / "external"
    return Path(os.environ.get("EXTERNAL_ROOT", default))


class LazyGTChannels:
    """Indexable stand-in for the (N, 256, 256, 5) per-class instance-channel array."""

    def __init__(self, inst: np.ndarray, typ: np.ndarray):
        self.inst, self.typ = inst, typ

    def __len__(self) -> int:
        return len(self.inst)

    def __getitem__(self, i: int) -> np.ndarray:
        inst, typ = np.asarray(self.inst[i]), np.asarray(self.typ[i])
        ch = np.zeros((*inst.shape, NUM_CLASSES), np.uint16)
        for c in range(NUM_CLASSES):
            ch[..., c] = np.where(typ == c + 1, inst, 0)
        return ch


@dataclass
class ExternalSet:
    name: str
    root: Path | None = None

    @property
    def dir(self) -> Path:
        return (self.root or external_root()) / self.name

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text())

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
    def gt_channels(self) -> LazyGTChannels:
        return LazyGTChannels(self.inst, self.type)

    def __len__(self) -> int:
        return len(self.tissue)
