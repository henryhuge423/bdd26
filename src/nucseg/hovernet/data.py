"""PanNuke dataset for HoVer-Net (fast mode) with full-patch supervision.

HoVer-Net fast mode shrinks the output by 92 px (256 -> 164). We reflect-pad each 256 patch by
48 px to 352, so the network outputs 260 and the central 256 covers the WHOLE labelled patch
(the official trainer only supervises the central 164).

Augmentation follows the official hover_net pipeline with the PanNuke-specific note in its
train_loader ("for pannuke v0, no rotation or translation, just flip to avoid mirror padding"):
geometric = random dihedral (flips + 90-degree rotations); photometric = the official
blur/noise + hue/saturation/brightness/contrast augmentations.
"""

from __future__ import annotations

import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from ..data.pannuke import PanNukeFold
from .official import (
    add_to_brightness, add_to_contrast, add_to_hue, add_to_saturation, gaussian_blur,
    median_blur,
)
from .targets import instance_hv_map

PAD = 48
HV_PAD = 4  # zero margin so the official hv generator's +-2 px boxes never wrap at borders


def official_input_augs(seed: int) -> iaa.Sequential:
    return iaa.Sequential([
        iaa.OneOf([
            iaa.Lambda(seed=seed, func_images=lambda *a: gaussian_blur(*a, max_ksize=3)),
            iaa.Lambda(seed=seed, func_images=lambda *a: median_blur(*a, max_ksize=3)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5, seed=seed),
        ]),
        iaa.Sequential([
            iaa.Lambda(seed=seed, func_images=lambda *a: add_to_hue(*a, range=(-8, 8))),
            iaa.Lambda(seed=seed, func_images=lambda *a: add_to_saturation(*a, range=(-0.2, 0.2))),
            iaa.Lambda(seed=seed, func_images=lambda *a: add_to_brightness(*a, range=(-26, 26))),
            iaa.Lambda(seed=seed, func_images=lambda *a: add_to_contrast(*a, range=(0.75, 1.25))),
        ], random_order=True),
    ])


def hv_targets(inst: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Full-patch HV; min_size is an inclusive cutoff in working pixels, not native pixels."""
    p = np.pad(inst.astype(np.int32), HV_PAD)
    hv = instance_hv_map(p, min_size=min_size)
    return hv[HV_PAD:-HV_PAD, HV_PAD:-HV_PAD]


def pad_image(img: np.ndarray) -> np.ndarray:
    return np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="reflect")


class PanNukeHoVer(Dataset):
    def __init__(self, folds: list[int], train: bool, seed: int = 10):
        self.folds = [PanNukeFold(k) for k in folds]
        self.index = [(fi, j) for fi, f in enumerate(self.folds) for j in range(len(f))]
        self.train = train
        self.seed = seed
        self._augs = None
        self._rng = None

    def __len__(self) -> int:
        return len(self.index)

    def _worker_setup(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        base = torch.initial_seed() % (2**31) if info is not None else self.seed
        self._rng = np.random.default_rng(base + wid)
        self._augs = official_input_augs(int(base + wid) % (2**31))

    def __getitem__(self, i: int) -> dict:
        fi, j = self.index[i]
        f = self.folds[fi]
        img = np.array(f.images[j])
        inst = np.array(f.inst[j]).astype(np.int32)
        typ = np.array(f.type[j]).astype(np.int64)
        if self.train:
            if self._rng is None:
                self._worker_setup()
            k = int(self._rng.integers(4))
            flip = bool(self._rng.integers(2))
            img, inst, typ = (np.rot90(a, k) for a in (img, inst, typ))
            if flip:
                img, inst, typ = (a[:, ::-1] for a in (img, inst, typ))
            img, inst, typ = (np.ascontiguousarray(a) for a in (img, inst, typ))
            img = self._augs.augment_image(img)
        return {
            "img": torch.from_numpy(pad_image(img)),               # (352, 352, 3) uint8
            "np_map": torch.from_numpy((inst > 0).astype(np.int64)),  # (256, 256)
            "hv_map": torch.from_numpy(hv_targets(inst)),          # (256, 256, 2)
            "tp_map": torch.from_numpy(typ),                        # (256, 256) 0..5
            "index": i,
        }
