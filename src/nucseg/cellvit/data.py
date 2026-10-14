"""PanNuke dataset + sampler for CellViT-style training (CellViT PanNuke recipe, albumentations 1.3).

Augmentations and probabilities are copied from the CellViT paper configs
(third_party/CellViT/logs_paper/PanNuke/CellViTHV/*/config.yaml). Sampling weights follow CellViT's
"cell+tissue" strategy with gamma 0.85, but the statistics are computed from the TRAINING fold only
(CellViT hard-codes counts over all folds).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
from albumentations.augmentations.crops import functional as crop_functional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..augment.copy_paste import CopyPasteConfig, NucleusBank, apply_copy_paste
from ..constants import DEAD_TYPE, NUM_CLASSES, TISSUES
from ..data.pannuke import PanNukeFold
from ..hovernet.data import hv_targets
from ..hovernet.targets import validate_hv_min_size
from .model import UNI_MEAN, UNI_STD

cv2.setNumThreads(0)


VISIBILITY_ARMS = ("A0", "A1", "A2", "A3")
VISIBILITY_COUNTS = ("eligible", "single_pixel", "single_axis", "zero_hv")


def original_border_ids(inst: np.ndarray) -> set[int]:
    """Original patch-border objects are already incomplete, even after a flip/rotation."""
    ids = np.unique(np.concatenate([inst[0], inst[-1], inst[:, 0], inst[:, -1]]))
    return set(map(int, ids)) - {0}


class VisibilityCrop(A.RandomSizedCrop):
    """Capture exact ID pixel loss before resize, without additional RNG draws.

    Only masks are inspected. The inherited image/mask transformations and parameters
    stay identical to pinned albumentations 1.3 RandomSizedCrop. ElasticTransform runs
    afterwards; its changes alone never label an object crop-truncated.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncated_ids = set()

    def apply_to_mask(self, mask, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        inst = mask[..., 0]
        cropped = crop_functional.random_crop(inst, crop_height, crop_width, h_start, w_start)
        full = np.bincount(inst.ravel())
        kept = np.bincount(cropped.ravel(), minlength=len(full))
        self.truncated_ids = set(map(int, np.flatnonzero((kept > 0) & (kept < full)))) - {0}
        return super().apply_to_mask(mask, crop_height=crop_height, crop_width=crop_width,
                                     h_start=h_start, w_start=w_start, **params)


def visibility_hv_targets(inst: np.ndarray, arm: str, truncated_ids: set[int]):
    """Selective intervention uses surviving IDs and FINAL working-pixel areas.

    Existing HV conventions, including legal degenerate coordinates, are preserved.
    Counts describe the eligible small objects, including those with all-zero HV.
    """
    if arm not in VISIBILITY_ARMS:
        raise ValueError("Unknown visibility arm")
    area = np.bincount(inst.ravel())
    ids = [i for i in truncated_ids if 0 < i < len(area) and 0 < area[i] < 30]
    counts = dict.fromkeys(VISIBILITY_COUNTS, 0)
    counts["eligible"] = len(ids)
    hv = hv_targets(inst, min_size=1 if arm == "A3" else 30)
    if ids:
        low = hv if arm == "A3" else hv_targets(inst, min_size=1)
        for i in ids:
            pixels = inst == i
            yy, xx = np.nonzero(pixels)
            counts["single_pixel"] += int(len(xx) == 1)
            counts["single_axis"] += int(np.ptp(xx) == 0 or np.ptp(yy) == 0)
            counts["zero_hv"] += int(not np.any(low[pixels]))
            if arm == "A2":
                hv[pixels] = low[pixels]
    return hv, counts


def cellvit_train_augs(size: int = 256, visibility_arm: str | None = None) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Downscale(p=0.15, scale_min=0.5, scale_max=0.5),
        A.Blur(p=0.2, blur_limit=10),
        A.GaussNoise(p=0.25, var_limit=50),
        A.ColorJitter(p=0.2, brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
        A.Superpixels(p=0.1, p_replace=0.1, n_segments=200, max_size=size // 2),
        A.ZoomBlur(p=0.1, max_factor=1.05),
        (VisibilityCrop if visibility_arm in ("A1", "A2", "A3") else A.RandomSizedCrop)(
            min_max_height=(size // 2, size), height=size, width=size,
            p=0.5 if visibility_arm in ("A1", "A2", "A3") else 0.1),
        A.ElasticTransform(p=0.2, sigma=25, alpha=0.5, alpha_affine=15),
    ])


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW with UNI (ImageNet) statistics."""
    x = (img.astype(np.float32) / 255.0 - np.array(UNI_MEAN, np.float32)) / np.array(UNI_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def small_nuclei(inst: np.ndarray, max_area: int) -> np.ndarray:
    """Pixels of instances with area < max_area (after augmentation)."""
    area = np.bincount(inst.ravel())
    area[0] = max_area  # background never small
    return area[inst] < max_area


def upsample_patch(img: np.ndarray, inst: np.ndarray, typ: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """Bilinear x2 (image) / nearest (labels) upsampling, applied AFTER augmentation and BEFORE
    normalisation/target construction. A fixed working-pixel HV cutoff consequently selects
    different native-area supports at different scales; use explicit hv_min_size controls
    to distinguish that supervision effect from the working resolution itself."""
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    inst = cv2.resize(inst, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    typ = cv2.resize(typ, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    return img, inst.astype(np.int32), typ.astype(np.int32)


def tissue_ids(names) -> np.ndarray:
    lut = {t: i for i, t in enumerate(TISSUES)}
    return np.array([lut[str(t)] for t in names], np.int64)


class PanNukeCellViT(Dataset):
    def __init__(self, folds: list[int], train: bool, small_area: int = 100,
                 copy_paste: CopyPasteConfig | None = None, synth: Path | None = None,
                 synth_frac: float = 0.0, upscale: int = 1, hv_min_size: int = 30,
                 dead_targets: bool = False, visibility_arm: str | None = None):
        if visibility_arm is not None and visibility_arm not in VISIBILITY_ARMS:
            raise ValueError("Unknown visibility arm")
        if visibility_arm is not None and (folds != ([1] if train else [2])):
            raise ValueError("visibility datasets require fold1 training / fold2 validation")
        self.visibility_arm = visibility_arm
        self.hv_min_size = validate_hv_min_size(hv_min_size)
        self.folds = [PanNukeFold(k) for k in folds]
        self.index = [("r", fi, j) for fi, f in enumerate(self.folds) for j in range(len(f))]
        self.train = train
        self.upscale = upscale
        self.dead_targets = dead_targets
        # pixel-area thresholds live at the working resolution (areas quadruple at 2x)
        self.small_area = small_area * upscale * upscale
        self.augs = cellvit_train_augs(visibility_arm=visibility_arm) if train else None
        self.copy_paste = copy_paste if (train and copy_paste is not None) else None
        self.bank = NucleusBank(self.folds) if self.copy_paste is not None else None
        self._tissue_ids = [tissue_ids(f.tissue) for f in self.folds]
        # optional synthetic samples appended after the real ones (pillar B): images.npy uint8,
        # inst.npy uint16, type.npy uint8, base.npy = index into the FIRST fold's tissue labels
        self.synth = None
        if train and synth is not None and synth_frac > 0:
            n_real = len(self.index)
            n_synth = min(int(synth_frac * n_real), len(np.load(synth / "base.npy", mmap_mode="r")))
            self.synth = {
                "dir": synth, "images": np.load(synth / "images.npy", mmap_mode="r"),
                "inst": np.load(synth / "inst.npy", mmap_mode="r"),
                "type": np.load(synth / "type.npy", mmap_mode="r"),
                "base": np.load(synth / "base.npy", mmap_mode="r"),
                "tissue": self._tissue_ids[0][np.load(synth / "base.npy")],
            }
            self.index += [("s", 0, j) for j in range(n_synth)]

    def __len__(self) -> int:
        return len(self.index)

    def tissue_labels(self) -> np.ndarray:
        out = [self._tissue_ids[fi] for fi in range(len(self.folds))]
        if self.synth is not None:
            out.append(self.synth["tissue"][:len(self) - sum(len(x) for x in out)])
        return np.concatenate(out)

    def class_presence(self) -> np.ndarray:
        """(N, 5) bool: does image contain at least one nucleus of class c."""
        out = []
        for f in self.folds:
            typ = f.type
            out.append(np.stack([[(typ[j] == c).any() for c in range(1, NUM_CLASSES + 1)] for j in range(len(f))]))
        if self.synth is not None:
            n = len(self) - sum(len(x) for x in out)
            typ = self.synth["type"][:n]
            out.append(np.stack([[(typ[j] == c).any() for c in range(1, NUM_CLASSES + 1)] for j in range(n)]))
        return np.concatenate(out)

    def __getitem__(self, i: int) -> dict:
        kind, fi, j = self.index[i]
        if kind == "s":
            s = self.synth
            img = np.array(s["images"][j])
            inst = np.array(s["inst"][j]).astype(np.int32)
            typ = np.array(s["type"][j]).astype(np.int32)
            tissue = int(s["tissue"][j])
        else:
            f = self.folds[fi]
            img = np.array(f.images[j])
            inst = np.array(f.inst[j]).astype(np.int32)
            typ = np.array(f.type[j]).astype(np.int32)
            tissue = int(self._tissue_ids[fi][j])
        truncated_ids = set()
        if self.train:
            incomplete_ids = original_border_ids(inst) if self.visibility_arm else set()
            if kind == "r" and self.copy_paste is not None:
                # BEFORE the geometric/photometric augmentations, so pasted nuclei are rotated,
                # blurred and colour-jittered together with the rest of the patch (pasting after
                # would let the network spot fakes by their un-augmented appearance)
                img, inst, typ = apply_copy_paste(
                    img, inst, typ, self.bank, self.copy_paste,
                    np.random.default_rng(np.random.randint(2**31)))
            crop = next((t for t in getattr(self.augs, "transforms", [])
                         if isinstance(t, VisibilityCrop)), None)
            if crop is not None:
                crop.truncated_ids = set()  # skipped crops cannot reuse the preceding sample
            r = self.augs(image=img, mask=np.stack([inst, typ], -1))
            if crop is not None:
                truncated_ids = crop.truncated_ids - incomplete_ids
            img, inst, typ = r["image"], r["mask"][..., 0], r["mask"][..., 1]
        if self.upscale > 1:
            img, inst, typ = upsample_patch(img, inst, typ, self.upscale)
        extra = {}
        if self.dead_targets:
            # computed AFTER augmentation/upsample: a crop that drops every Dead pixel must yield
            # dead_pos=False (the masked loss keys on it); the HV cutoff matches the main branch
            dead_inst = np.where(typ == DEAD_TYPE, inst, 0).astype(np.int32)
            extra = {
                "dead_np_map": torch.from_numpy((dead_inst > 0).astype(np.int64)),
                "dead_hv_map": torch.from_numpy(hv_targets(dead_inst, min_size=self.hv_min_size)),
                "dead_pos": torch.tensor(bool((dead_inst > 0).any())),
            }
        if self.visibility_arm is None:
            hv = hv_targets(inst, min_size=self.hv_min_size)
        else:
            # Validation loss always uses baseline HV30, including direct armed datasets.
            hv, counts = visibility_hv_targets(inst, self.visibility_arm if self.train else "A0", truncated_ids)
            extra["visibility_counts"] = torch.tensor([counts[k] for k in VISIBILITY_COUNTS], dtype=torch.int64)
        return {
            "img": torch.from_numpy(normalize(img)),                     # (3, 256, 256) float
            "np_map": torch.from_numpy((inst > 0).astype(np.int64)),     # (256, 256)
            "hv_map": torch.from_numpy(hv),  # working-pixel cutoff
            "tp_map": torch.from_numpy(typ.astype(np.int64)),            # (256, 256) 0..5
            "small_map": torch.from_numpy(small_nuclei(inst, self.small_area)),  # (256, 256) bool
            "tissue": tissue,
            "index": i,
            **extra,
        }


def cell_tissue_weights(ds: PanNukeCellViT, gamma: float = 0.85) -> torch.Tensor:
    """CellViT 'cell+tissue' sampling weights (get_sampling_weights_cell_tissue), train-fold statistics."""
    tissue = ds.tissue_labels()
    counts = np.bincount(tissue, minlength=len(TISSUES)).astype(np.float64)
    k = counts.sum()
    tw_tab = k / (gamma * np.maximum(counts, 1) + (1 - gamma) * k)
    tw = tw_tab[tissue]

    pres = ds.class_presence().astype(np.float64)
    bf = pres.sum(0)
    k = bf.sum()
    wv = k / (gamma * np.maximum(bf, 1) + (1 - gamma) * k)
    cw = (1 - gamma) * pres.max(-1) + gamma * (pres * wv).sum(-1)
    cw[cw == 0] = cw[cw > 0].min()
    return torch.from_numpy(tw / tw.max() + cw / cw.max())
