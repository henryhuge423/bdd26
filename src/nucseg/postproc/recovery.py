"""Training-free recovery of nuclei missed by the NP (foreground) branch (docs/findings.md 2026-09-25).

Variants of the official HoVer-Net post-processing (third_party/hover_net/models/hovernet/post_proc.py),
applied to saved network outputs:
  foreground  fg = (1 - beta) * P_NP(fg) + beta * (1 - P_TP(background)), then fg = max(fg, k * P_TP(Dead))
  threshold   blob = fg >= thr (official: 0.5)
  orphans     connected blobs that received no watershed marker (dropped by the official code) are
              kept as instances
Everything else (markers, watershed, per-instance majority type vote, contour sanity filter) is the
official code. With beta = 0, k = 0, thr = 0.5, orphans = False the output is identical to the official
post-processing (tests/test_recovery.py). This module does not import torch (ugrad address-space limit).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.segmentation import watershed

HOVER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "hover_net"
if str(HOVER_ROOT) not in sys.path:
    sys.path.insert(0, str(HOVER_ROOT))
from misc.utils import remove_small_objects  # noqa: E402
from models.hovernet import post_proc as _official  # noqa: E402

DEAD = 4  # type id of Dead (0 = background)
NR_TYPES = 6


@dataclass(frozen=True)
class Recovery:
    beta: float = 0.0
    k_dead: float = 0.0
    thr: float = 0.5
    orphans: bool = False

    @property
    def is_official(self) -> bool:
        return self == Recovery()

    def name(self) -> str:
        return f"b{self.beta:g}_k{self.k_dead:g}_t{self.thr:g}_o{int(self.orphans)}"

    def as_dict(self) -> dict:
        return asdict(self)


def foreground(np_fg: np.ndarray, tp_prob: np.ndarray, cfg: Recovery) -> np.ndarray:
    """np_fg (H, W) NP foreground probability, tp_prob (H, W, 6) type probabilities."""
    fg = np_fg.astype(np.float32)
    if cfg.beta:
        fg = (1 - cfg.beta) * fg + cfg.beta * (1 - tp_prob[..., 0].astype(np.float32))
    if cfg.k_dead:
        fg = np.maximum(fg, cfg.k_dead * tp_prob[..., DEAD].astype(np.float32))
    return fg


def proc_np_hv(pred: np.ndarray, thr: float = 0.5, orphans: bool = False) -> np.ndarray:
    """Official __proc_np_hv with a configurable blob threshold and optional orphan-blob recovery.
    pred (H, W, 3): foreground probability, horizontal map, vertical map."""
    pred = np.array(pred, dtype=np.float32)
    blb_raw, h_dir_raw, v_dir_raw = pred[..., 0], pred[..., 1], pred[..., 2]

    blb = np.array(blb_raw >= thr, dtype=np.int32)
    blb = label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb_lab = blb.copy()
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    inst = watershed(dist, markers=marker, mask=blb)
    if orphans:
        covered = np.unique(blb_lab[inst > 0])
        orphan_ids = np.setdiff1d(np.unique(blb_lab), np.append(covered, 0))
        nxt = int(inst.max()) + 1
        for o in orphan_ids:
            inst[blb_lab == o] = nxt
            nxt += 1
    return inst


def postprocess(np_fg: np.ndarray, hv: np.ndarray, tp_prob: np.ndarray, cfg: Recovery = Recovery()):
    """Saved network outputs of one patch -> (inst (H, W) int32, type (H, W) uint8), as
    nucseg.hovernet.engine._post but with the recovery variant `cfg`."""
    fg = foreground(np_fg, tp_prob, cfg)
    tp = tp_prob.argmax(-1)[..., None].astype(np.float32)
    pred_map = np.concatenate([tp, fg[..., None], hv.astype(np.float32)], -1)
    # official `process` looks the instance function up as a module global at call time
    _official.__dict__["__proc_np_hv"] = lambda p: proc_np_hv(p, cfg.thr, cfg.orphans)
    try:
        inst, info = _official.process(pred_map, nr_types=NR_TYPES)
    finally:
        _official.__dict__["__proc_np_hv"] = _ORIGINAL
    typ = np.zeros(inst.shape, np.uint8)
    keep = np.zeros(inst.shape, bool)
    for iid, d in info.items():
        m = inst == iid
        keep |= m
        typ[m] = d["type"] if d["type"] is not None else 0
    return np.where(keep, inst, 0).astype(np.int32), typ


_ORIGINAL = _official.__dict__["__proc_np_hv"]
