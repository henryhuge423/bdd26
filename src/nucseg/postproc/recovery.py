"""Training-free recovery of nuclei missed by the NP (foreground) branch (docs/findings.md 2026-09-25).

Variants of the official HoVer-Net post-processing (third_party/hover_net/models/hovernet/post_proc.py),
applied to saved network outputs:
  foreground  fg = (1 - beta) * P_NP(fg) + beta * (1 - P_TP(background)), then fg = max(fg, k * P_TP(Dead))
  threshold   blob = fg >= thr (official: 0.5)
  orphans     connected blobs that received no watershed marker (dropped by the official code) are
              kept as instances
Everything else (markers, watershed, per-instance majority type vote, contour sanity filter) is the
official code. With beta = 0, k = 0, thr = 0.5, orphans = False the output is identical to the official
post-processing (tests/test_recovery.py). This module does not import torch, saving address space.
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


def scaled_min_size(u: float) -> int:
    """Official AREA constant (remove_small_objects min_size=10 px^2 at the native 256 px), scaled to
    a u x working resolution by area (docs/findings.md 2026-09-29 pre-registered decode fix)."""
    return max(1, int(round(10 * u * u)))


def scaled_ksize(u: float) -> int:
    """Official Sobel LENGTH constant (ksize=21 px at the native 256 px), scaled by u, kept odd."""
    k = max(1, int(round(21 * u)))
    return k if k % 2 else k - 1


def scaled_marker_ksize(u: float) -> int:
    """Official marker-open kernel (cv2 ellipse (5, 5) px at the native 256 px), scaled by u with the
    same round-down-to-odd rule as `scaled_ksize` (findings 2026-09-30: residual x2 decode suspect —
    at u=2 the fixed 5x5 open relaxes to 2.5 native px and lets spurious markers through)."""
    k = max(3, int(round(5 * u)))
    return k if k % 2 else k - 1


def _rescaled_sobel_kernels(ksize: int):
    """Separable kernels of cv2.Sobel(ksize=21) resampled to an odd ksize > 31 (OpenCV hard-caps the
    Sobel aperture at 31, so the pre-registered u-scaled 41 is not directly expressible). The official
    operator's kernels are measured from its impulse response and linearly resampled, preserving its
    profile shape while scaling the support; the following cv2.normalize makes kernel scale moot."""
    d = np.zeros((63, 63))
    d[31, 31] = 1.0
    r = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=21)
    kx = r[31, 21:42] / np.abs(r[31, 21:42]).max()    # x profile through the impulse row (21 taps)
    ky = r[21:42, 33] / np.abs(r[21:42, 33]).max()    # y profile; +2 off center (x kernel is 0 there)
    kx = cv2.resize(kx.reshape(21, 1), (1, ksize), interpolation=cv2.INTER_LINEAR)
    ky = cv2.resize(ky.reshape(21, 1), (1, ksize), interpolation=cv2.INTER_LINEAR)
    return kx, ky


def _sobel_hv(h_dir: np.ndarray, v_dir: np.ndarray, ksize: int):
    """The official (dx=1, dy=0) Sobel pair at aperture ksize."""
    if ksize <= 31:
        return (cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize),
                cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize))
    kx, ky = _rescaled_sobel_kernels(ksize)
    return (cv2.sepFilter2D(h_dir, cv2.CV_64F, kx, ky),
            cv2.sepFilter2D(v_dir, cv2.CV_64F, ky, kx))


def proc_np_hv(pred: np.ndarray, thr: float = 0.5, orphans: bool = False, u: float = 1.0,
               marker_u: float | None = None) -> np.ndarray:
    """Official __proc_np_hv with a configurable blob threshold, optional orphan-blob recovery, and
    u-scaling of the px-unit decode constants (min_size 10 -> 10*u^2, Sobel ksize 21 -> odd(21*u);
    u=1 is bit-identical to the official function). marker_u scales the 5x5 marker-open kernel the
    same way (None = follow u). pred (H, W, 3): foreground probability, horizontal map, vertical map."""
    pred = np.array(pred, dtype=np.float32)
    blb_raw, h_dir_raw, v_dir_raw = pred[..., 0], pred[..., 1], pred[..., 2]

    blb = np.array(blb_raw >= thr, dtype=np.int32)
    blb = label(blb)[0]
    blb = remove_small_objects(blb, min_size=scaled_min_size(u))
    blb_lab = blb.copy()
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelh, sobelv = _sobel_hv(h_dir, v_dir, scaled_ksize(u))
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
    mu = u if marker_u is None else marker_u
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (scaled_marker_ksize(mu), scaled_marker_ksize(mu)))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = label(marker)[0]
    marker = remove_small_objects(marker, min_size=scaled_min_size(u))

    inst = watershed(dist, markers=marker, mask=blb)
    if orphans:
        covered = np.unique(blb_lab[inst > 0])
        orphan_ids = np.setdiff1d(np.unique(blb_lab), np.append(covered, 0))
        nxt = int(inst.max()) + 1
        for o in orphan_ids:
            inst[blb_lab == o] = nxt
            nxt += 1
    return inst


def decode_pred_map(pred_map: np.ndarray, nr_types: int = NR_TYPES, u: float = 1.0,
                    marker_u: float = 1.0):
    """Official `process` with the px-unit decode constants scaled by u (pre-registered x2 re-decode,
    docs/findings.md 2026-09-29: min_size 10 -> 10*u^2, Sobel ksize 21 -> odd(21*u)); marker_u scales
    the 5x5 marker-open kernel (findings 2026-09-30 residual suspect; default 1 = official kernel, so
    `--decode-u 2` keeps exactly the du2 semantics of the findings entry). u=1 + marker_u=1 is the
    unmodified official decode."""
    if u == 1.0 and marker_u == 1.0:
        return _official.process(pred_map, nr_types=nr_types)
    # official `process` looks the instance function up as a module global at call time
    _official.__dict__["__proc_np_hv"] = lambda p: proc_np_hv(p, u=u, marker_u=marker_u)
    try:
        return _official.process(pred_map, nr_types=nr_types)
    finally:
        _official.__dict__["__proc_np_hv"] = _ORIGINAL


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
