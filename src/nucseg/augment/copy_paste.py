"""Failure-driven copy-paste augmentation (pillar B control arm CP1).

Layout follows the 2026-09-26 failure-mining findings: missed Dead nuclei are ISOLATED small
objects (mostly 60-200 px, zero touching neighbours) sitting in stroma; cross-class touching is
rare in PanNuke and not the failure mode. CP1 therefore pastes REAL nucleus crops from the same
training fold into nucleus-free stroma with a clearance margin (never touching existing nuclei),
class-weighted towards Dead but including every class, so the model cannot learn an
"isolated small nucleus == Dead" shortcut.

Runs BEFORE the geometric/photometric augmentations (see PanNukeCellViT), so pasted nuclei are
rotated / blurred / colour-jittered together with the patch and cannot be spotted by their
un-augmented appearance; all downstream targets (np / hv / tp / small_map) are then computed from
the augmented label maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEAD = 4  # type-map value of the Dead class (1-based, 0 = background)


@dataclass
class CopyPasteConfig:
    prob: float = 0.5  # fraction of training patches that receive insertions
    lam: float = 3.0  # Poisson mean number of insertions per augmented patch
    dead_w: float = 0.4  # donor weight for Dead; the rest is split over the other classes
    area: tuple[int, int] = (50, 400)  # donor nucleus area range (px)
    clearance: int = 8  # min distance (px) between a pasted nucleus and any other nucleus
    border: int = 4  # keep pasted nuclei this far from the patch border
    max_tries: int = 40  # placement attempts per insertion


class NucleusBank:
    """Catalogue of donor nuclei across folds; crops are fetched lazily from the memmaps.

    One row per instance: (fold_i, img_j, inst_id, cls, area, y0, y1, x0, x1)."""

    def __init__(self, folds, edge_margin: int = 2):
        self.folds = folds
        fi_, im_, id_, cl_, ar_, y0_, y1_, x0_, x1_, edge_, touch_ = [], [], [], [], [], [], [], [], [], [], []
        kern3 = np.ones((3, 3), np.uint8)
        for fi, f in enumerate(folds):
            H, W = f.inst[0].shape
            for j in range(len(f)):
                inst = np.asarray(f.inst[j])
                n = int(inst.max())
                if n == 0:
                    continue
                # per-instance majority type via a joint histogram (inst * 8 + type)
                typ = np.asarray(f.type[j])
                hist = np.bincount((inst.ravel().astype(np.int64) * 8 + typ.ravel()).clip(0, None),
                                   minlength=(n + 1) * 8).reshape(n + 1, 8)
                areas = np.bincount(inst.ravel(), minlength=n + 1)
                boxes = _find_objects(inst, n)
                for k in range(1, n + 1):
                    (y0, y1), (x0, x1) = boxes[k]
                    fi_.append(fi)
                    im_.append(j)
                    id_.append(k)
                    cl_.append(int(hist[k].argmax()))
                    ar_.append(int(areas[k]))
                    y0_.append(int(y0))
                    y1_.append(int(y1))
                    x0_.append(int(x0))
                    x1_.append(int(x1))
                    # donors clipped at the patch border would be pasted as fragments with
                    # straight cut edges -> unusable (caught in the 2026-09-26 montage review)
                    edge_.append(y0 < edge_margin or x0 < edge_margin
                                 or y1 > H - edge_margin or x1 > W - edge_margin)
                    # donors that touch a neighbour in the SOURCE patch carry a straight shared
                    # boundary (chord) in their mask -> also unusable
                    win = inst[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1]
                    m = win == k
                    dil = cv2.dilate(m.astype(np.uint8), kern3).astype(bool)
                    touch_.append(bool((dil & (win > 0) & ~m).any()))
        self.rows = np.rec.fromarrays(
            [np.array(x) for x in (fi_, im_, id_, cl_, ar_, y0_, y1_, x0_, x1_, edge_, touch_)],
            names="fi,img,id,cls,area,y0,y1,x0,x1,edge,touch")
        self.usable = ~self.rows.edge & ~self.rows.touch
        self.by_class = [np.flatnonzero((self.rows.cls == c) & self.usable) for c in range(8)]

    def sample(self, cfg: CopyPasteConfig, rng: np.random.Generator) -> int:
        """Row index of a donor, class-weighted and area-filtered (falls back to any class)."""
        weights = np.full(8, (1.0 - cfg.dead_w) / 4.0)
        weights[DEAD] = cfg.dead_w
        weights[0] = 0.0
        for _ in range(8):  # retry a few times if the drawn class has no in-range donors
            c = int(rng.choice(8, p=weights / weights.sum()))
            ok = self.by_class[c][
                (self.rows.area[self.by_class[c]] >= cfg.area[0])
                & (self.rows.area[self.by_class[c]] <= cfg.area[1])]
            if len(ok):
                return int(rng.choice(ok))
        ok = np.flatnonzero(self.usable &
                            (self.rows.area >= cfg.area[0]) & (self.rows.area <= cfg.area[1]))
        return int(rng.choice(ok if len(ok) else np.flatnonzero(self.usable)))


def _find_objects(inst: np.ndarray, n: int):
    """Bounding boxes for labels 1..n; pure-numpy (avoids a scipy dependency here)."""
    boxes = [(slice(0, 0), slice(0, 0))] * (n + 1)
    ys, xs = np.nonzero(inst)
    order = np.argsort(inst[ys, xs], kind="stable")
    ys, xs, labs = ys[order], xs[order], inst[ys, xs][order]
    bounds = np.searchsorted(labs, np.arange(1, n + 2))
    for k in range(1, n + 1):
        a, b = bounds[k - 1], bounds[k]
        if a == b:
            continue
        boxes[k] = ((int(ys[a:b].min()), int(ys[a:b].max()) + 1),
                    (int(xs[a:b].min()), int(xs[a:b].max()) + 1))
    return boxes


def apply_copy_paste(img: np.ndarray, inst: np.ndarray, typ: np.ndarray,
                     bank: NucleusBank, cfg: CopyPasteConfig, rng: np.random.Generator):
    """Insert `max(1, Poisson(lam))` donor nuclei into empty stroma; returns new (img, inst, typ)."""
    if rng.random() >= cfg.prob:
        return img, inst, typ
    img = img.copy()
    inst = inst.copy()
    typ = typ.copy()
    H, W = inst.shape
    occupied = inst > 0
    blocked = cv2.dilate(occupied.astype(np.uint8),
                         np.ones((2 * cfg.clearance + 1, 2 * cfg.clearance + 1), np.uint8))
    # destination stroma colour (for a per-channel donor shift) and a darkness map that catches
    # UNANNOTATED nuclei (PanNuke annotation gaps): pasting over them would label real nuclei bg
    stroma = np.median(img[~blocked.astype(bool)].reshape(-1, 3), axis=0) if (blocked == 0).any() \
        else np.array([180.0, 180.0, 180.0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray < otsu_thr) & ~blocked.astype(bool)
    free = np.argwhere(~blocked.astype(bool))
    if not len(free):
        return img, inst, typ
    rng.shuffle(free)
    n_ins = max(1, int(rng.poisson(cfg.lam)))
    next_id = int(inst.max()) + 1
    placed = 0
    pi = 0
    for _ in range(n_ins):
        r = bank.rows[bank.sample(cfg, rng)]
        f = bank.folds[r.fi]
        src_inst = np.asarray(f.inst[r.img])
        mask = src_inst == r.id
        y0, y1, x0, x1 = r.y0, r.y1, r.x0, r.x1
        crop = np.asarray(f.images[r.img])[y0:y1, x0:x1].astype(np.float32)
        cmask = mask[y0:y1, x0:x1]
        h, w = cmask.shape
        # shift the donor so its own stroma matches the destination stroma (keeps nucleus
        # contrast, removes gross hue mismatch flagged in the montage review)
        donor_out = crop[~cmask]
        if len(donor_out) > 30:
            crop = np.clip(crop + np.clip(stroma - np.median(donor_out, axis=0), -40, 40), 0, 255)
        for try_ in range(cfg.max_tries):
            if pi >= len(free):
                pi = 0
            cy, cx = free[pi]
            pi += 1
            ty0, tx0 = int(cy) - h // 2, int(cx) - w // 2
            if ty0 < cfg.border or tx0 < cfg.border or ty0 + h > H - cfg.border or tx0 + w > W - cfg.border:
                continue
            window = blocked[ty0:ty0 + h, tx0:tx0 + w]
            if (cmask & window.astype(bool)).any():
                continue
            dark_win = dark[ty0:ty0 + h, tx0:tx0 + w]
            if (cmask & dark_win).sum() > 0.3 * cmask.sum():
                continue  # mostly covering unannotated nuclear material
            alpha = cv2.GaussianBlur(cmask.astype(np.float32), (0, 0), 1.0)[..., None]
            win = img[ty0:ty0 + h, tx0:tx0 + w].astype(np.float32)
            img[ty0:ty0 + h, tx0:tx0 + w] = (win * (1 - alpha) + crop.astype(np.float32) * alpha).astype(np.uint8)
            region = np.zeros((H, W), bool)
            region[ty0:ty0 + h, tx0:tx0 + w] = cmask
            inst[region] = next_id
            typ[region] = r.cls
            occupied |= region
            # keep future insertions a full clearance away from the pasted nucleus as well
            blocked = cv2.dilate(occupied.astype(np.uint8),
                                 np.ones((2 * cfg.clearance + 1, 2 * cfg.clearance + 1), np.uint8))
            next_id += 1
            placed += 1
            break
    return img, inst, typ
