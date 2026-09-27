"""HoVer-NeXt-T inference on PanNuke-format patch datasets (pillar C third evaluator).

Wraps the official Zenodo per-fold weights (digitalpathologybern/hover_next_inference layout:
``<weights>/params.toml``, ``<weights>/train/best_model``, ``<weights>/pannuke_test_param_dict.json``).
The model is the vendored :mod:`nucseg.hovernext.model`; the post-processing is vendored from
``third_party/hover_next_inference/src/post_process_utils.py`` with the zarr stores replaced by
plain numpy (we process one 256x256 patch at a time, so there is nothing to stitch).

Fold semantics verified against hover_next_train ``PANNUKE_FOLDS = [[1,2],[0,2],[1,0]]`` with
``fold = params["fold"] - 1``: weights ``pannuke_convnextv2_tiny_{1,2}`` test fold 3 (our
splits 1/2), ``_3`` tests fold 1 (our split 3).
"""

from __future__ import annotations

import json
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10 (nuclei env) has tomli instead
    import tomli as tomllib
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import find_objects
from skimage.segmentation import watershed

from .model import get_model

# constants.py of hover_next_inference (PanNuke profile)
MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]
MAX_HOLE_SIZE = 128


@dataclass
class HNModel:
    model: torch.nn.Module
    fg: np.ndarray  # per-class foreground thresholds (pannuke_test_param_dict.json)
    seed: np.ndarray  # per-class seed thresholds


def build_model(weights: Path | str) -> HNModel:
    weights = Path(weights)
    params = tomllib.loads((weights / "params.toml").read_text())
    model = get_model(enc=params["encoder"], out_channels_cls=params["out_channels_cls"],
                      out_channels_inst=params["inst_channels"], pretrained=False)
    cp = torch.load(weights / "train" / "best_model", map_location="cpu", weights_only=False)
    state = cp["model_state_dict"]
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()})
    model.eval()
    dt = json.loads((weights / "pannuke_test_param_dict.json").read_text())
    return HNModel(model, np.asarray(dt["best_fg_pannuke"]), np.asarray(dt["best_seed_pannuke"]))


# ---------------- vendored post-processing (numpy port) ----------------
# Semantics fixed by hover_next_inference `proc_tile`: the cls map enters post-processing as a
# per-pixel one-hot bool (argmax over the 5 non-background channels), and only the first two
# inst-softmax channels (background, foreground) are used — the boundary channel is dropped.
# `make_ct` is therefore a majority VOTE over pixels, not a softmax average.

def _remove_small_holes_cv2(img, sz):
    img = np.logical_not(img).astype(np.uint8)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img)
    sizes = stats[1:, -1]
    im_result = np.zeros(img.shape, dtype=np.uint16)
    for blob in range(nb_blobs - 1):
        if sizes[blob] >= sz:
            im_result[im_with_separated_blobs == blob + 1] = 1
    return np.logical_not(im_result.astype(bool))


def _faster_instance_seg(out_img, out_cls, best_fg, best_seed):
    """out_img (2,H,W) bg/fg softmax; out_cls (5,H,W) bool one-hot class map."""
    _, rois = cv2.connectedComponents((out_img[0] > 0).astype(np.uint8), connectivity=8)
    bboxes = find_objects(rois)
    labelling = np.zeros(out_cls.shape[1:], np.int32)
    if len(bboxes) == 0:
        return labelling, True
    max_inst = 0
    for bb in bboxes:
        bg_pred = out_img[(slice(0, 1, None), *bb)].squeeze()
        if ((np.array(bg_pred.shape[-2:]) <= 2).any() or (np.array(bg_pred.shape).sum() <= 64)
                or (len(bg_pred.shape) < 2)):
            continue
        fg_pred = out_img[(slice(1, 2, None), *bb)].squeeze()
        sem = out_cls[(slice(0, len(best_fg), None), *bb)]
        ws_surface = 1.0 - fg_pred
        fg = np.zeros_like(ws_surface, dtype=bool)
        seeds = np.zeros_like(ws_surface, dtype=bool)
        for cl, fg_t in enumerate(best_fg):
            mask = sem[cl]
            fg[mask] |= (1.0 - bg_pred[mask]) > fg_t
            seeds[mask] |= fg_pred[mask] > best_seed[cl]
        _, markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
        bb_ws = watershed(ws_surface, markers, mask=fg, connectivity=2)
        bb_ws[bb_ws != 0] += max_inst
        labelling[bb] = bb_ws
        max_inst = np.max(bb_ws)
    return labelling, False


def _post_proc_inst(pred_inst, hole_size):
    pshp = pred_inst.shape
    out = np.zeros(pshp, dtype=np.int32)
    i = 1
    for j, sl in enumerate(find_objects(pred_inst)):
        if not sl:
            continue
        rm_small_hole = _remove_small_holes_cv2(pred_inst[sl] == (j + 1), hole_size)
        out[sl][rm_small_hole > 0] = i
        i += 1
    out_ = np.zeros(out.shape, dtype=np.int32)
    i_ = 1
    for j, sl in enumerate(find_objects(out)):
        if not sl:
            continue
        nr_objects, relabeled = cv2.connectedComponents((out[sl] == (j + 1)).astype(np.uint8),
                                                        connectivity=8)
        for new_lab in range(1, nr_objects):
            out_[sl] += (relabeled == new_lab) * i_
            i_ += 1
    return out_


def _make_ct(pred_class, instance_map):
    """Majority-vote class per instance over the one-hot map -> {id: class 1..5}."""
    pred_class = np.rollaxis(np.asarray(pred_class), 0, 3)  # (5,H,W) -> (H,W,5)
    out = {0: 0}
    for i, sl in enumerate(find_objects(instance_map)):
        if not sl:
            continue
        inst = instance_map[sl] == (i + 1)
        out[i + 1] = int(np.sum(pred_class[sl][inst], axis=0).argmax()) + 1
    return {k: v for k, v in out.items() if k and v}


def _remove_obj_cls(pred_inst, pred_ct, min_threshs, max_threshs):
    out_oi = np.zeros_like(pred_inst, dtype=np.int32)
    i_ = 1
    keep = {}
    for i, sl in enumerate(find_objects(pred_inst)):
        if not sl:
            continue
        px = np.sum(pred_inst[sl] == (i + 1))
        cls_ = pred_ct.get(i + 1, 1)
        if min_threshs[cls_ - 1] < px < max_threshs[cls_ - 1]:
            keep[i_] = cls_
            out_oi[sl][pred_inst[sl] == (i + 1)] = i_
            i_ += 1
    return out_oi, keep


def _post(args):
    inst3, cls6, fg, seed = args
    # proc_tile equivalents: keep bg/fg softmax, one-hot the classes without background
    out_img = inst3[:2]
    sem = np.zeros((5, *cls6.shape[1:]), bool)
    am = cls6[1:].argmax(0)  # cls channel 0 is background
    for c in range(5):
        sem[c] = am == c
    lab, skip = _faster_instance_seg(out_img, sem, fg, seed)
    if skip:
        return (np.zeros(inst3.shape[1:], np.int32), np.zeros(inst3.shape[1:], np.uint8))
    lab = _post_proc_inst(lab, MAX_HOLE_SIZE)
    ct = _make_ct(sem, lab)
    inst, ct = _remove_obj_cls(lab, ct, MIN_THRESHS_PANNUKE, MAX_THRESHS_PANNUKE)
    typ = np.zeros(inst.shape, np.uint8)
    for k, v in ct.items():
        typ[inst == k] = v
    return inst, typ


def predict_fold(m: HNModel, fold_ds, device="cuda", batch_size=16, workers=8, tta=False):
    """Returns (inst, type) for every patch of a fold, in fold order (no TTA — parity with the
    paper's plain inference and with our no-TTA HoVer-Net external rows)."""
    if tta:
        raise NotImplementedError("HoVer-NeXt TTA uses their stochastic view sampling; skipped")
    m.model = m.model.to(device)
    insts, types = [], []
    with Pool(workers) as pool, torch.inference_mode():
        for s in range(0, len(fold_ds), batch_size):
            x = torch.from_numpy(np.asarray(fold_ds.images[s:s + batch_size], np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2).to(device)
            with torch.autocast("cuda", torch.float16):
                out = m.model(x)
            inst3 = out[:, 2:5].softmax(1).float().cpu().numpy()
            cls6 = out[:, 5:].softmax(1).float().cpu().numpy()
            res = pool.map(_post, [(inst3[i], cls6[i], m.fg, m.seed) for i in range(len(x))],
                           chunksize=2)
            insts += [r[0] for r in res]
            types += [r[1] for r in res]
    return np.stack(insts), np.stack(types)
