"""Convert official PanNuke fold zips into a compact on-disk format.

The official release stores everything as float64 (~37 GB unzipped). We stream arrays
directly out of the zip (never extracting them) and write, per fold:

    fold{k}/images.npy      uint8  (N, 256, 256, 3)
    fold{k}/inst.npy        uint16 (N, 256, 256)   instance ids, 0 = background
    fold{k}/type.npy        uint8  (N, 256, 256)   0 = background, 1..5 = CLASS_NAMES
    fold{k}/tissue.npy      <U     (N,)            tissue name per patch
    fold{k}/gt_channels.npz uint16 (N, 256, 256, 5) original per-class instance channels
                                                   (compressed; used for exact official eval)
    fold{k}/meta.json       counts and conversion sanity statistics

The merged instance map follows the official `binarize` rule: channels are visited in
order and later instances overwrite earlier ones, so bPQ computed on `inst` matches the
official script exactly.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..constants import NUM_CLASSES, PATCH_SIZE


def _open_npy_in_zip(zf: zipfile.ZipFile, member: str):
    f = zf.open(member)
    version = np.lib.format.read_magic(f)
    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    assert not fortran, f"{member}: fortran order not supported"
    return f, shape, dtype


def _iter_npy_chunks(zf: zipfile.ZipFile, member: str, chunk: int = 64):
    """Yield (start, array_chunk) along axis 0 without materialising the full array."""
    f, shape, dtype = _open_npy_in_zip(zf, member)
    per_item = int(np.prod(shape[1:])) * dtype.itemsize
    n = shape[0]
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        buf = f.read(per_item * m)
        assert len(buf) == per_item * m, f"{member}: truncated read"
        yield start, np.frombuffer(buf, dtype=dtype).reshape((m,) + tuple(shape[1:]))
    f.close()


def _find(zf: zipfile.ZipFile, suffix: str) -> str:
    hits = [n for n in zf.namelist() if n.endswith(suffix)]
    assert len(hits) == 1, f"expected one member ending with {suffix}, got {hits}"
    return hits[0]


def merge_channels(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(H, W, 5) per-class instance channels -> (inst uint16, type uint8, n_overlap_px).

    Mirrors PanNuke-metrics `binarize`: sequential ids across channels, later overwrite.
    """
    inst = np.zeros(ch.shape[:2], np.int32)
    typ = np.zeros(ch.shape[:2], np.uint8)
    occupied = np.zeros(ch.shape[:2], np.int32)
    count = 1
    for c in range(ch.shape[2]):
        x = ch[..., c]
        for v in np.unique(x):
            if v == 0:
                continue
            m = x == v
            occupied += m
            inst[m] = count
            typ[m] = c + 1
            count += 1
    n_overlap = int((occupied > 1).sum())
    # Instances can be fully overwritten; relabel contiguously.
    ids = np.unique(inst)
    ids = ids[ids > 0]
    lut = np.zeros(count, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    inst = lut[inst]
    assert inst.max() < 2**16
    return inst.astype(np.uint16), typ, n_overlap


def convert_fold(zip_path: Path, out_dir: Path, fold: int, chunk: int = 64) -> dict:
    out = out_dir / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        img_m = _find(zf, "images.npy")
        msk_m = _find(zf, "masks.npy")
        typ_m = _find(zf, "types.npy")
        with zf.open(typ_m) as f:
            tissue = np.load(f, allow_pickle=True).astype(str)
        n = len(tissue)

        images = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (n, PATCH_SIZE, PATCH_SIZE, 3))
        max_frac = 0.0
        for s, a in tqdm(_iter_npy_chunks(zf, img_m, chunk), total=-(-n // chunk), desc=f"fold{fold} images"):
            max_frac = max(max_frac, float(np.abs(a - np.round(a)).max()))
            images[s:s + len(a)] = np.clip(np.round(a), 0, 255).astype(np.uint8)
        images.flush()

        inst = np.lib.format.open_memmap(out / "inst.npy", "w+", np.uint16, (n, PATCH_SIZE, PATCH_SIZE))
        typ = np.lib.format.open_memmap(out / "type.npy", "w+", np.uint8, (n, PATCH_SIZE, PATCH_SIZE))
        gt_ch = np.zeros((n, PATCH_SIZE, PATCH_SIZE, NUM_CLASSES), np.uint16)
        n_overlap_px = 0
        n_overlap_img = 0
        class_counts = np.zeros(NUM_CLASSES, np.int64)
        for s, a in tqdm(_iter_npy_chunks(zf, msk_m, chunk), total=-(-n // chunk), desc=f"fold{fold} masks"):
            a = a[..., :NUM_CLASSES]
            assert a.max() < 2**16 and np.all(a == np.round(a))
            a = a.astype(np.uint16)
            gt_ch[s:s + len(a)] = a
            for i in range(len(a)):
                inst[s + i], typ[s + i], ov = merge_channels(a[i])
                n_overlap_px += ov
                n_overlap_img += ov > 0
                for c in range(NUM_CLASSES):
                    class_counts[c] += len(np.unique(a[i, ..., c])) - 1
        inst.flush()
        typ.flush()

    np.save(out / "tissue.npy", tissue)
    np.savez_compressed(out / "gt_channels.npz", gt=gt_ch)
    meta = {
        "fold": fold,
        "n_patches": int(n),
        "image_max_nonint_residual": max_frac,
        "overlap_pixels": int(n_overlap_px),
        "images_with_overlap": int(n_overlap_img),
        "nuclei_per_class": {k: int(v) for k, v in zip(
            ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"], class_counts)},
        "tissue_counts": {t: int(c) for t, c in zip(*np.unique(tissue, return_counts=True))},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
