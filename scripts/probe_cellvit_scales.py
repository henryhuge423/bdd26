#!/usr/bin/env python
"""Indexed, validation-only train/inference-scale probe; predict and eval are separate processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import cached_property
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tissue_hash(tissue):
    return hashlib.sha256(json.dumps(np.asarray(tissue).astype(str).tolist()).encode()).hexdigest()


def validate_indices(indices, n):
    ids = np.asarray(indices)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
        raise ValueError("indices must be a nonempty integer vector")
    if np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids):
        raise ValueError("indices must be unique and in fold range")
    return ids.astype(np.int64)


def stratified_indices(tissue, count, seed):
    tissue = np.asarray(tissue)
    groups = np.unique(tissue)
    if count < len(groups) or count > len(tissue):
        raise ValueError("count must cover every tissue and not exceed the fold")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(tissue == t)).tolist() for t in groups]
    result = []
    while len(result) < count:
        for pool in pools:
            if pool and len(result) < count:
                result.append(pool.pop())
    return np.array(sorted(result), np.int64)


def make_manifest(path, tissue, count, seed, fold):
    ids = stratified_indices(tissue, count, seed)
    data = {"fold": fold, "indices": ids.tolist(), "seed": seed,
            "selection": "round-robin tissue strata; no class labels or predictions",
            "fold_size": len(tissue), "tissue_sha256": tissue_hash(tissue)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2)


def read_manifest(path, tissue, fold):
    d = json.loads(Path(path).read_text())
    if d["fold"] != fold or d["tissue_sha256"] != tissue_hash(tissue):
        raise ValueError("manifest fold or tissue ordering differs from the dataset")
    return validate_indices(d["indices"], len(tissue))


class IndexedFold:
    def __init__(self, fold, indices):
        self.fold = fold
        self.indices = validate_indices(indices, len(fold.tissue))

    def __len__(self):
        return len(self.indices)

    @cached_property
    def images(self):
        return np.asarray(self.fold.images[self.indices])

    @cached_property
    def inst(self):
        return np.asarray(self.fold.inst[self.indices])

    @cached_property
    def type(self):
        return np.asarray(self.fold.type[self.indices])

    @cached_property
    def tissue(self):
        return np.asarray(self.fold.tissue[self.indices])

    @cached_property
    def gt_channels(self):
        return np.asarray(self.fold.gt_channels[self.indices])


def validate_prediction_rows(saved, selected, n):
    if n != len(selected) or not np.array_equal(saved, selected):
        raise ValueError("prediction rows do not match the original image-index manifest")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--fold", type=int, required=True, choices=(1, 2, 3))
    p.add_argument("--indices", type=Path, required=True)
    p.add_argument("--make-indices", action="store_true")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--upscale", type=int, choices=(1, 2))
    p.add_argument("--decode-u", type=int, choices=(1, 2))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path)
    p.add_argument("--evaluate", action="store_true")
    a = p.parse_args()
    from nucseg.data.pannuke import PanNukeFold, split_folds
    f = PanNukeFold(a.fold)
    if a.make_indices:
        make_manifest(a.indices, f.tissue, a.count, a.seed, a.fold)
        print(f"manifest: {a.indices}", flush=True)
        return
    if a.out is None:
        p.error("--out is required")
    ids = read_manifest(a.indices, f.tissue, a.fold)
    sub = IndexedFold(f, ids)
    if a.evaluate:
        from nucseg.metrics.pannuke_eval import evaluate, save_report, format_summary
        meta = json.loads((a.out / "metadata.json").read_text())
        pred = a.out / "pred.npz"
        if meta["fold"] != a.fold or meta["indices_sha256"] != sha256(a.indices) or meta["prediction_sha256"] != sha256(pred):
            raise ValueError("prediction or index provenance differs from recorded metadata")
        with np.load(pred) as d:
            inst, typ = d["inst"], d["type"]
            validate_prediction_rows(d["image_indices"], ids, len(inst))
        out = a.out / "eval"
        out.mkdir(exist_ok=False)
        result = evaluate(sub.gt_channels, sub.inst, sub.type, sub.tissue, inst, typ, workers=a.workers)
        save_report(result, out)
        gt_files = ("inst.npy", "type.npy", "tissue.npy", "gt_channels.npz")
        evidence = {"dataset_sha256": {n: sha256(f.dir / n) for n in gt_files},
                    "pred_sha256": sha256(pred), "indices_sha256": sha256(a.indices),
                    "canonical_evaluator_sha256": sha256(ROOT / "src/nucseg/metrics/pannuke_eval.py")}
        (out / "provenance.json").write_text(json.dumps(evidence, indent=2))
        print(format_summary(result["summary"]), flush=True)
        return
    if a.run is None or a.upscale is None or a.decode_u != a.upscale:
        p.error("prediction requires --run and equal explicit --upscale/--decode-u")
    if not 1 <= a.batch_size <= 2 or a.workers < 1:
        p.error("probe batch-size must be 1 or 2; workers must be positive")
    config = json.loads((a.run / "config.json").read_text())
    if split_folds(config["split"])[1] != a.fold:
        p.error("this probe accepts only the run's validation fold")
    if a.out.exists():
        raise FileExistsError(a.out)
    # No CUDA/model import until all manifest/output checks have passed.
    import torch
    from nucseg.cellvit.engine import build_model, predict_fold
    ckpt = a.run / "final.pth"
    ckpt_hash = sha256(ckpt)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    model = build_model(pretrained=False).cuda()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    res = predict_fold(model, sub, batch_size=a.batch_size, workers=a.workers,
                       tta=False, inst_probs=False, upscale=a.upscale, decode_u=a.decode_u)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / "pred.npz").open("xb") as out:
        np.savez_compressed(out, inst=res[0], type=res[1], tissue_prob=res[2], image_indices=ids)
    source = ["scripts/probe_cellvit_scales.py", "src/nucseg/cellvit/engine.py",
              "src/nucseg/cellvit/model.py", "src/nucseg/postproc/recovery.py"]
    meta = {"fold": a.fold, "split": config["split"], "seed": config["seed"],
            "train_upscale": config.get("upscale", 1), "infer_upscale": a.upscale,
            "decode_u": a.decode_u, "marker_u": 1, "tta": False, "batch_size": a.batch_size,
            "n_images": len(ids), "role": "validation mechanism probe, not benchmark test",
            "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_hash,
            "config_sha256": sha256(a.run / "config.json"),
            "indices_sha256": sha256(a.indices), "prediction_sha256": sha256(a.out / "pred.npz"),
            "source_sha256": {n: sha256(ROOT / n) for n in source},
            "image_file_sha256": sha256(f.dir / "images.npy"),
            "host": socket.gethostname(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds_model_load_and_inference": elapsed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out}: {len(ids)} images in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
