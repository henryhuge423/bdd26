#!/usr/bin/env python
"""Reduced E0 (2026-10-08): inference cost of the EF-P2 two-model system vs single-model x1.

Fixed protocol (RESEARCH_NEXT E0 row, reduced): one device, fp32, batch 32, decode workers 16,
chunked wall-clock timing of the UNMODIFIED predict_fold over split 1's validation fold
(fold 2 — no test fold is read). Arms: x1 (256), x2 du2 (512 pipeline: upscale + decode_u 2),
x1+TTA (8 dihedral forwards) as the same-budget single-model reference. The CPU side times the
frozen split2_seed19 EF-P2 application (P2 a200_p0.9_interior0 + EF a30) per image on the
matched-seed round's cached validation predictions; the replayed added-count must reproduce the
frozen selection's 1542 additions or the run aborts (deployment-path check). M/S lever timing is
NOT included (CPU relabel of x1 predictions, same order as fuse; noted in the output).

    CUDA_VISIBLE_DEVICES=<gpu> python -B scripts/inference_cost.py \
        --x1-run runs/cellvit_uni/split1 --x2-run runs/cellvit_uni_x2/split1 --fold 2 \
        --ef-selection runs/analysis/existence_ef_20261005/p2ef/split2_seed19/selection.json \
        --p2-selection runs/analysis/matched_seed_20261004/p2/split2_seed19/selection.json \
        --out runs/analysis/inference_cost_20261008/cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nucseg.cellvit.engine import TrainConfig, build_model, predict_fold, run_upscale  # noqa: E402
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.postproc.existence import existence_pass  # noqa: E402
from nucseg.postproc.scale_fusion import candidate_info, fuse, probability_rows  # noqa: E402
from run_scale_fusion import image_confidence  # noqa: E402

CHUNK = 256


def chunk_bounds(n: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def per_image_summary(chunk_seconds) -> dict:
    """Per-image latency stats over chunk wall times (per-image value = chunk_time/n_chunk)."""
    per_img_ms = [1000.0 * t / n for t, n in chunk_seconds]
    total_t, total_n = sum(t for t, _ in chunk_seconds), sum(n for _, n in chunk_seconds)
    if not per_img_ms:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "img_per_s": None, "n": 0}
    return {"mean_ms": float(np.mean(per_img_ms)),
            "p50_ms": float(np.percentile(per_img_ms, 50)),
            "p95_ms": float(np.percentile(per_img_ms, 95)),
            "img_per_s": float(total_n / total_t), "n": int(total_n)}


def frozen_ef_config(p2_selection: dict, ef_selection: dict) -> dict:
    """Frozen deployment parameters: P2 gate from its selection, EF floor from the EF selection."""
    p2 = p2_selection["selected"]
    p2cfg = None if p2["name"] == "identity" else {
        "max_area": p2["max_area"], "min_prob": p2["min_prob"], "interior_only": p2["interior_only"]}
    ef = ef_selection["selected"]
    return {"p2": p2cfg,
            "ef": {"min_area": ef.get("min_area", 0), "dead_exempt": ef.get("dead_exempt", False)}}


def ef_apply(bi, bt, xi, xt, rows, p2cfg, ef):
    """One image of the frozen EF-P2 deployment path (candidate scan -> EF floor -> P2 fuse)."""
    conf = image_confidence(xi, xt, rows)
    cands = candidate_info(bi, xi, xt)
    allowed = {c["id"] for c in cands if existence_pass(c, ef["min_area"], ef["dead_exempt"])}
    return fuse(bi, bt, xi, xt, conf, p2cfg["max_area"], p2cfg["min_prob"],
                p2cfg["interior_only"], allowed_ids=allowed)


class _ChunkView:
    """Slice view of a fold's uint8 image stack — predict_fold only needs .images and len()."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)


def time_arm(model, images, reps, batch, workers, upscale, decode_u, tta, device, chunk):
    torch.cuda.reset_peak_memory_stats()
    predict_fold(model, _ChunkView(images[:batch]), device=device, batch_size=batch,
                 workers=workers, upscale=upscale, decode_u=decode_u, tta=tta)  # warmup
    reps_chunks = []
    for _ in range(reps):
        reps_chunks.append([])
        for s, e in chunk_bounds(len(images), chunk):
            t0 = time.perf_counter()
            predict_fold(model, _ChunkView(images[s:e]), device=device, batch_size=batch,
                         workers=workers, upscale=upscale, decode_u=decode_u, tta=tta)
            reps_chunks[-1].append((time.perf_counter() - t0, e - s))
    return {"reps": [per_image_summary(rc) for rc in reps_chunks],
            "peak_mem_gib": float(torch.cuda.max_memory_allocated() / 2 ** 30)}


def _dist_ms(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean_ms": float(np.mean(v)), "p50_ms": float(np.percentile(v, 50)),
            "p95_ms": float(np.percentile(v, 95))}


def time_fusion(base_npz: Path, x2_npz: Path, p2cfg, ef, reps, expected_added: int) -> dict:
    bz, xz = np.load(base_npz), np.load(x2_npz, allow_pickle=True)
    rows_all = probability_rows(xz)
    # NpzFile member access re-reads + re-decompresses the WHOLE array every time —
    # materialize each stack once or the loop measures npz decompression, not fusion
    bi_all, bt_all = bz["inst"], bz["type"]
    xi_all, xt_all = xz["inst"], xz["type"]
    n = len(bi_all)
    reps_ms = []
    for _ in range(reps):
        per_wall, per_cpu, total_added = [], [], 0
        for j in range(n):
            w0, c0 = time.perf_counter(), time.process_time()
            _, _, added = ef_apply(bi_all[j], bt_all[j], xi_all[j], xt_all[j],
                                   rows_all.get(j, {}), p2cfg, ef)
            per_wall.append(1000.0 * (time.perf_counter() - w0))
            per_cpu.append(1000.0 * (time.process_time() - c0))
            total_added += len(added)
            if j % 500 == 499:
                print(f"fusion {j + 1}/{n}", flush=True)
        if total_added != expected_added:
            raise SystemExit(f"deployment-path check FAILED: replay added {total_added} != "
                             f"frozen {expected_added} additions")
        reps_ms.append({"wall": _dist_ms(per_wall), "cpu": _dist_ms(per_cpu)})
    return {"n_images": n, "expected_added": expected_added, "reps": reps_ms}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x1-run", type=Path, required=True)
    p.add_argument("--x2-run", type=Path, required=True)
    p.add_argument("--fold", type=int, default=2, help="validation fold to time on (no test reads)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--reps-x1", type=int, default=2)
    p.add_argument("--reps-x2", type=int, default=1)
    p.add_argument("--reps-tta", type=int, default=1)
    p.add_argument("--reps-fuse", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ef-selection", type=Path, required=True)
    p.add_argument("--p2-selection", type=Path, required=True)
    p.add_argument("--base-pred", type=Path, required=True,
                   help="frozen base M/S val predictions npz of the timed pair")
    p.add_argument("--x2-pred", type=Path, required=True,
                   help="frozen x2 du2 val predictions npz (with inst table) of the timed pair")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    ef_sel = json.loads(a.ef_selection.read_text())
    p2_sel = json.loads(a.p2_selection.read_text())
    cfg = frozen_ef_config(p2_sel, ef_sel)
    if cfg["p2"] is None:
        raise SystemExit("the timed pair's frozen P2 selection is identity — nothing to time")

    images = PanNukeFold(a.fold).images
    import os
    result = {"protocol": {"device": a.device, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "batch": a.batch, "workers": a.workers,
                           "chunk": a.chunk, "precision": "fp32",
                           "fold": a.fold, "n_images": int(len(images)),
                           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
                           "ms_lever": "not separately timed (CPU relabel, same order as fuse)"},
              "arms": {}, "fusion": None, "verdict_inputs": {}}

    for name, run, reps, upscale, decode_u, tta in (
            ("x1", a.x1_run, a.reps_x1, run_upscale(a.x1_run), 1.0, False),
            ("x2_du2", a.x2_run, a.reps_x2, run_upscale(a.x2_run), 2.0, False),
            ("x1_tta", a.x1_run, a.reps_tta, run_upscale(a.x1_run), 1.0, True)):
        tc = TrainConfig(split=1, out_dir=str(run), upscale=upscale)
        model = build_model(tc, pretrained=False).to(a.device)
        model.load_state_dict(torch.load(run / "final.pth", map_location=a.device,
                                         weights_only=False)["model"])
        result["arms"][name] = time_arm(model, images, reps, a.batch, a.workers, upscale,
                                        decode_u, tta, a.device, a.chunk)
        del model
        torch.cuda.empty_cache()
        print(name, json.dumps(result["arms"][name]["reps"][-1]), flush=True)

    result["fusion"] = time_fusion(a.base_pred, a.x2_pred, cfg["p2"], cfg["ef"], a.reps_fuse,
                                   int(ef_sel["selected"]["added"]))
    # decision inputs: cost ratios at the best (max-throughput) rep of each arm
    def best_ms(arm):
        return min(r["mean_ms"] for r in result["arms"][arm]["reps"])
    x1_ms, x2_ms, tta_ms = best_ms("x1"), best_ms("x2_du2"), best_ms("x1_tta")
    fuse_wall = min(r["wall"]["mean_ms"] for r in result["fusion"]["reps"])
    fuse_cpu = min(r["cpu"]["mean_ms"] for r in result["fusion"]["reps"])
    result["verdict_inputs"] = {
        "two_model_over_x1_wall": (x1_ms + x2_ms + fuse_wall) / x1_ms,
        "two_model_over_x1_fuse_cpu": (x1_ms + x2_ms + fuse_cpu) / x1_ms,
        "x1_tta_over_x1": tta_ms / x1_ms,
        "fuse_wall_ms_per_image": fuse_wall, "fuse_cpu_ms_per_image": fuse_cpu}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["verdict_inputs"], indent=2))


if __name__ == "__main__":
    main()
