# R1 variant (frozen 2026-10-18): byte-copy of the frozen P1 probe (sha256
# f52519c27a2127bb280158ff289ad6d589c4895a1fdc3859c4d69371798786d6) with the
# pre-declared convergence-budget patches documented in
# docs/proposals/P1R1_CONVERGENCE_CORRECTED_2026-10-18.md: convergence-stopped
# epoch loop (train-loss rule only), primary-arm-only run, in-sample fold1 fit
# diagnostic, head checkpoint every 5 epochs, larger wall-clock budget. All
# scientific parameters (data, weights, architecture, loss, optimizer, seeds,
# decode, metrics, gates, bootstrap) are unchanged from P1.
"""P1: frozen-feature point-heatmap probe (bounded train-only diagnostic; no method claim).

Trains a lightweight center-heatmap head on FROZEN UNI2-h / UNI token features (fold1,
2656 images) and evaluates point recall on fold2 (2523 images) against the trained
CellViT-UNI baseline's NP-foreground components. Frozen design:
docs/superpowers/specs/2026-10-18-p1-frozen-point-heatmap-design.md. Single GPU, no test
fold, no method admission; failure stops without retry.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)

FROZEN_DATA_SHA = {
    "fold1/images.npy": "b4f24c4d117a44fa930120d33b52175dc8189afdceb89a2cdf91d2682293105b",
    "fold1/inst.npy": "59fb7dc60354e54e642f41072595ee3a4bec700c0b36bd2c94075ae030e5de7e",
    "fold1/type.npy": "cda88107c0860fd3be6c58d9dafde7623b9709ee639649246e67c4844ae593c0",
    "fold2/images.npy": "d7c93f1734d218e1313d276ab2c3c02130289e3076108f724324d9f71860aa39",
    "fold2/inst.npy": "145420bb21d05d5047ecc6d0d043ef05d6a2789c608fe987488480d16f7b7c21",
    "fold2/type.npy": "82d09f44f79575fd8629b02495c25dca2ddce12b49c25191e3d084f473f57077",
}
UNI_BIN_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
UNI2H_BIN_SHA = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
BASELINE_CKPT_SHA = "da9a7efddf9cc53ff4fa18160f5bf51f11ab866e51abb16c1104bd55be599bcb"

PROJ = 192
EPOCHS = 10  # P1 value, kept for provenance; R1 uses the convergence rule below
EPOCH_CAP = 100
R1_LOSS_TARGET_5EP = 0.02
R1_PLATEAU_MIN_REL_DROP = 0.05
R1_CHECK_MIN_EPOCH = 15
R1_CHECK_EVERY = 5
BATCH = 16
LR = 3e-4
SEEDS = (19, 1)
DECODE_CAP = 512
DECODE_THRESHOLD = 0.5
NMS_K = 100
HIT_RADIUS_SQ = 9.0  # <= 3 px
GATE_DEAD_DELTA = 0.10
GATE_PRECISION_TOL = 0.10
GATE_NONDEG = 0.95
BOOT_DRAWS = 1000
BOOT_SEED = 20261018
DEADLINE_S = 12000.0  # R1: 3h20m in-script budget (controller allows 12600s)
RADII = (1.0, 2.0, 3.0, 5.0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- frozen GT rules

def gt_objects(inst, typ):
    """Per-GT-id (class, point(y,x), interior) replicating the frozen rules.

    class = majority type vote; point = pixel nearest the mask centroid, first
    occurrence in row-major order (make_targets); interior = no pixel on the
    256-px outer ring (original_full)."""
    inst = np.asarray(inst)
    typ = np.asarray(typ)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    ring = set()
    for edge in (inst[0], inst[-1], inst[:, 0], inst[:, -1]):
        ring.update(int(v) for v in np.unique(edge) if v != 0)
    out = {}
    for raw in ids:
        oid = int(raw)
        yy, xx = np.nonzero(inst == oid)
        votes = np.bincount(typ[yy, xx].astype(np.int64), minlength=6)
        if votes[1:].sum() == 0:
            raise ValueError(f"instance {oid} has no foreground class votes")
        cls = int(votes[1:].argmax() + 1)
        dist = (yy - yy.mean()) ** 2 + (xx - xx.mean()) ** 2
        j = int(dist.argmin())
        out[oid] = (cls, (int(yy[j]), int(xx[j])), oid not in ring)
    return out


def gaussian_target(points, height, width, radius=3, sigma=1.0):
    """E17 center_target: max-combined Gaussian patches; exactly 1 at point cells."""
    result = np.zeros((height, width), dtype=np.float32)
    d = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(d, d, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    for y, x in points:
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        patch = kernel[y0 - y + radius:y1 - y + radius, x0 - x + radius:x1 - x + radius]
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], patch)
    return result


def focal_center_loss(logits, target, count):
    """E17 penalty-reduced focal center loss; logits/target (N,H,W)."""
    p = torch.sigmoid(logits)
    positive = target == 1
    negative = target < 1
    return -(F.logsigmoid(logits) * (1 - p).square() * positive
             + F.logsigmoid(-logits) * p.square() * (1 - target).pow(4) * negative
             ).sum() / max(1, count)


# ---------------------------------------------------------------- frozen decode

def decode_peaks(prob_map, threshold=DECODE_THRESHOLD, cap=DECODE_CAP):
    """D1: 3x3 local maxima with p >= threshold (plateau-inclusive), stable descending
    probability then row-major, top cap. Deterministic CPU float64."""
    p = np.asarray(prob_map, dtype=np.float64)
    pad = np.pad(p, 1, mode="constant", constant_values=-1.0)
    maximum = np.maximum.reduce([pad[i:i + p.shape[0], j:j + p.shape[1]]
                                 for i in range(3) for j in range(3)])
    idx = np.flatnonzero((p >= threshold) & (p == maximum))
    order = np.lexsort((idx, -p.ravel()[idx]))
    idx = idx[order[:cap]]
    return [(int(i // p.shape[1]), int(i % p.shape[1])) for i in idx]


def decode_nms(prob_map, k=NMS_K, radius=1):
    """D2: threshold-free greedy 3x3 NMS, top k, deterministic (argmax = first max)."""
    p = np.asarray(prob_map, dtype=np.float64).copy()
    points = []
    for _ in range(k):
        flat = int(np.argmax(p))
        if p.ravel()[flat] <= 0:
            break
        y, x = divmod(flat, p.shape[1])
        points.append((y, x))
        y0, y1 = max(0, y - radius), min(p.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(p.shape[1], x + radius + 1)
        p[y0:y1, x0:x1] = -1.0
    return points


def hits_for_points(points, gt, mask, radius_sq=HIT_RADIUS_SQ):
    """(hits_all, hits_idead, candidates_inside) for one image."""
    n_all = len(gt)
    n_idead = sum(1 for _, (cls, _, interior) in gt.items() if cls == 4 and interior)
    hits_all = 0
    hits_idead = 0
    inside = 0
    for y, x in points:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            inside += 1
        best_all = False
        best_idead = False
        for _, (cls, (py, px), interior) in gt.items():
            if (y - py) ** 2 + (x - px) ** 2 <= radius_sq:
                best_all = True
                if cls == 4 and interior:
                    best_idead = True
                    break
        hits_all += int(best_all)
        hits_idead += int(best_idead)
    return hits_all, hits_idead, inside, n_all, n_idead


def radius_curve(points, gt):
    """Probe/baseline recall of internal Dead at each radius (descriptive)."""
    out = {}
    for r in RADII:
        hits = 0
        for _, (cls, (py, px), interior) in gt.items():
            if cls == 4 and interior:
                if any((y - py) ** 2 + (x - px) ** 2 <= r * r for y, x in points):
                    hits += 1
        out[f"{r:g}px"] = hits
    return out


# ---------------------------------------------------------------- head

class PointHead(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(w, PROJ, 1) for w in widths])
        self.stem = nn.Conv2d(PROJ * len(widths), 256, 3, padding=1)
        self.up1 = nn.Conv2d(256, 192, 3, padding=1)
        self.up2 = nn.Conv2d(192, 128, 3, padding=1)
        self.up3 = nn.Conv2d(128, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias.fill_(float(np.log(0.01 / 0.99)))

    def forward(self, feats):
        x = torch.cat([proj(f) for proj, f in zip(self.proj, feats)], dim=1)
        x = F.relu(self.stem(x))
        x = F.relu(self.up1(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = F.relu(self.up2(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = F.relu(self.up3(F.interpolate(x, scale_factor=2, mode="nearest")))
        logits = self.out(x)
        return F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------- encoders (frozen)

def load_uni2h(bin_path):
    import timm
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
              dynamic_img_size=True)
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("head")], missing[:5]
    assert not [k for k in unexpected if not k.startswith("head")], unexpected[:5]
    return model.eval()


def load_uni_encoder(snapshot, bin_path):
    from nucseg.cellvit.model import UNIEncoder
    enc = UNIEncoder()
    enc.load_state_dict(torch.load(bin_path, map_location="cpu", weights_only=True), strict=True)
    return enc.eval()


@torch.no_grad()
def extract_uni2h(model, images, batch, device):
    layers = {6, 12, 18, 24}
    n = len(images)
    probe = _probe_features(model, images[:2], device, layers, mode="uni2h")
    _, _, c, g, _ = probe.shape
    cache = np.lib.format.open_memmap(_current_cache_path, mode="w+", dtype=np.float16,
                                      shape=(n, 4, c, g, g))
    for i in range(0, n, batch):
        chunk = torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).to(device)
        x = chunk.permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(252, 252), mode="bilinear", align_corners=False)
        mean = torch.tensor(UNI_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(UNI_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = _forward_uni2h(model, x, layers)
        for k, f in enumerate(feats):
            cache[i:i + len(f), k] = f.half().cpu().numpy()
    cache.flush()
    return str(_current_cache_path), (c, g)


def _forward_uni2h(model, x, layers):
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    feats = []
    for j, blk in enumerate(model.blocks, 1):
        x = blk(x)
        if j in layers:
            feats.append(x[:, model.num_prefix_tokens:])
    out = []
    for f in feats:
        b, p, c = f.shape
        g = int(round(p ** 0.5))
        out.append(f.transpose(1, 2).reshape(b, c, g, g).float())
    return out


def _probe_features(model, images, device, layers, mode):
    chunk = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    x = chunk.permute(0, 3, 1, 2).float().div_(255.0)
    if mode == "uni2h":
        x = F.interpolate(x, size=(252, 252), mode="bilinear", align_corners=False)
    mean = torch.tensor(UNI_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(UNI_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if mode == "uni2h":
            feats = _forward_uni2h(model, x, layers)
        else:
            _, raw = model(x)
            feats = []
            for f in raw:
                b, p, c = f.shape
                g = int(round(p ** 0.5))
                feats.append(f.transpose(1, 2).reshape(b, c, g, g).float())
    return torch.stack(feats, dim=1)  # (B,4,C,g,g)


@torch.no_grad()
def extract_uni(enc, images, batch, device):
    n = len(images)
    probe = _probe_features(enc, images[:2], device, None, mode="uni")
    _, _, c, g, _ = probe.shape
    cache = np.lib.format.open_memmap(_current_cache_path, mode="w+", dtype=np.float16,
                                      shape=(n, 4, c, g, g))
    for i in range(0, n, batch):
        stack = _probe_features(enc, images[i:i + batch], device, None, mode="uni")
        cache[i:i + len(stack)] = stack.half().cpu().numpy()
    cache.flush()
    return str(_current_cache_path), (c, g)


# ---------------------------------------------------------------- baseline forward

@torch.no_grad()
def baseline_np_maps(snapshot, ckpt, images, batch, device):
    from nucseg.cellvit.model import CellViTUNI
    model = CellViTUNI(uni_ckpt=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.eval().to(device)
    mean = torch.tensor(UNI_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(UNI_STD, device=device).view(1, 3, 1, 1)
    out = np.empty((len(images), 256, 256), dtype=np.float16)
    for i in range(0, len(images), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).to(device)
        x = chunk.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)["np"].float()
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float16)
        out[i:i + len(prob)] = prob
    del model
    torch.cuda.empty_cache()
    return out


def np_components(prob_map):
    """Baseline candidates: 8-connected components of NP >= 0.5; candidate per
    component = pixel nearest its centroid (same frozen rule)."""
    from scipy import ndimage
    lab, n = ndimage.label(np.asarray(prob_map) >= DECODE_THRESHOLD,
                           structure=np.ones((3, 3), dtype=int))
    points = []
    for comp in range(1, n + 1):
        yy, xx = np.nonzero(lab == comp)
        dist = (yy - yy.mean()) ** 2 + (xx - xx.mean()) ** 2
        j = int(dist.argmin())
        points.append((int(yy[j]), int(xx[j])))
    return points


# ---------------------------------------------------------------- metrics helpers

def pooled_recall(rows, hits_key, n_key):
    hits = sum(r[hits_key] for r in rows)
    n = sum(r[n_key] for r in rows)
    return hits / n if n else float("nan")


def pooled_precision(rows, inside_key, pred_key):
    inside = sum(r[inside_key] for r in rows)
    pred = sum(r[pred_key] for r in rows)
    return inside / pred if pred else float("nan")


def bootstrap_delta(rows, tissues, endpoint, seed=BOOT_SEED, draws=BOOT_DRAWS):
    """Paired image-level tissue-stratified bootstrap CI for probe - baseline pooled recall."""
    hits_key = f"p_{endpoint}_hits"
    n_key = f"p_{endpoint}_n"  # probe and baseline share GT denominators
    active = [i for i, r in enumerate(rows) if r[n_key] > 0]
    by_tissue = {}
    for i in active:
        by_tissue.setdefault(tissues[i], []).append(i)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for d in range(draws):
        sampled = []
        for idxs in by_tissue.values():
            pick = rng.integers(0, len(idxs), size=len(idxs))
            sampled.extend(idxs[j] for j in pick)
        sub = [rows[i] for i in sampled]
        probe = pooled_recall(sub, hits_key, n_key)
        base = pooled_recall(sub, f"b_{endpoint}_hits", n_key)
        deltas[d] = probe - base
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- run

def r1_stop_decision(epoch_losses, checked_at):
    """R1 pre-declared early-stop rule (pure; train-loss only; frozen 2026-10-18).

    Checked only at epoch multiples of R1_CHECK_EVERY and never before
    R1_CHECK_MIN_EPOCH. Fires when the mean loss of the last 5 epochs is below
    R1_LOSS_TARGET_5EP, or when it has improved by less than
    R1_PLATEAU_MIN_REL_DROP relative to the previous 5-epoch mean."""
    if checked_at < R1_CHECK_MIN_EPOCH or checked_at % R1_CHECK_EVERY != 0:
        return False, ""
    if len(epoch_losses) != checked_at:
        raise ValueError("history length must equal checked_at")
    last5 = sum(epoch_losses[-5:]) / 5.0
    if last5 < R1_LOSS_TARGET_5EP:
        return True, "loss_target:%.5f" % last5
    prev5 = sum(epoch_losses[-10:-5]) / 5.0
    if prev5 > 0 and (prev5 - last5) / prev5 < R1_PLATEAU_MIN_REL_DROP:
        return True, "plateau:%.5f->%.5f" % (prev5, last5)
    return False, ""


def run(args):
    start = time.monotonic()
    snapshot = args.snapshot.resolve()
    sys.path.insert(0, str(snapshot / "src"))
    torch.set_num_threads(4)

    def deadline():
        if time.monotonic() - start > DEADLINE_S:
            raise RuntimeError("P1 budget exhausted; stop, no retry")

    assert torch.cuda.device_count() == 1
    import shutil
    assert shutil.disk_usage(args.tmp.parent).free > 50 * 1024 ** 3, "need >=50GiB free"
    args.out.mkdir(parents=True, exist_ok=False)
    args.tmp.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.30)

    for rel, digest in FROZEN_DATA_SHA.items():
        assert sha256_file(args.data_root / rel) == digest, rel
    uni_bin = args.weights_root / "MahmoodLab" / "UNI" / "pytorch_model.bin"
    uni2h_bin = args.weights_root / "MahmoodLab" / "UNI2-h" / "pytorch_model.bin"
    assert sha256_file(uni_bin) == UNI_BIN_SHA
    assert sha256_file(uni2h_bin) == UNI2H_BIN_SHA
    assert sha256_file(args.baseline_ckpt) == BASELINE_CKPT_SHA

    manifest = {
        "design_sha256": sha256_file(args.design),
        "script_sha256": sha256_file(Path(__file__)),
        "data_sha256": FROZEN_DATA_SHA,
        "uni_bin_sha256": UNI_BIN_SHA,
        "uni2h_bin_sha256": UNI2H_BIN_SHA,
        "baseline_ckpt_sha256": BASELINE_CKPT_SHA,
        "torch": torch.__version__, "numpy": np.__version__,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": {"proj": PROJ, "r1_epoch_rule": "cap=%d loss<%.2f plateau_rel<%.2f check_every=%d from=%d" % (EPOCH_CAP, R1_LOSS_TARGET_5EP, R1_PLATEAU_MIN_REL_DROP, R1_CHECK_EVERY, R1_CHECK_MIN_EPOCH), "batch": BATCH, "lr": LR, "seeds": list(SEEDS),
                   "decode": "D1 local maxima >=0.5 top512; D2 NMS top100",
                   "hit_radius_px": 3.0, "gates": {"dead_delta": GATE_DEAD_DELTA,
                                                   "precision_tol": GATE_PRECISION_TOL,
                                                   "nondeg": GATE_NONDEG},
                   "boot": {"draws": BOOT_DRAWS, "seed": BOOT_SEED}},
        "scope": "Frozen features, fold1 train / fold2 dev validation; fresh full train fold; no test fold; no method admission",
    }

    device = "cuda"
    fold1 = {k: np.load(args.data_root / "fold1" / (k + ".npy"), mmap_mode="r")
             for k in ("images", "inst", "type")}
    fold2 = {k: np.load(args.data_root / "fold2" / (k + ".npy"), mmap_mode="r")
             for k in ("images", "inst", "type", "tissue")}
    assert len(fold1["images"]) == 2656 and len(fold2["images"]) == 2523

    gts1 = [gt_objects(fold1["inst"][i], fold1["type"][i]) for i in range(len(fold1["images"]))]
    deadline()
    print(json.dumps({"gt_fold1_objects": sum(len(g) for g in gts1)}), flush=True)
    gts2 = [gt_objects(fold2["inst"][i], fold2["type"][i]) for i in range(len(fold2["images"]))]
    deadline()
    print(json.dumps({"gt_fold2_objects": sum(len(g) for g in gts2),
                      "fold2_internal_dead": sum(1 for g in gts2 for c, _, it in g.values() if c == 4 and it)}), flush=True)

    # ---- baseline NP forward + components ----
    np_maps = baseline_np_maps(snapshot, args.baseline_ckpt, fold2["images"], 32, device)
    deadline()
    base_rows = []
    for i in range(len(fold2["images"])):
        pts = np_components(np_maps[i])
        hits, idead_hits, inside, n_all, n_idead = hits_for_points(pts, gts2[i], fold2["inst"][i])
        base_rows.append({"image": i, "pred": len(pts), "inside": inside,
                          "all_hits": hits, "idead_hits": idead_hits,
                          "all_n": n_all, "idead_n": n_idead,
                          "curve": radius_curve(pts, gts2[i])})
    del np_maps
    torch.cuda.empty_cache()
    print(json.dumps({"baseline_candidates": sum(r["pred"] for r in base_rows)}), flush=True)

    results = {"manifest": manifest, "baseline": base_rows}
    arm_reports = {}
    for arm, loader, extractor in (
        # R1: primary arm only (pre-declared in the addendum; secondary arm omitted)
        ("uni2h", load_uni2h, "uni2h"),
    ):
        if arm == "uni2h":
            encoder = load_uni2h(uni2h_bin).to(device)
        else:
            encoder = load_uni_encoder(snapshot, uni_bin).to(device)
        feats1_path = args.tmp / f"feats_{arm}_fold1.npy"
        feats2_path = args.tmp / f"feats_{arm}_fold2.npy"
        global _current_cache_path
        _current_cache_path = feats1_path
        _, (c, g) = (extract_uni2h(encoder, fold1["images"], 16, device) if arm == "uni2h"
                     else extract_uni(encoder, fold1["images"], 16, device))
        deadline()
        _current_cache_path = feats2_path
        (extract_uni2h(encoder, fold2["images"], 16, device) if arm == "uni2h"
         else extract_uni(encoder, fold2["images"], 16, device))
        deadline()
        del encoder
        torch.cuda.empty_cache()
        print(json.dumps({"arm": arm, "features": [c, g]}), flush=True)

        feats1 = np.load(feats1_path, mmap_mode="r")
        feats2 = np.load(feats2_path, mmap_mode="r")
        # fold1-only per-layer stats
        acc = np.zeros((4, c), dtype=np.float64)
        acc2 = np.zeros((4, c), dtype=np.float64)
        count = 0
        for i in range(0, len(feats1), 16):
            chunk = feats1[i:i + 16].astype(np.float32)
            acc += chunk.sum(axis=(0, 3, 4), dtype=np.float64)
            acc2 += np.square(chunk, dtype=np.float64).sum(axis=(0, 3, 4))
            count += chunk.shape[0] * g * g
        mean = (acc / count).astype(np.float32)
        var = np.maximum(acc2 / count - np.square(acc / count), 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.maximum(std, 1e-3).astype(np.float32)
        np.save(args.out / f"norm_stats_{arm}.npy", np.stack([mean, std]))
        deadline()

        seed_rows = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = PointHead([c] * 4).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)
            rng = np.random.default_rng(seed)
            head.train()
            step = 0
            epoch_losses = []
            stop_reason = "cap:%d" % EPOCH_CAP
            for epoch in range(1, EPOCH_CAP + 1):
                epoch_sum = 0.0
                epoch_steps = 0
                perm = rng.permutation(len(feats1))
                for b in range(0, len(perm) - BATCH + 1, BATCH):
                    idxs = perm[b:b + BATCH]
                    batch_np = np.stack([feats1[i] for i in idxs])
                    feats = torch.from_numpy(batch_np.astype(np.float32)).to(device)
                    feats = (feats - torch.from_numpy(mean).to(device)[None, :, :, None, None]
                             ) / torch.from_numpy(std).to(device)[None, :, :, None, None]
                    target = np.stack([gaussian_target(
                        [p for _, (_, p, _) in gts1[i].items()], 256, 256) for i in idxs])
                    count_gt = sum(len(gts1[i]) for i in idxs)
                    logits = head([feats[:, k] for k in range(4)])
                    loss = focal_center_loss(logits[:, 0], torch.from_numpy(target).to(device), count_gt)
                    assert bool(torch.isfinite(loss))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                               for p in head.parameters())
                    opt.step()
                    step += 1
                    epoch_sum += float(loss.detach())
                    epoch_steps += 1
                    if step % 200 == 0:
                        print(json.dumps({"arm": arm, "seed": seed, "step": step,
                                          "loss": float(loss.detach())}), flush=True)
                    deadline()
                epoch_losses.append(epoch_sum / max(1, epoch_steps))
                print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch,
                                  "epoch_loss": epoch_losses[-1]}), flush=True)
                if epoch % R1_CHECK_EVERY == 0:
                    torch.save(head.state_dict(),
                               str(args.out / ("r1_head_%s_%d_ep%d.pt" % (arm, seed, epoch))))
                stop, reason = r1_stop_decision(epoch_losses, epoch)
                if stop:
                    stop_reason = reason
                    break
            head.eval()
            rows = []
            with torch.no_grad():
                for i in range(len(feats2)):
                    batch_np = np.stack([feats2[i]]).astype(np.float32)
                    feats = torch.from_numpy(batch_np).to(device)
                    feats = (feats - torch.from_numpy(mean).to(device)[None, :, :, None, None]
                             ) / torch.from_numpy(std).to(device)[None, :, :, None, None]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = head([feats[:, k] for k in range(4)])
                    prob = logits[0, 0].float().cpu().numpy()
                    pts = decode_peaks(prob)
                    nms = decode_nms(prob)
                    hits, idead_hits, inside, n_all, n_idead = hits_for_points(
                        pts, gts2[i], fold2["inst"][i])
                    nms_hits, nms_idead, _, _, _ = hits_for_points(nms, gts2[i], fold2["inst"][i])
                    rows.append({"image": i, "pred": len(pts), "inside": inside,
                                 "all_hits": hits, "idead_hits": idead_hits,
                                 "all_n": n_all, "idead_n": n_idead,
                                 "nms_pred": len(nms), "nms_all_hits": nms_hits,
                                 "nms_idead_hits": nms_idead,
                                 "curve": radius_curve(pts, gts2[i])})
                    deadline()
            train_fit = {"images": len(feats1), "d1_pred": 0, "d1_all_hits": 0,
                         "d1_all_n": 0, "d1_idead_hits": 0, "d1_idead_n": 0, "d1_inside": 0}
            with torch.no_grad():
                for i in range(len(feats1)):
                    batch_np = np.stack([feats1[i]]).astype(np.float32)
                    feats = torch.from_numpy(batch_np).to(device)
                    feats = (feats - torch.from_numpy(mean).to(device)[None, :, :, None, None]
                             ) / torch.from_numpy(std).to(device)[None, :, :, None, None]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = head([feats[:, k] for k in range(4)])
                    prob = logits[0, 0].float().cpu().numpy()
                    pts = decode_peaks(prob)
                    hits, idead_hits, inside, n_all, n_idead = hits_for_points(
                        pts, gts1[i], fold1["inst"][i])
                    train_fit["d1_pred"] += len(pts)
                    train_fit["d1_all_hits"] += hits
                    train_fit["d1_all_n"] += n_all
                    train_fit["d1_idead_hits"] += idead_hits
                    train_fit["d1_idead_n"] += n_idead
                    train_fit["d1_inside"] += inside
                    deadline()
            train_fit["d1_all_recall"] = (train_fit["d1_all_hits"] / train_fit["d1_all_n"]
                                          if train_fit["d1_all_n"] else float("nan"))
            train_fit["d1_precision"] = (train_fit["d1_inside"] / train_fit["d1_pred"]
                                         if train_fit["d1_pred"] else float("nan"))
            print(json.dumps({"arm": arm, "seed": seed, "in_sample_diagnostic": train_fit}),
                  flush=True)
            merged = []
            for pb, bs in zip(rows, base_rows):
                merged.append({"image": pb["image"],
                               "p_all_hits": pb["all_hits"], "p_all_n": pb["all_n"],
                               "p_idead_hits": pb["idead_hits"], "p_idead_n": pb["idead_n"],
                               "b_all_hits": bs["all_hits"], "b_all_n": bs["all_n"],
                               "b_idead_hits": bs["idead_hits"], "b_idead_n": bs["idead_n"],
                               "p_pred": pb["pred"], "p_inside": pb["inside"],
                               "b_pred": bs["pred"], "b_inside": bs["inside"],
                               "p_nms_pred": pb["nms_pred"], "p_nms_all_hits": pb["nms_all_hits"],
                               "p_nms_idead_hits": pb["nms_idead_hits"],
                               "p_curve": pb["curve"], "b_curve": bs["curve"]})
            tissues = [str(t) for t in fold2["tissue"]]
            probe_dead = pooled_recall(merged, "p_idead_hits", "p_idead_n")
            base_dead = pooled_recall(merged, "b_idead_hits", "b_idead_n")
            probe_all = pooled_recall(merged, "p_all_hits", "p_all_n")
            base_all = pooled_recall(merged, "b_all_hits", "b_all_n")
            p_prec = pooled_precision(merged, "p_inside", "p_pred")
            b_prec = pooled_precision(merged, "b_inside", "b_pred")
            probe_nms_dead = pooled_recall(merged, "p_nms_idead_hits", "p_idead_n")
            lo, hi = bootstrap_delta(merged, tissues, "idead")
            active_images = [r for r in merged if r["p_idead_n"] > 0]
            nondeg_frac = (sum(1 for r in active_images if r["p_pred"] > 0) / len(active_images)
                           if active_images else float("nan"))
            clauses = {
                "dead_delta_ge_0.10": (probe_dead - base_dead) >= GATE_DEAD_DELTA,
                "bootstrap_ci_gt_0": lo > 0,
                "precision_ge_baseline_minus_0.10": p_prec >= (b_prec - GATE_PRECISION_TOL),
                "nondegenerate_ge_0.95": nondeg_frac >= GATE_NONDEG,
            }
            seed_rows[str(seed)] = {
                "metrics": {"probe_dead_recall_d1": probe_dead, "baseline_dead_recall": base_dead,
                            "delta_dead_d1": probe_dead - base_dead,
                            "probe_all_recall_d1": probe_all, "baseline_all_recall": base_all,
                            "probe_precision_d1": p_prec, "baseline_precision": b_prec,
                            "probe_dead_recall_d2": probe_nms_dead,
                            "candidates_per_image_d1": sum(r["p_pred"] for r in merged) / len(merged),
                            "candidates_per_image_d2": sum(r["p_nms_pred"] for r in merged) / len(merged),
                            "bootstrap_ci_delta_dead_d1": [lo, hi],
                            "nondeg_fraction": nondeg_frac},
                "clauses": clauses,
                "pass": all(clauses.values()),
                "per_image": merged,
                "radius_curve_probe_dead": {k: sum(r["p_curve"][k] for r in merged) for k in merged[0]["p_curve"]},
                "radius_curve_baseline_dead": {k: sum(r["b_curve"][k] for r in merged) for k in merged[0]["b_curve"]},
                "dead_n": sum(r["p_idead_n"] for r in merged),
                "train_fit_diagnostic": train_fit,
                "epochs_run": len(epoch_losses),
                "stop_reason": stop_reason,
                "epoch_loss_history": epoch_losses,
            }
            print(json.dumps({"arm": arm, "seed": seed,
                              **{k: v for k, v in seed_rows[str(seed)]["metrics"].items()},
                              "pass": seed_rows[str(seed)]["pass"]}), flush=True)
            np.save(args.out / f"norm_reused_{arm}_{seed}.npy", np.stack([mean, std]))
            del head, opt, rows, merged
            torch.cuda.empty_cache()
        arm_reports[arm] = seed_rows
        del feats1, feats2

    primary = arm_reports["uni2h"]
    verdict = {
        "primary_arm": "uni2h",
        "pass_primary": all(primary[s]["pass"] for s in primary),
        "r1": True,
        "note": "pass_primary permits a separately recorded P2 design only; no test evaluation, no method claim",
    }
    results["arms"] = {arm: {s: {k: v for k, v in seed_rows[s].items() if k != "per_image"}
                             for s in seed_rows} for arm, seed_rows in arm_reports.items()}
    results["verdict"] = verdict
    results["elapsed_seconds"] = time.monotonic() - start
    results["peak_reserved_gib"] = float(torch.cuda.max_memory_reserved() / 1024 ** 3)
    (args.out / "results.json").write_text(json.dumps(results))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for arm, seed_rows in arm_reports.items():
        for s in seed_rows:
            (args.out / f"{arm}_seed{s}.json").write_text(json.dumps(seed_rows[s]))
    print(json.dumps({"verdict": verdict, "elapsed": results["elapsed_seconds"],
                      "peak_gib": results["peak_reserved_gib"]}), flush=True)


_current_cache_path = None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--baseline-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tmp", type=Path, required=True)
    run(p.parse_args())
