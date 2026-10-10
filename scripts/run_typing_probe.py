#!/usr/bin/env python
"""T 实验 runner：固定几何分型探针 T0–T3，split1/fold2 验证折，预注册判定（spec 2026-10-10）。

    python scripts/run_typing_probe.py \
        --pred runs/cellvit_uni/split1/pred_fold2.npz \
        --feat-fold1 runs/analysis/typing_probe_20261010/feat_fold1.npz \
        --feat-fold2 runs/analysis/typing_probe_20261010/feat_fold2.npz \
        --baseline-mpq <runs/cellvit_uni/split1/eval_fold2/summary.json 的 official.mPQ> \
        --out runs/analysis/typing_probe_20261010

只读 split1 的 train/val 折；fold3 不触碰。判定规则 = spec §2 六条件，运行前冻结，不放宽。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from nucseg.constants import CLASS_NAMES, TISSUES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.pannuke_eval import instance_classes
from nucseg.typing_probe import HEAD_SEEDS, apply_head, sklearn_logreg, t1_classes, train_linear_head

BOOT_N = 1000
BOOT_SEED = 20261010


def t0_classes(inst: np.ndarray, typ: np.ndarray, inst_img: np.ndarray,
               inst_id: np.ndarray) -> np.ndarray:
    """Majority-vote class per instance, rows aligned with the (img,id) table (Review Focus #1)."""
    out = []
    for j in range(len(inst)):
        _, cls = instance_classes(inst[j], typ[j])
        ids = np.unique(inst[j]); ids = ids[ids > 0]
        assert np.array_equal(ids, inst_id[inst_img == j]), \
            f"image {j}: map ids {ids} != table rows {inst_id[inst_img == j]}"
        out.append(cls)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def evaluate_decision(d: dict) -> dict:
    """Spec §2 六条件（冻结）：全部满足才 pass_gate。"""
    checks = {
        "c1_delta_ge_.003": bool(d["delta"] >= .003),
        "c2_ci_lo_gt_0": bool(d["lo"] > 0),
        "c3_strict_not_down": bool(d["strict_delta"] >= 0),
        "c4_dead_within_.002": bool(d["dead_delta"] >= -.002),
        "c5_beats_t1": bool(d["beats_t1"]),
        "c6_seeds_same_direction": bool(d["seeds_same_direction"]),
    }
    return {**checks, "values": {k: d[k] for k in d}, "pass_gate": all(checks.values())}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _per_tissue(ev: RetypeEvaluator, cls: np.ndarray) -> dict:
    pq = ev.per_image_class_pq(cls)
    with np.errstate(all="ignore"):
        img = np.nanmean(pq, 1)
        return {t: float(np.nanmean(img[ev.tissue == t]))
                for t in TISSUES if (ev.tissue == t).any()}


def _score(ev, cls):
    m, per = ev.mpq(cls)
    s, _ = ev.mpq_strict(cls)
    return {"mpq": float(m), "strict": float(s), "per_class": per.tolist(),
            "per_tissue": _per_tissue(ev, cls)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--feat-fold1", type=Path, required=True)
    ap.add_argument("--feat-fold2", type=Path, required=True)
    ap.add_argument("--baseline-mpq", type=float, required=True,
                    help="eval_fold2 official mPQ（T0 恒等绑定）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if (a.out / "results.json").exists():
        raise SystemExit(f"{a.out}/results.json exists; refusing to overwrite")
    a.out.mkdir(parents=True, exist_ok=True)

    pred = dict(np.load(a.pred))
    f2 = PanNukeFold(2)
    inst_img = pred["inst_img"].astype(np.int64)
    inst_id = pred["inst_id"].astype(np.int64)
    ev = RetypeEvaluator(f2.gt_channels, pred["inst"], inst_img, inst_id, f2.tissue)

    # ---- T0（恒等复评，绑定官方 eval_fold2）
    cls0 = t0_classes(pred["inst"], pred["type"], inst_img, inst_id)
    s0 = _score(ev, cls0)
    assert abs(s0["mpq"] - a.baseline_mpq) < 1e-6, \
        f"T0 identity check failed: {s0['mpq']} vs baseline {a.baseline_mpq}"

    # ---- T1（无训练聚合对照）
    cls1 = t1_classes(pred["inst_prob"].astype(np.float32))
    s1 = _score(ev, cls1)
    b1 = ev.mpq_bootstrap_delta(cls1, cls0, BOOT_N, BOOT_SEED)

    # ---- 特征缓存（行序与 pred 表逐位绑定）；每变体训练特征与验证特征同构造
    ff1 = dict(np.load(a.feat_fold1))
    ff2 = dict(np.load(a.feat_fold2))
    assert np.array_equal(ff2["inst_img"], inst_img) and np.array_equal(ff2["inst_id"], inst_id), \
        "feat_fold2 rows do not match the pred instance table"
    ytr = (ff1["cls"] - 1).astype(np.int64)
    variants = {"T2": (ff1["feat"], ff2["feat"]),
                "T3": (ff1["feat"] + ff1["ring"], ff2["feat"] + ff2["ring"])}
    gt_variants = {"T2": ff2["gt_feat"], "T3": ff2["gt_feat"] + ff2["gt_ring"]}

    results = {"T0": {"score": s0}, "T1": {"score": s1, "bootstrap_vs_t0": b1,
                                           "matched_typed_correct": float((cls1[ev.m_row] == ev.m_cls).mean()),
                                           "confusion_t0_to_tx": np.histogram2d(
                                               cls0, cls1, bins=range(1, 7))[0].tolist()}}
    cls_store = {}          # (name, seed) -> cls；不进 JSON
    head0 = {}              # name -> (W, b) seed0，供诊断/重分型复用
    for name, (Xtr_v, Xv) in variants.items():
        per_seed = {}
        for seed in HEAD_SEEDS:
            W, b = train_linear_head(Xtr_v, ytr, seed=seed)
            if seed == HEAD_SEEDS[0]:
                head0[name] = (W, b)
            t0 = time.perf_counter()
            cls = apply_head(W, b, Xv)
            head_ms = (time.perf_counter() - t0) * 1000
            cls_store[(name, seed)] = cls
            per_seed[seed] = {"score": _score(ev, cls), "head_ms": head_ms,
                              "boot": ev.mpq_bootstrap_delta(cls, cls0, BOOT_N, BOOT_SEED)}
        deltas = [v["score"]["mpq"] - s0["mpq"] for v in per_seed.values()]
        results[name] = {
            "per_seed": per_seed,
            "mean_mpq": float(np.mean([v["score"]["mpq"] for v in per_seed.values()])),
            "mean_strict": float(np.mean([v["score"]["strict"] for v in per_seed.values()])),
            "mean_dead": float(np.mean([v["score"]["per_class"][3] for v in per_seed.values()])),
            "mean_delta": float(np.mean(deltas)),
            "seeds_same_direction": bool(len({np.sign(d) for d in deltas}) == 1),
        }
        # GT 区域诊断（spec §2 单列；不进主端点）：seed0 头在 fold2 GT 实例诊断行上的准确率
        g_pred = apply_head(*head0[name], gt_variants[name])
        results[name]["gt_region_diag"] = {
            "acc": float((g_pred == ff2["gt_cls"]).mean()),
            "confusion": np.histogram2d(ff2["gt_cls"], g_pred, bins=range(1, 7))[0].tolist(),
        }
        # sklearn 确定性交叉核对（在验证变体上打分）
        sk_cls = sklearn_logreg_variant(Xtr_v, ytr, Xv)
        results[name]["sklearn_val_mpq"] = _score(ev, sk_cls)["mpq"]
        results[name]["confusion_t0_to_tx"] = np.histogram2d(
            cls0, cls_store[(name, HEAD_SEEDS[0])], bins=range(1, 7))[0].tolist()
        results[name]["matched_typed_correct"] = float(
            (cls_store[(name, HEAD_SEEDS[0])][ev.m_row] == ev.m_cls).mean())

    # ---- 判定（spec §2）：最优学习配置 = mean_delta 更高者（平局取更简单的 T2）
    best = max(("T2", "T3"), key=lambda n: (results[n]["mean_delta"], n == "T2"))
    boots = [results[best]["per_seed"][s]["boot"] for s in HEAD_SEEDS]
    decision = evaluate_decision({
        "delta": results[best]["mean_delta"],
        "lo": min(b["lo"] for b in boots),
        "hi": max(b["hi"] for b in boots),
        "strict_delta": results[best]["mean_strict"] - s0["strict"],
        "dead_delta": results[best]["mean_dead"] - s0["per_class"][3],
        "beats_t1": results[best]["mean_mpq"] > s1["mpq"],
        "seeds_same_direction": results[best]["seeds_same_direction"],
    })

    # ---- 成本（部署增量：导出侧车池化 ms/img + 头前向 ms）
    side2 = json.loads(a.feat_fold2.with_suffix(".npz.json").read_text())
    cost = {"pool_ms_per_img": side2.get("pool_ms_per_img"),
            "head_ms_val_total": results[best]["per_seed"][HEAD_SEEDS[0]]["head_ms"]}

    # ---- 重分型 npz（最优学习配置 seed0；canonical 复核用；inst 不动，type=每实例新类）
    cls_best = cls_store[(best, HEAD_SEEDS[0])]
    lut = np.zeros(int(inst_id.max()) + 1, np.uint8)
    lut[inst_id] = cls_best
    retyped_type = pred["type"].copy()
    for j in range(len(pred["inst"])):
        m = pred["inst"][j] > 0
        retyped_type[j][m] = lut[pred["inst"][j][m]]
    retyped_path = a.out / f"pred_fold2_retyped_{best.lower()}_seed0.npz"
    np.savez_compressed(retyped_path, inst=pred["inst"], type=retyped_type.astype(np.uint8))

    payload = {
        "spec": "docs/superpowers/specs/2026-10-10-typing-probe-t-design.md",
        "class_names": list(CLASS_NAMES),
        "inputs": {str(p): _sha256(p) for p in (a.pred, a.feat_fold1, a.feat_fold2)},
        "baseline_mpq": a.baseline_mpq, "t0_identity": s0["mpq"],
        "results": results, "best_learning": best, "decision": decision, "cost": cost,
        "retyped": str(retyped_path),
        "canonical_check_cmd": f"python scripts/eval_pannuke.py --pred {retyped_path} "
                               f"--fold 2 --out {a.out}/eval_{best.lower()}_seed0",
        "notes": ["bPQ must be bit-identical to eval_fold2 in the canonical check (geometry fixed)",
                  "dev-fold only (split1/fold2); fold3 never read; head seeds cover linear-head "
                  "randomness only; train features = GT contours, val = predicted contours"],
    }
    (a.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"T0": round(s0["mpq"], 6), "T1": round(s1["mpq"], 6),
                      "T2": round(results["T2"]["mean_mpq"], 6),
                      "T3": round(results["T3"]["mean_mpq"], 6), "best": best,
                      "pass_gate": decision["pass_gate"]}, indent=2))


def sklearn_logreg_variant(Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray) -> np.ndarray:
    """Deterministic cross-check: fit lbfgs logistic regression, predict classes 1..5 on Xv."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Xtr, ytr)
    return (lr.predict(Xv) + 1).astype(np.int64)


if __name__ == "__main__":
    main()
