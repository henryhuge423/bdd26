#!/usr/bin/env python
"""Stage-report-IV figures from frozen artifacts (E2a contrasts, DSB gate B, E0 cost).

Reads only frozen analysis outputs; performs no experiment selection or model
evaluation. Writes per-language PDF+PNG figures plus numerical CSVs:

    figs/e2a_contrasts_{en,zh}.pdf/.png   figs/e2a_contrasts.csv
    figs/gate_b_cosines_{en,zh}.pdf/.png  figs/gate_b_cosines.csv
    figs/e0_cost_{en,zh}.pdf/.png         figs/e0_cost.csv
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE = "#2A78D6"   # validated pair (dataviz six-checks PASS, light surface)
ORANGE = "#EB6834"
INK = "#333333"
GRID = "#CCCCCC"
ROWS_GATE_B = ["skips", "np_branch", "hv_branch", "tp_branch", "encoder", "decoder"]
ROW_LABELS = {
    "en": {
        "skips": "skip adapters", "np_branch": "NP branch", "hv_branch": "HV branch",
        "tp_branch": "TP branch", "encoder": "encoder", "decoder": "decoder composite",
    },
    "zh": {
        "skips": "跳接适配层", "np_branch": "NP 分支", "hv_branch": "HV 分支",
        "tp_branch": "TP 分支", "encoder": "编码器", "decoder": "解码器复合",
    },
}
CONTRAST_LABELS = {
    "en": ["x1: hv8 − hv30", "x2: hv30 − hv120", "x2 − x1\n(matched high support)", "x2 − x1\n(matched low support)"],
    "zh": ["x1：hv8 − hv30", "x2：hv30 − hv120", "x2 − x1\n（高支持配对）", "x2 − x1\n（低支持配对）"],
}
CONTRAST_KEYS = ["lower_support_x1", "lower_support_x2", "scale_at_high_support", "scale_at_low_support"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def setup_fonts(zh_font):
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.45, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if zh_font and Path(zh_font).exists():
        font_manager.fontManager.addfont(str(zh_font))
        fam = font_manager.FontProperties(fname=str(zh_font)).get_name()
        plt.rcParams["font.family"] = ["DejaVu Sans", fam]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]


def save_figure(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=180)
    plt.close(fig)


def fig_e2a_contrasts(hv, out, lang, labels, axis_names):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    specs = [("Dead PQ", BLUE, axes[0]), ("bPQ", ORANGE, axes[1])]
    y = list(range(len(CONTRAST_KEYS)))[::-1]
    for endpoint, color, ax in specs:
        means, lo, hi = [], [], []
        for key in CONTRAST_KEYS:
            c = hv["contrasts"][key]
            means.append(c["mean"][endpoint])
            ci = c["image_bootstrap"][endpoint]["ci95"]
            lo.append(ci[0]); hi.append(ci[1])
        ax.axvline(0, color=INK, lw=1.0)
        ax.errorbar(means, y, xerr=[ [m - l for m, l in zip(means, lo)],
                                     [h - m for m, h in zip(means, hi)] ],
                    fmt="o", color=color, ms=7, capsize=3.5, elinewidth=1.6, capthick=1.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.locator_params(axis="x", nbins=3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}" if abs(v) >= 0.0005 else "0"))
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel(axis_names[endpoint])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for m, v in zip(y, means):
            ax.annotate(f"{v:+.4f}", (v, m), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
    fig.suptitle(axis_names["title"], y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out, f"e2a_contrasts_{lang}")


def fig_gate_b(gb2, out, lang, axis_names):
    cos = gb2["per_batch_cosines"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    y = list(range(len(ROWS_GATE_B)))[::-1]
    xmax = max(max(v) for v in cos.values())
    xmin = min(min(v) for v in cos.values())
    for row, mod in zip(y, ROWS_GATE_B):
        vals = cos[mod]
        ax.scatter(vals, [row + 0.18] * len(vals), s=9, color=BLUE, alpha=0.45, edgecolors="none")
        mean = sum(vals) / len(vals)
        neg = sum(c < 0 for c in vals)
        ax.scatter([mean], [row - 0.14], marker="D", s=46, color=ORANGE, zorder=3)
        note = f"neg {neg}/{len(vals)}"
        if mean > 0.7 * xmax:  # keep the label inside the axes for near-max means
            ax.annotate(f"{note} · mean {mean:+.3f}", (mean, row - 0.14), xytext=(-7, -3),
                        textcoords="offset points", va="center", ha="right", fontsize=8.5, color=INK)
        else:
            ax.annotate(f"mean {mean:+.3f} · {note}", (mean, row - 0.14), xytext=(6, -3),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(xmin - 0.06, xmax + 0.08)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([ROW_LABELS[lang][m] for m in ROWS_GATE_B])
    ax.set_ylim(-0.6, len(ROWS_GATE_B) - 0.4)
    ax.set_xlabel(axis_names["x"])
    ax.set_title(axis_names["title"], fontsize=11)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, out, f"gate_b_cosines_{lang}")


def fig_e0(cost, out, lang, axis_names):
    best = {a: min(r["mean_ms"] for r in spec["reps"]) for a, spec in cost["arms"].items()}
    fuse = min(r["wall"]["mean_ms"] for r in cost["fusion"]["reps"])
    total = best["x1"] + best["x2_du2"] + fuse
    arms = [
        ("x1", best["x1"], BLUE), ("fusion", fuse, BLUE),
        ("x2-du2", best["x2_du2"], BLUE), ("x1-TTA", best["x1_tta"], BLUE),
        ("two-model", total, ORANGE),
    ]
    labels = {
        "en": ["x1", "fusion (CPU)", "x2-du2", "x1-TTA", "two-model system\n(x1+x2-du2+fusion)"],
        "zh": ["x1", "融合（CPU）", "x2-du2", "x1-TTA", "两模型系统\n(x1+x2-du2+融合)"],
    }[lang]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    y = list(range(len(arms)))[::-1]
    for row, (_, ms, color), lab in zip(y, arms, labels):
        ax.barh(row, ms, height=0.55, color=color, edgecolor="white", linewidth=1)
        ratio = ms / best["x1"]
        ax.annotate(f"{ms:.1f} ms/img ({ratio:.2f}×)", (ms, row), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(a[1] for a in arms) * 1.30)
    ax.set_xlabel(axis_names["x"])
    ax.set_title(axis_names["title"], fontsize=11)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, out, f"e0_cost_{lang}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-root", type=Path, default=Path("runs/analysis"))
    ap.add_argument("--out", type=Path, default=Path("docs/report/stage_report_4/figs"))
    ap.add_argument("--zh-font", type=Path,
                    default=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_fonts(args.zh_font)

    hv = jload(args.analysis_root / "hv_threshold_controls_20261006" / "verified_numbers.json")
    for key in CONTRAST_KEYS:
        if key not in hv["contrasts"]:
            raise ValueError(f"missing contrast {key} in E2a snapshot")
    gb2 = jload(args.analysis_root / "dsb_gates_20261007" / "gate_b_v2.json")
    n = gb2["batches"]
    if any(len(v) != n for v in gb2["per_batch_cosines"].values()):
        raise ValueError("gate B v2 per-batch lists have inconsistent length")
    cost = jload(args.analysis_root / "inference_cost_20261008" / "cost.json")

    for lang in ("en", "zh"):
        fig_e2a_contrasts(hv, args.out, lang, CONTRAST_LABELS[lang], {
            "Dead PQ": {"en": "ΔDead PQ (95% CI)", "zh": "ΔDead PQ（95% 区间）"}[lang],
            "bPQ": {"en": "ΔbPQ (95% CI)", "zh": "ΔbPQ（95% 区间）"}[lang],
            "title": {"en": "E2a pre-registered contrasts (2-seed means, image bootstrap)",
                      "zh": "E2a 预注册对比（双种子均值，图像 bootstrap 区间）"}[lang],
        })
        fig_gate_b(gb2, args.out, lang, {
            "x": {"en": "per-batch gradient cosine (dots: 64 batches; diamond: mean)",
                  "zh": "逐批梯度余弦（圆点：64 批；菱形：均值）"}[lang],
            "title": {"en": "DSB gate B amended rerun: Dead vs common-class gradients (fold 1)",
                      "zh": "DSB 闸门 B 修正重跑：Dead 与常见类梯度（fold 1）"}[lang],
        })
        fig_e0(cost, args.out, lang, {
            "x": {"en": "ms per image (shared A100, batch 32)", "zh": "毫秒/图（共享 A100，batch 32）"}[lang],
            "title": {"en": "E0 inference cost: EF-P2 two-model system vs single-model x1",
                      "zh": "E0 推理成本：EF-P2 两模型系统 vs 单模型 x1"}[lang],
        })

    with open(args.out / "e2a_contrasts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "endpoint", "mean", "ci95_low", "ci95_high", "seed19", "seed1"])
        for key in CONTRAST_KEYS:
            c = hv["contrasts"][key]
            for e in ("mPQ", "bPQ", "Dead PQ", "strict Dead PQ"):
                ci = c["image_bootstrap"].get(e, {}).get("ci95", ["", ""])
                lo = f"{ci[0]:.8f}" if isinstance(ci[0], float) else ""
                hi = f"{ci[1]:.8f}" if isinstance(ci[1], float) else ""
                w.writerow([key, e, f"{c['mean'][e]:.8f}", lo, hi,
                            f"{c['per_seed']['19'][e]:.8f}", f"{c['per_seed']['1'][e]:.8f}"])
    with open(args.out / "gate_b_cosines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["module", "mean_cosine", "negative_batches", "n_batches", "conflict_fraction"])
        for mod in ROWS_GATE_B:
            vals = gb2["per_batch_cosines"][mod]
            neg = sum(c < 0 for c in vals)
            w.writerow([mod, f"{sum(vals) / len(vals):.8f}", neg, len(vals), f"{neg / len(vals):.6f}"])
    with open(args.out / "e0_cost.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "best_ms_per_img", "img_per_s", "peak_gib", "ratio_vs_x1"])
        x1 = min(r["mean_ms"] for r in cost["arms"]["x1"]["reps"])
        for arm in ("x1", "x2_du2", "x1_tta"):
            spec = cost["arms"][arm]
            b = min(spec["reps"], key=lambda r: r["mean_ms"])
            w.writerow([arm, f"{b['mean_ms']:.6f}", f"{b['img_per_s']:.3f}",
                        f"{spec['peak_mem_gib']:.3f}", f"{b['mean_ms'] / x1:.4f}"])
        fuse = min(r["wall"]["mean_ms"] for r in cost["fusion"]["reps"])
        w.writerow(["fusion_cpu", f"{fuse:.6f}", "", "", f"{fuse / x1:.4f}"])
        w.writerow(["two_model", f"{x1 + min(r['mean_ms'] for r in cost['arms']['x2_du2']['reps']) + fuse:.6f}",
                    "", "", f"{(x1 + min(r['mean_ms'] for r in cost['arms']['x2_du2']['reps']) + fuse) / x1:.4f}"])
    print(args.out)


if __name__ == "__main__":
    main()
