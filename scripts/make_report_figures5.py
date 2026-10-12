#!/usr/bin/env python
"""Stage-report-V figures from frozen artifacts (E1-C support, E1-T pairs, E3 strict axis).

Reads only frozen analysis outputs; performs no experiment selection or model
evaluation. Writes per-language PDF+PNG figures plus numerical CSVs:

    figs/e1c_support_{en,zh}.pdf/.png   figs/e1c_support.csv
    figs/e1t_pairs_{en,zh}.pdf/.png     figs/e1t_pairs.csv
    figs/e3_strict_{en,zh}.pdf/.png     figs/e3_strict.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE = "#2A78D6"   # validated pair (dataviz six-checks PASS, light surface)
ORANGE = "#EB6834"
INK = "#333333"
GRID = "#CCCCCC"
CLASSES = {0: "untyped", 1: "Neo", 2: "Inf", 3: "Conn", 4: "Dead", 5: "Epi"}


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


def load_e1c(analysis_root):
    c = jload(analysis_root / "e1_consensus_20261010" / "consensus.json")
    nc = jload(analysis_root / "e1_consensus_20261010" / "null_control.json")
    z = np.load(analysis_root / "e1_consensus_20261010" / "per_candidate.npz")
    iou = z["iou"]
    sup = iou >= c["primary_thr"]
    cnt = sup.sum(1)
    jf = z["jury_fold"]
    ja = np.array([str(x) for x in z["jury_arch"]])
    crossfold = ((sup & (jf[None, :] == 3)).sum(1) > 0) | \
                ((sup & (ja[None, :] != "CellViT")).sum(1) > 0)
    rec = np.load(analysis_root / "e1_input_check_20261010_split1_seed1" / "records.npz")
    added = rec["added"].astype(bool)
    ioub = rec["iou_best"]
    matched = rec["matched"].astype(bool)
    return c, nc, cnt, crossfold, added, ioub, matched


def fig_e1c(c, nc, cnt, crossfold, added, ioub, matched, out, lang, texts):
    n_jury = len(c["jury"])
    ks = np.arange(n_jury + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    # (a) support histograms, log scale, with the null control overlaid
    ax = axes[0]
    h_s2 = np.array(c["per_source"]["S2_added"]["support_hist_thr0.3"], dtype=float)
    h_s3 = np.array(c["per_source"]["S3_rejected"]["support_hist_thr0.3"], dtype=float)
    h_null = np.array(nc["support_hist"], dtype=float)
    ax.plot(ks, h_s2, "o-", color=BLUE, ms=4, lw=1.4, label=texts["s2"])
    ax.plot(ks, h_s3, "s-", color=ORANGE, ms=4, lw=1.4, label=texts["s3"])
    ax.plot(ks, h_null, "^:", color=INK, ms=4, lw=1.2, label=texts["null"])
    ax.set_yscale("log")
    ax.set_xlabel(texts["support"])
    ax.set_ylabel(texts["count"])
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title(texts["title_a"], fontsize=10)
    # (b) GT match rate by binned support
    ax = axes[1]
    bins = [(0, 2), (3, 4), (5, n_jury)]
    xs = np.arange(len(bins))
    for mask, color, lab in ((added, BLUE, texts["s2"]), (~added, ORANGE, texts["s3"])):
        rates, ns = [], []
        for lo, hi in bins:
            b = mask & (cnt >= lo) & (cnt <= hi)
            ns.append(int(b.sum()))
            rates.append(matched[b].mean() if b.any() else float("nan"))
        ax.bar(xs + (0.18 if color == BLUE else -0.18), rates, width=0.34,
               color=color, edgecolor="white", linewidth=1, label=lab)
        for x, r, n in zip(xs, rates, ns):
            ax.annotate(f"{r:.2f}\n(n={n})", (x + (0.18 if color == BLUE else -0.18), r),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([texts["bin"].format(lo, hi if hi < n_jury else n_jury) for lo, hi in bins])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel(texts["match_rate"])
    ax.set_title(texts["title_b"], fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, f"e1c_support_{lang}")


def fig_e1t(e, out, lang, texts):
    overall = e["overall"]["G1"]
    pairs = sorted(((k, v) for k, v in e["by_pair"].items() if v["n"] >= 460),
                   key=lambda kv: -kv[1]["n"])[:6]
    keys = ["OVERALL"] + [k for k, _ in pairs] + ["5,1"]
    labels, base, gt, lo, hi, ns = [], [], [], [], [], []
    for k in keys:
        v = overall if k == "OVERALL" else e["by_pair"][k]
        n = v["n"]
        ns.append(n)
        base.append(v["maj_eq_base"] / n)
        gt.append(v["maj_eq_gt"] / n)
        if k == "OVERALL":
            lo.append(e["bootstrap"]["G1_maj_eq_base"][0])
            hi.append(e["bootstrap"]["G1_maj_eq_base"][1])
        elif k in ("2,3", "3,2"):
            lo.append(e["bootstrap"][f"pair_{k}_maj_eq_base"][0])
            hi.append(e["bootstrap"][f"pair_{k}_maj_eq_base"][1])
        else:
            lo.append(float("nan"))
            hi.append(float("nan"))
        if k == "OVERALL":
            labels.append(texts["overall"])
        else:
            gt_c, base_c = k.split(",")
            labels.append(f"{CLASSES[int(gt_c)]}→{CLASSES[int(base_c)]}\n(n={n})"
                          + ("  *" if k == "5,1" else ""))
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.barh(y + 0.19, base, height=0.36, color=BLUE, edgecolor="white", linewidth=1,
            label=texts["base_bar"])
    ax.barh(y - 0.19, gt, height=0.36, color=ORANGE, edgecolor="white", linewidth=1,
            label=texts["gt_bar"])
    for yy, b, g, l, h in zip(y, base, gt, lo, hi):
        if np.isfinite(l):
            ax.plot([l, h], [yy + 0.19, yy + 0.19], color=INK, lw=1.2, zorder=3)
            for x in (l, h):
                ax.plot([x, x], [yy + 0.13, yy + 0.25], color=INK, lw=1.2, zorder=3)
            lab_x = max(b, h)  # keep the value label clear of the whisker cap
        else:
            lab_x = b
        ax.annotate(f"{b:.3f}", (lab_x, yy + 0.19), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
        ax.annotate(f"{g:.3f}", (g, yy - 0.19), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel(texts["share"])
    ax.set_title(texts["title"], fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, out, f"e1t_pairs_{lang}")


def fig_e3(q, out, lang, texts):
    rows = q["rows"]
    ident = next(r for r in rows if r["name"] == "identity")
    a1 = next(r for r in rows if r["name"] == "a1_incumbent")
    learned = [r for r in rows if r["arm"] in ("A2", "A3", "A4")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xmin = min(r["Dead_PQ"] for r in q["rows"]) - 0.0015
    xmax = max(r["Dead_PQ"] for r in q["rows"]) + 0.0015
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.110, 0.142)
    ax.axhline(ident["strict_dead"], color=INK, lw=1.1, ls="--", zorder=1)
    ax.annotate(texts["floor"].format(ident["strict_dead"]),
                (xmin + 0.0002, ident["strict_dead"]), xytext=(0, -3), textcoords="offset points",
                va="top", fontsize=8.5, color=INK)
    for r in learned:
        ax.scatter(r["Dead_PQ"], r["strict_dead"], s=26, color=BLUE, alpha=0.75,
                   edgecolors="none", zorder=2)
    ax.scatter([ident["Dead_PQ"]], [ident["strict_dead"]], marker="*", s=170, color=INK,
               zorder=4, label=texts["ident"])
    ax.scatter([a1["Dead_PQ"]], [a1["strict_dead"]], marker="D", s=60, color=ORANGE,
               zorder=4, label=texts["a1"])
    best = max(learned, key=lambda r: r["strict_dead"])
    ax.annotate(f"{best['name']}\n{best['strict_dead']:.4f}",
                (best["Dead_PQ"], best["strict_dead"]), xytext=(-8, -6),
                textcoords="offset points", ha="right", va="top", fontsize=8, color=BLUE)
    ax.annotate(texts["a1_label"].format(a1["strict_dead"]),
                (a1["Dead_PQ"], a1["strict_dead"]), xytext=(-8, -4),
                textcoords="offset points", ha="right", va="top", fontsize=8, color=ORANGE)
    ax.set_xlabel(texts["x"])
    ax.set_ylabel(texts["y"])
    ax.set_title(texts["title"], fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, f"e3_strict_{lang}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-root", type=Path, default=Path("runs/analysis"))
    ap.add_argument("--out", type=Path, default=Path("docs/report/stage_report_5/figs"))
    ap.add_argument("--zh-font", type=Path,
                    default=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_fonts(args.zh_font)

    c, nc, cnt, crossfold, added, ioub, matched = load_e1c(args.analysis_root)
    e = jload(args.analysis_root / "e1t_typing_20261011" / "stats.json")
    q = jload(args.analysis_root / "e3_quality_20261011" / "results.json")

    for lang in ("en", "zh"):
        fig_e1c(c, nc, cnt, crossfold, added, ioub, matched, args.out, lang, {
            "s2": {"en": "S2 accepted (n=1515)", "zh": "S2 已接受（n=1515）"}[lang],
            "s3": {"en": "S3 rejected (n=2569)", "zh": "S3 被拒绝（n=2569）"}[lang],
            "null": {"en": "null blobs (n=2000)", "zh": "空对照圆盘（n=2000）"}[lang],
            "support": {"en": "jury support (members with IoU ≥ 0.3)", "zh": "jury 支持度（IoU ≥ 0.3 的成员数）"}[lang],
            "count": {"en": "candidates (log)", "zh": "候选数（对数轴）"}[lang],
            "title_a": {"en": "Support distribution vs synthetic null control", "zh": "支持度分布与合成空对照"}[lang],
            "bin": {"en": "{}–{}", "zh": "{}–{}"}[lang],
            "match_rate": {"en": "GT match rate", "zh": "GT 匹配率"}[lang],
            "title_b": {"en": "GT match rate by binned support", "zh": "按支持度分箱的 GT 匹配率"}[lang],
        })
        fig_e1t(e, args.out, lang, {
            "overall": {"en": "all mistyped\n(n=6858)", "zh": "全部分错\n(n=6858)"}[lang],
            "base_bar": {"en": "jury majority = base class", "zh": "jury 多数类 = 基类"}[lang],
            "gt_bar": {"en": "jury majority = GT class", "zh": "jury 多数类 = 真值类"}[lang],
            "share": {"en": "share of mistyped matched instances", "zh": "占分错已匹配实例的比例"}[lang],
            "title": {"en": "E1-T: jury sides with the base model on its mistypes "
                             "(GT→base cells, * = exception where GT wins)",
                      "zh": "E1-T：基模型分错实例上 jury 系统性支持基类"
                            "（真值→基类大格；* 为真值占优的例外）"}[lang],
        })
        fig_e3(q, args.out, lang, {
            "floor": {"en": "selection floor: identity strict Dead = {:.5f}",
                      "zh": "选择下限：identity strict Dead = {:.5f}"}[lang],
            "ident": {"en": "A0 identity", "zh": "A0 identity"}[lang],
            "a1": {"en": "A1 frozen EF-P2 (incumbent)", "zh": "A1 冻结 EF-P2（现任）"}[lang],
            "a1_label": {"en": "A1 incumbent\n{:.5f}", "zh": "A1 现任\n{:.5f}"}[lang],
            "x": {"en": "official Dead PQ (fold 2)", "zh": "官方 Dead PQ（fold 2）"}[lang],
            "y": {"en": "strict Dead PQ (fold 2)", "zh": "strict Dead PQ（fold 2）"}[lang],
            "title": {"en": "E3: learned rows raise official Dead PQ but sit below "
                             "the strict-Dead floor (dots: A2–A4 sweep)",
                      "zh": "E3：学得行抬升官方 Dead PQ，但全部低于 strict Dead 下限"
                            "（圆点：A2–A4 菜单）"}[lang],
        })

    n_jury = len(c["jury"])
    with open(args.out / "e1c_support.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["support", "n_S2", "n_S3", "n_null", "match_rate_S2", "match_rate_S3"])
        h2 = c["per_source"]["S2_added"]["support_hist_thr0.3"]
        h3 = c["per_source"]["S3_rejected"]["support_hist_thr0.3"]
        m2 = c["per_source"]["S2_added"]["matched_rate_by_support"]
        m3 = c["per_source"]["S3_rejected"]["matched_rate_by_support"]
        for k in range(n_jury + 1):
            w.writerow([k, h2[k], h3[k], nc["support_hist"][k],
                        f"{m2[str(k)]:.6f}" if str(k) in m2 else "",
                        f"{m3[str(k)]:.6f}" if str(k) in m3 else ""])
    with open(args.out / "e1t_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_gt_base", "n", "maj_eq_base", "maj_eq_gt", "maj_other",
                    "no_majority", "ci95_base_low", "ci95_base_high"])
        o1 = e["overall"]["G1"]
        w.writerow(["overall", o1["n"], o1["maj_eq_base"], o1["maj_eq_gt"], o1["maj_other"],
                    o1["no_majority"], f"{e['bootstrap']['G1_maj_eq_base'][0]:.6f}",
                    f"{e['bootstrap']['G1_maj_eq_base'][1]:.6f}"])
        for k, v in sorted(e["by_pair"].items(), key=lambda kv: -kv[1]["n"]):
            ci = e["bootstrap"].get(f"pair_{k}_maj_eq_base", ["", ""])
            w.writerow([k, v["n"], v["maj_eq_base"], v["maj_eq_gt"], v["maj_other"],
                        v["no_majority"],
                        f"{ci[0]:.6f}" if ci[0] != "" else "",
                        f"{ci[1]:.6f}" if ci[1] != "" else ""])
    with open(args.out / "e3_strict.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "arm", "mPQ", "bPQ", "Dead_PQ", "strict_dead", "added", "unmatched"])
        for r in q["rows"]:
            w.writerow([r["name"], r["arm"], f"{r['mPQ']:.8f}", f"{r['bPQ']:.8f}",
                        f"{r['Dead_PQ']:.8f}", f"{r['strict_dead']:.8f}",
                        r["added"], r["unmatched"]])
    print(args.out)


if __name__ == "__main__":
    main()
