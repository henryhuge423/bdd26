"""Report figures for stage report I (+ plain-language summary), from verified numbers.

Sources (do not edit without re-checking the logs):
- Error decomposition (CellViT-UNI, fold 3, IoU 0.5, fraction of GT nuclei): docs/findings.md 2026-09-25
  "Error decomposition of CellViT-UNI on fold 3" (== Tab. 4 of the report).
- Axis decomposition (architecture means): logs/analyze_cf3.log (2026-09-28) (== Tab. 8).
- Dead PQ in-domain vs PUMA: docs/findings.md 2026-09-25 baselines table + 2026-09-27 PUMA entry
  (CellViT .176->.002, HoVer-NeXt-T .136->.005, HoVer-Net .127->.033).

Palette: dataviz reference categorical slots 1-5 (validated light/white).
Output: docs/report/stage_report_1/figs/*.png (300 dpi).
Run: ~/.conda/envs/nuclei/bin/python3 scripts/make_report_figures.py
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

FIGS = Path(__file__).resolve().parent.parent / "docs" / "report" / "stage_report_1" / "figs"

# dataviz reference palette (light mode, validated on white)
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# CJK fallback for the plain-summary variants (Droid Sans Fallback is a single-face TTF;
# Noto Sans CJK SC works too where the .ttc is index-resolved).
for p in [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]:
    if Path(p).exists():
        fontManager.addfont(p)
CJK = FontProperties(
    fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
    if Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf").exists()
    else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)

mpl.rcParams.update({
    # Per-glyph fallback only fires when font.family is the direct list in
    # mpl 3.7 (the sans-serif alias expansion drops the fallback) -- DejaVu
    # first (latin), Droid Sans Fallback catches CJK.
    "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
    "font.size": 11.5,
    "axes.edgecolor": BASE,
    "axes.linewidth": 1.1,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def style_ax(ax, xmax=1.0):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def seg_text_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#ffffff"


# ---------------------------------------------------------------- Chart A: error decomposition
DECOMP = {  # class: (matched, merged, missed-no-overlap, missed-poor-shape, split)
    "Neoplastic":   (.811, .048, .092, .040, .009),
    "Inflammatory": (.849, .046, .079, .023, .004),
    "Connective":   (.752, .038, .153, .044, .013),
    "Dead":         (.470, .089, .385, .048, .008),
    "Epithelial":   (.837, .042, .075, .036, .011),
}
DECOMP_LABELS = ["matched", "merged", "missed (no overlap)", "missed (poor shape)", "split"]
DECOMP_COLORS = [C1, C2, C3, C4, C5]


def decomp_chart(path, classes, groups, class_display=None):
    """groups: list of (display_label, color, [source indices summed])."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = range(len(classes))[::-1]
    for row, cls in zip(y, classes):
        vals = DECOMP[cls]
        total = sum(vals)
        left = 0.0
        for lbl, colr, src in groups:
            v = sum(vals[i] for i in src) / total
            ax.barh(row, v, left=left, height=0.62, color=colr,
                    edgecolor="white", linewidth=2.0)
            if v >= 0.055:  # direct-label segments wide enough to hold text
                ax.text(left + v / 2, row, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=10, color=seg_text_color(colr))
            left += v
    names = [class_display.get(c, c) if class_display else c for c in classes]
    ax.set_yticks(list(y), names)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("凋亡") or lbl.get_text() == "Dead":
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, .25, .5, .75, 1.0],
                  ["0%", "25%", "50%", "75%", "100%"])
    style_ax(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=colr, ec="none")
                       for _, colr, _ in groups],
              labels=[lbl for lbl, _, _ in groups], ncol=len(groups), loc="lower left",
              bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10.5,
              handlelength=1.6, handleheight=1.6, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- Chart B: axis decomposition (2 panels)
ARCHS = ["CellViT-UNI", "HoVer-NeXt-T", "HoVer-Net"]  # ascending real mPQ drop
REAL_FD, REAL_TYP = (.007, .045, .163), (.088, .085, .069)
CF_FD, CF_TYP = (.102, .090, .084), (.082, .064, .034)


def axes_chart(path, arch_names, panel_titles, ser_labels):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    x = range(len(ARCHS))
    w = 0.34
    for ax, fd, typ, title in zip(axs, [REAL_FD, CF_FD], [REAL_TYP, CF_TYP], panel_titles):
        ax.bar([i - w / 2 for i in x], fd, width=w, color=C1, edgecolor="white", linewidth=2.0)
        ax.bar([i + w / 2 for i in x], typ, width=w, color=C2, edgecolor="white", linewidth=2.0)
        for i, v in enumerate(fd):  # label the detection bars only (the story)
            ax.text(i - w / 2, v + .007, f"{v:.3f}".lstrip("0"), ha="center",
                    fontsize=10, color=INK2)
        ax.set_title(title, fontsize=11.5, color=INK2, loc="left", pad=8)
        ax.set_ylim(0, 0.20)
        style_ax(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.9)
        ax.xaxis.grid(False)
        ax.set_xticks(list(x), arch_names)
    axs[0].set_yticks([0, .05, .10, .15, .20], ["0", ".05", ".10", ".15", ".20"])
    fig.suptitle("")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in (C1, C2)]
    fig.legend(handles, ser_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               frameon=False, fontsize=10.5, handlelength=1.6, handleheight=1.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- Chart C: Dead PQ, in-domain vs PUMA
DEAD_IN, DEAD_PUMA = (.176, .136, .127), (.002, .005, .033)


def dumbbell_chart(path, arch_names, ser_labels, in_vals, puma_vals):
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    y = list(range(len(arch_names)))[::-1]
    for row, iv, pv in zip(y, in_vals, puma_vals):
        ax.plot([pv, iv], [row, row], color=MUTED, linewidth=2.8, zorder=1)
        ax.scatter([iv], [row], s=90, color=C1, zorder=2)
        ax.scatter([pv], [row], s=90, color=C2, zorder=2)
        halo = [pe.withStroke(linewidth=3.5, foreground="white")]
        ax.text(iv + .008, row, f"{iv:.3f}".lstrip("0"), va="center", fontsize=10.5,
                color=INK2, path_effects=halo, zorder=3)
        ax.text(pv + .008, row, f"{pv:.3f}".lstrip("0"), va="center", fontsize=10.5,
                color=INK2, path_effects=halo, zorder=3)
    ax.set_yticks(y, arch_names)
    ax.set_xlim(0, 0.21)
    ax.set_xticks([0, .05, .10, .15, .20], ["0", ".05", ".10", ".15", ".20"])
    style_ax(ax)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, markersize=10)
                       for c in (C1, C2)],
              labels=ser_labels, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02),
              frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    # -- stage report (English labels; both language versions share them)
    en_groups = [("matched", C1, [0]), ("merged", C2, [1]),
                 ("missed (no overlap)", C3, [2]), ("miss (shape)", C4, [3]),
                 ("split", C5, [4])]
    decomp_chart(FIGS / "chart_decomp.png", list(DECOMP), en_groups)
    axes_chart(FIGS / "chart_axes.png", ARCHS,
               ["Real transfer drop", "CF context-swap drop"],
               ["detection ($F_d$ drop)", "typing (gap increase)"])
    dumbbell_chart(FIGS / "chart_dead_puma.png", ARCHS,
                   ["in domain (PanNuke)", "cross domain (PUMA)"], DEAD_IN, DEAD_PUMA)
    # -- plain-language summary (Chinese labels, simplified categories)
    zh_classes = {"Neoplastic": "肿瘤核", "Inflammatory": "炎症核", "Connective": "结缔核",
                  "Dead": "凋亡核（Dead）", "Epithelial": "上皮核"}
    zh_groups = [("检出", C1, [0]), ("完全漏检", C2, [2]), ("其它错误", C3, [1, 3, 4])]
    decomp_chart(FIGS / "chart_decomp_zh.png", list(DECOMP), zh_groups,
                 class_display=zh_classes)
    dumbbell_chart(FIGS / "chart_dead_puma_zh.png", ARCHS,
                   ["域内（PanNuke）", "跨域（黑色素瘤 PUMA）"], DEAD_IN, DEAD_PUMA)
    print("wrote:", *[p.name for p in sorted(FIGS.glob("chart_*.png"))], sep="\n  ")


if __name__ == "__main__":
    main()
