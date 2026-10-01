"""Report figures for stage report II (+ plain-language summary), from verified numbers.

Sources (every value re-derived from artifacts 2026-10-01; see runs/analysis/report2_numbers.txt):
- Size-binned match-rate deltas (split-1 TTA, x2 seed19 minus 3-seed base mean): [E].
- du2 recovery (3-split seed-19 means, noTTA): [B].
- 3x3 seed grid per-run values: scripts/x2_grid_sum.py output (re-derived identical).

Style: same dataviz reference categorical slots 1-5 (validated light/white) as stage
report I; series colors = fixed slot order, all text in ink tokens.
Output: docs/report/stage_report_2/figs/*.png (300 dpi; _zh variants with CJK labels).
Run: ~/.conda/envs/nuclei/bin/python3 scripts/make_report_figures2.py
"""
from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

FIGS = Path(__file__).resolve().parent.parent / "docs" / "report" / "stage_report_2" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# dataviz reference palette (light mode, validated on white) -- same as stage report I
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

for p in [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]:
    if Path(p).exists():
        fontManager.addfont(p)

mpl.rcParams.update({
    # Per-glyph fallback fires when font.family is the direct list in mpl 3.7
    "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
    "font.size": 14,
    "axes.edgecolor": BASE,
    "axes.linewidth": 1.2,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------- fig 1: size bins
# x2 - base match-rate delta by GT-area bin (split-1 TTA). Sign is the story: blue gain,
# orange loss; the Dead bar (slot 3) is a distinct entity, labeled directly on the axis.
BINS = ["0–200", "200–400", "400–800", "800–1600", "1600–3200", "3200–6400"]
DR = [0.104, -0.010, -0.020, -0.029, -0.070, -0.115]     # nonDead, x2 - base (split1 TTA)
DEAD_DR = 0.091                                           # Dead 0-200 (n=858)

def fig_sizebins(suffix, title, xlab, ylab, dead_label):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xs = list(range(len(BINS)))
    colors = [C1 if v >= 0 else C2 for v in DR]
    ax.bar(xs, DR, width=0.62, color=colors, zorder=3)
    dx = len(BINS) + 0.9
    ax.bar([dx], [DEAD_DR], width=0.62, color=C3, zorder=3)
    for x, v in zip(xs, DR):
        ax.annotate(f"{v:+.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 5 if v >= 0 else -16), ha="center", fontsize=12.5, color=INK2)
    ax.annotate(f"{DEAD_DR:+.3f}", (dx, DEAD_DR), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=12.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1.2, zorder=4)
    ax.set_xticks(xs + [dx])
    ax.set_xticklabels(BINS + [dead_label])
    ax.set_xlim(-0.7, dx + 0.7)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title, fontsize=15, color=INK, pad=10)
    ax.set_ylim(-0.145, 0.145)
    fig.savefig(FIGS / f"chart_size_bins{suffix}.png")
    plt.close(fig)

fig_sizebins("", "Working resolution 2×: match-rate change by nucleus size",
             "GT nucleus area (px, 256-frame)", "Δ share of GT nuclei matched (IoU ≥ 0.5)",
             "Dead\n0–200")
fig_sizebins("_zh", "工作分辨率 2 倍：不同尺寸核的检出率变化",
             "真值核面积（256 像素坐标系）", "匹配率变化（IoU ≥ 0.5 的真值核占比）",
             "凋亡核\n0–200")

# ---------------------------------------------------------------- fig 2: du2 recovery
# 3-split seed-19 means, noTTA (report2_numbers.txt [B]). Arms = categorical slots 1-3.
METRICS = ["mPQ", "bPQ", "mPQ+", "Dead PQ"]
ARMS = ["base (256)", "x2 (512)", "x2 + du2"]
VALS = {
    "base (256)": [0.4995, 0.6653, 0.5178, 0.1760],
    "x2 (512)":   [0.4791, 0.6413, 0.5012, 0.1816],
    "x2 + du2":   [0.4894, 0.6583, 0.5157, 0.1831],
}
RECOVER = [50, 71, 87]  # % of the x2-vs-base gap recovered by du2 (mPQ/bPQ/mPQ+)

def fig_du2(suffix, title, ylab, rec_word):
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    xs = range(len(METRICS))
    w = 0.26
    for i, (arm, color) in enumerate(zip(ARMS, [C1, C2, C3])):
        off = (i - 1) * w
        ax.bar([x + off for x in xs], VALS[arm], width=w * 0.92, color=color, label=arm, zorder=3)
    for j in range(3):  # recovery note above the du2 bar (text in ink, not series color)
        v = VALS[ARMS[2]][j]
        ax.annotate(f"{rec_word}{RECOVER[j]}%", (j + w, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=12.5, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0.0, 0.78)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=15, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=12.5, loc="upper right")
    fig.savefig(FIGS / f"chart_du2_recovery{suffix}.png")
    plt.close(fig)

fig_du2("", "Scaling the decode constants recovers most of the 2× cost",
        "3-split mean PQ (no TTA)", "+")
fig_du2("_zh", "按分辨率缩放解码常数，收回 2 倍分辨率的大部分代价",
        "三折平均 PQ（无 TTA）", "收回 ")

# ---------------------------------------------------------------- fig 3: seed grid
# per-run test values from x2_grid_sum.py. Base extra seeds exist on split 1 only.
GRID = {
    1: {"base": [0.1424, 0.1308, 0.1446], "du2": [0.1606, 0.1392, 0.1515],
        "base_m": [0.4951, 0.4910, 0.4924], "du2_m": [0.4861, 0.4825, 0.4854]},
    2: {"base": [0.1621], "du2": [0.1683, 0.1574, 0.1560],
        "base_m": [0.4930], "du2_m": [0.4856, 0.4846, 0.4839]},
    3: {"base": [0.2236], "du2": [0.2204, 0.2415, 0.2241],
        "base_m": [0.5104], "du2_m": [0.4964, 0.4987, 0.4973]},
}
SPLIT_LBL = {"": ["split 1", "split 2", "split 3"], "_zh": ["折 1", "折 2", "折 3"]}

def fig_grid(suffix, titles, ylabs, legend_labels):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, key, title, ylab in zip(axes, ["dead", "m"], titles, ylabs):
        for i, s in enumerate((1, 2, 3)):
            g = GRID[s]
            bv = g["base"] if key == "dead" else g["base_m"]
            dv = g["du2"] if key == "dead" else g["du2_m"]
            x0 = i - 0.13
            ax.scatter([x0] * len(bv), bv, s=52, color=C1, zorder=3,
                       label=legend_labels[0] if (i == 0 and key == "dead") else None)
            ax.scatter([i + 0.13] * len(dv), dv, s=52, color=C2, zorder=3,
                       label=legend_labels[1] if (i == 0 and key == "dead") else None)
            bm = sum(bv) / len(bv); dm = sum(dv) / len(dv)
            ax.plot([x0, i + 0.13], [bm, dm], color=BASE, lw=1.6, zorder=2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(SPLIT_LBL[suffix])
        ax.set_title(title, fontsize=14.5, color=INK, pad=8)
        ax.set_ylabel(ylab)
        if key == "dead":
            ax.set_ylim(0.10, 0.26)
        else:
            ax.set_ylim(0.475, 0.515)
    axes[0].legend(frameon=False, fontsize=12.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / f"chart_seed_grid{suffix}.png")
    plt.close(fig)

fig_grid("", ["Dead PQ by split and seed", "mPQ by split and seed"],
         ["Dead PQ (test fold)", "mPQ (test fold)"], ["base 256", "x2 + du2"])
fig_grid("_zh", ["凋亡核（Dead）PQ：逐折逐种子", "整体 mPQ：逐折逐种子"],
         ["Dead PQ（测试折）", "mPQ（测试折）"], ["基线 256", "x2 + du2"])

# ---------------------------------------------------------------- copy montages (self-contained figs/)
for name in ("x2_montage_gain.png", "x2_montage_damage.png", "x2_montage_fragments.png"):
    src = Path("runs/analysis") / name
    if src.exists():
        shutil.copy(src, FIGS / name)
        print("copied", name)
print("figures written to", FIGS)
