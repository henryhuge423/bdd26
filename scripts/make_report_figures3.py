"""Figures for stage report III, derived from frozen matched-seed statistics.

Run from the repository root:
    python scripts/make_report_figures3.py

No experiment selection or model evaluation is performed here. Numerical table
counterparts are saved alongside the figures for inspection and reuse.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


COLORS = {"P2": "#2a78d6", "EF-P2": "#eb6834"}
INK = "#252b32"
GRID = "#dfe3e8"


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK, pad=6)
    ax.axvline(0, color="#929ba5", linewidth=0.9, zorder=1)


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("runs/analysis"))
    parser.add_argument("--out", type=Path, default=Path("docs/report/stage_report_3/figs"))
    parser.add_argument("--zh-font", type=Path,
                        default=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
                        help="CJK TrueType font; OpenType/CFF is incompatible with Type 42 embedding")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {"P2": args.analysis_root / "matched_seed_20261004" / "stats",
             "EF-P2": args.analysis_root / "existence_ef_20261005" / "stats"}
    stats = {arm: read_json(root / "stats.json") for arm, root in roots.items()}
    pairs = [f"split{split}_seed{seed}" for split in (1, 2, 3) for seed in (19, 1, 2)]
    for arm, data in stats.items():
        if set(data["pairs"]) != set(pairs):
            raise ValueError(f"{arm}: expected nine distinct split/seed pairs")
    for pair in pairs:
        if stats["P2"]["pairs"][pair]["baseline"] != stats["EF-P2"]["pairs"][pair]["baseline"]:
            raise ValueError(f"Different baselines: {pair}")

    rows = []
    for pair in pairs:
        for arm in roots:
            rows.append({"pair": pair, "arm": arm,
                         **{ep: stats[arm]["pairs"][pair]["delta"][ep]
                            for ep in ("mPQ", "bPQ", "Dead PQ")}})
    with (args.out / "paired_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "arm", "mPQ", "bPQ", "Dead PQ"])
        writer.writeheader()
        writer.writerows(rows)

    ci_rows = []
    for arm, root in roots.items():
        for split in (1, 2, 3):
            boot = read_json(root / f"bootstrap_split{split}.json")
            for ep in ("mPQ", "bPQ", "Dead PQ"):
                ci_rows.append({"arm": arm, "split": split, "metric": ep,
                                "point": boot[ep]["point"],
                                "low": boot[ep]["ci95"][0], "high": boot[ep]["ci95"][1]})
    with (args.out / "split_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "split", "metric", "point", "low", "high"])
        writer.writeheader()
        writer.writerows(ci_rows)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for lang in ("en", "zh"):
        if lang == "zh":
            if not args.zh_font.is_file():
                raise FileNotFoundError("Chinese font missing; pass --zh-font with a local font file")
            font_manager.fontManager.addfont(str(args.zh_font))
            # Keep Latin/Greek glyphs in DejaVu, with a TrueType CJK fallback.
            plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(
                fname=str(args.zh_font)).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
        titles = ["Dead PQ", "bPQ"]
        for ax, ep in zip(axes, titles):
            style_axis(ax)
            for arm, offset, marker in (("P2", -0.13, "o"), ("EF-P2", 0.13, "D")):
                values = [stats[arm]["pairs"][pair]["delta"][ep] * 100 for pair in pairs]
                ax.scatter(values, np.arange(9) + offset, s=47, color=COLORS[arm],
                           marker=marker, edgecolor="white", linewidth=0.9, label=arm, zorder=3)
            for y in (2.5, 5.5):
                ax.axhline(y, color=GRID, linewidth=0.6)
            ax.set_title(ep, loc="left", pad=12)
            ax.set_xlabel("相对同种子 x1+M/S 的变化（分，Δ×100）" if lang == "zh"
                          else "Change from matched x1+M/S (points, delta x 100)")
        axes[0].set_yticks(range(9))
        axes[0].set_yticklabels([f"S{split} / seed {seed}" for split in (1, 2, 3) for seed in (19, 1, 2)])
        axes[0].invert_yaxis()
        axes[1].legend(frameon=False, loc="upper right", fontsize=10)
        fig.tight_layout(w_pad=2.2)
        save_figure(fig, args.out, f"paired_gains_{lang}")

        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=True)
        for ax, ep in zip(axes, ("mPQ", "bPQ", "Dead PQ")):
            style_axis(ax)
            for arm, offset, marker in (("P2", -0.12, "o"), ("EF-P2", 0.12, "D")):
                entries = [r for r in ci_rows if r["arm"] == arm and r["metric"] == ep]
                pts = np.array([r["point"] for r in entries]) * 100
                low = np.array([r["low"] for r in entries]) * 100
                high = np.array([r["high"] for r in entries]) * 100
                ax.errorbar(pts, np.arange(3) + offset, xerr=np.array([pts-low, high-pts]),
                            fmt=marker, markersize=6.5, markeredgecolor="white",
                            markeredgewidth=0.8, color=COLORS[arm], elinewidth=1.2,
                            capsize=3, label=arm, zorder=3)
            ax.set_title(ep, loc="left", pad=12)
            ax.set_xlabel("变化（分，Δ×100）" if lang == "zh" else "Change (points, delta x 100)")
        axes[0].set_yticks(range(3))
        axes[0].set_yticklabels(["split 1", "split 2", "split 3"])
        axes[0].invert_yaxis()
        axes[2].legend(frameon=False, loc="upper right", fontsize=9)
        fig.tight_layout(w_pad=2)
        save_figure(fig, args.out, f"split_intervals_{lang}")
    print(f"Wrote paired figures, split intervals and numerical CSVs to {args.out}")


if __name__ == "__main__":
    main()
