"""Plot descriptive E6/E7 validation means and fixed-train-batch transitions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.with_suffix(".png").exists() or a.out.with_suffix(".svg").exists():
        raise ValueError("Refusing to overwrite an existing figure")
    s = json.loads(a.summary.read_text())
    t = json.loads(a.attribution.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    colors = {"B": "#246A85", "Q": "#C85F36"}
    for ax, metric, title in zip(axes[:2], ("mPQ", "bPQ"),
                                ("A  Validation mPQ", "B  Validation bPQ")):
        for arm in ("B", "Q"):
            means = [s["means"][r][arm][metric] for r in ("E6", "E7")]
            ax.plot([0, 1], means, "o-", color=colors[arm], lw=2,
                    label="Dense B" if arm == "B" else "Query Q")
            for x, rnd in enumerate(("E6", "E7")):
                vals = [s[rnd][f"{arm}_seed{seed}"][metric] for seed in (19, 1)]
                ax.scatter([x - .035, x + .035], vals, color=colors[arm],
                           s=22, marker="x", zorder=3)
            ax.annotate(f"{means[-1]:.3f}", (1, means[-1]), xytext=(6, 5),
                        textcoords="offset points", color=colors[arm])
        ax.set(xticks=[0, 1], xticklabels=["E6: gamma .85", "E7: gamma .99"],
               title=title, xlim=(-.2, 1.35), ylim=(0, .55 if metric == "mPQ" else .73))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, loc="center left")
    ax = axes[2]
    for seed, color in (("19", "#8162A2"), ("1", "#45856C")):
        c = t["results"][seed]["counts"]
        vals = [c["raw_good"], c["raw_good_accepted"],
                c["raw_good_accepted"] - c["raw_good_accepted_lost_after_competition"],
                c["decoded_good"]]
        y = [100 * n / c["targets"] for n in vals]
        ax.plot(range(4), y, "o-", color=color, lw=2, label=f"E7 Q seed{seed}")
        for x, v in enumerate(y):
            ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 7 if seed == "1" else -14),
                        textcoords="offset points", ha="center", color=color, fontsize=8)
    ax.set(title="C  Assigned masks on fixed training batch", ylim=(0, 75),
           xticks=range(4), xticklabels=["Raw\nmask", "Query\nretained", "After\ncompetition", "Final\n(with gains)"],
           ylabel="Targets with paired IoU > .5 (%)")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Slower decay does not rescue this query recipe; output conversion loses usable masks",
                 fontsize=13)
    fig.supxlabel("A/B: reused fold2, last checkpoints, two fixed seeds (crosses); C: 16 fold1 images, 331 targets, no training updates.", fontsize=9)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".png"), dpi=200)
    fig.savefig(a.out.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
