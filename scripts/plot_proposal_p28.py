"""Reproduce P28 figures from preserved images and matching coordinates."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


import argparse
parser=argparse.ArgumentParser(description="Plot verified P28 fixed objects and paired stage changes")
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--prior-stage-pairs',type=Path,required=True)
parser.add_argument('--p27-analysis',type=Path,required=True)
args=parser.parse_args()
out=args.root
raw = out / "p28_files"
figures = out / "figures"
figures.mkdir(exist_ok=True)
analysis = json.loads((out / "p28_analysis.json").read_text())
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(figure, name):
    figure.savefig(figures / name, dpi=175, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def records(arm, seed):
    path = raw / f"A_{arm}_s{seed}" / "point_examples_audit" / "coordinates.json"
    return {record["image"]: record for record in json.loads(path.read_text())["records"]}


def match_map(record):
    pairs = dict(record["unique_pairs"])
    assert len(pairs) == len(record["unique_pairs"]) == record["score"]["matched"]
    assert all(math.dist(record["gt"][g][1], record["points"][p]) <= 3 for g, p in pairs.items())
    return pairs


colors = {"gained": "#65ffaf", "lost": "#ffb947", "retained": "#fffbe3", "missed_both": "#ff6191"}
states = {(True, True): "retained", (False, True): "gained",
          (True, False): "lost", (False, False): "missed_both"}
objects = []
for seed in (19, 1):
    c, e = records("C", seed), records("I", seed)
    assert c.keys() == e.keys()
    for image, left in c.items():
        right = e[image]
        assert left["gt"] == right["gt"]
        lc, re = match_map(left), match_map(right)
        for index, (klass, (y, x), interior, area) in enumerate(left["gt"]):
            if klass != 4 or not interior:
                continue
            row = dict(seed=seed, image=image, tissue=left["tissue"], index=index, y=y, x=x,
                       area=area, state=states[index in lc, index in re])
            for arm, record, pairs in [("c", left, lc), ("i", right, re)]:
                row[f"{arm}_matched_yx"] = record["points"][pairs[index]] if index in pairs else None
                row[f"{arm}_nearest_px"] = min(
                    (math.dist([y, x], p) for p in record["points"]), default=None
                )
            objects.append(row)
    assert sum(row["seed"] == seed for row in objects) == 53

with (out / "fixed_objects.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(objects[0]))
    writer.writeheader()
    writer.writerows(objects)

summary = {}
for seed in (19, 1):
    rows = [row for row in objects if row["seed"] == seed]
    pair = analysis["pairs"][str(seed)]
    changes = pair["dead_images"]
    net = sum(row["net"] for row in changes)
    leave_one = [(r["image"], (net-r["net"])/(145-r["dead_total"])) for r in changes]
    summary[str(seed)] = dict(fixed=dict(Counter(r["state"] for r in rows)),
                              leave_one_min=min(leave_one, key=lambda r:r[1]),
                              leave_one_max=max(leave_one, key=lambda r:r[1]))
    for image in [776, 1037]:
        examples = [row for row in rows if row["image"] == image]
        if not examples:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True)
        for axis, arm, title in zip(axes, ("C", "I"), ("Control · last 12 blocks", "Intervention · odd-index 12 blocks")):
            record = records(arm, seed)[image]
            path = raw / f"A_{arm}_s{seed}" / "point_examples_audit" / record["png"]
            with Image.open(path) as panel:
                assert panel.size == (1024, 310)
                axis.imshow(np.asarray(panel.crop((0, 42, 256, 298))))
            if record["points"]:
                y, x = np.asarray(record["points"]).T
                axis.scatter(x, y, marker="+", s=40, color="#6deaff", linewidths=1.2)
            for row in examples:
                axis.add_patch(plt.Circle((row["x"], row["y"]), 4, fill=False,
                                          color=colors[row["state"]], linewidth=1.5))
                if row["state"] in ("gained", "lost"):
                    axis.annotate(str(row["index"]), (row["x"], row["y"]), xytext=(5, 5),
                                  textcoords="offset points", color=colors[row["state"]], fontsize=10,
                                  bbox=dict(facecolor="#292333", alpha=.85, edgecolor="none", pad=1))
            axis.set_title(title)
            axis.set_axis_off()
        handles = [plt.Line2D([], [], marker="+", color="#6deaff", linestyle="none", label="Predicted center")]
        handles += [plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgewidth=2,
                               color=colors[state], linestyle="none", label=label)
                    for state, label in [("retained", "Both match"), ("gained", "Interleaved only"),
                                         ("lost", "Control only"), ("missed_both", "Both miss")]]
        figure.legend(handles=handles, loc="outside lower center", ncols=5, frameon=True,
                      facecolor="#454052", labelcolor="white")
        figure.suptitle(f'{examples[0]["tissue"]} image {image} · seed {seed} · annotated Dead nuclei')
        save(figure, f"p28_image_{image}_seed_{seed}.png")
(out / "object_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
for seed, axis in zip((19, 1), axes):
    rows = analysis["pairs"][str(seed)]["dead_images"]
    axis.barh([f'{r["tissue"]} {r["image"]} ({r["dead_total"]})' for r in rows],
              [r["net"] for r in rows], color=["#27788a" if r["net"] >= 0 else "#ae513d" for r in rows])
    axis.invert_yaxis()
    axis.axvline(0, color="#444", linewidth=0.8)
    axis.set(title=f"Seed {seed}", xlabel="Interleaved minus last 12 matched nuclei")
    axis.grid(axis="x", alpha=.22)
figure.suptitle("Interior Dead nuclei across the 11 annotated development images")
save(figure, "p28_dead_by_image.png")

pairs = list(csv.DictReader(args.prior_stage_pairs.open()))
prior = json.loads(args.p27_analysis.read_text())
for seed in (19,1):
    c,e=[prior['runs'][f'{arm}_{seed}']['audit'] for arm in ('C','E')]
    pairs.append(dict(pair=f'P27_{seed}', net_dead=e['dead_hit']-c['dead_hit'],net_all=e['matched']-c['matched'],net_extra=(e['candidates']-e['matched'])-(c['candidates']-c['matched'])))
for seed in (19, 1):
    c, e = [analysis["runs"][f"{arm}_{seed}"]["audit"] for arm in ("C", "I")]
    pairs.append(dict(pair=f"P28_{seed}", net_dead=e["dead_hit"]-c["dead_hit"],
                      net_all=e["matched"]-c["matched"],
                      net_extra=(e["candidates"]-e["matched"])-(c["candidates"]-c["matched"])))
labels = ["Input scale", "Class supervision", "All 24 blocks", "Parameter averaging",
          "Higher encoder rate", "Layerwise rates", "Weight anchor", "First 12 blocks", "Odd-index 12 blocks"]
figure, axes = plt.subplots(1, 3, figsize=(16, 6.4), constrained_layout=True)
for axis, key, title in zip(axes, ("net_dead", "net_all", "net_extra"),
                            ("Matched interior Dead", "All matched nuclei", "Unmatched predictions")):
    axis.axvline(0, color="#777", linewidth=0.8)
    for seed, marker, color, shift in [(19, "o", "#156b82", -.13), (1, "D", "#a94525", .13)]:
        values = [int(r[key]) for r in pairs if r["pair"].endswith(f"_{seed}")]
        assert len(values) == len(labels)
        axis.scatter(values, np.arange(len(labels)) + shift, color=color, marker=marker, label=f"Seed {seed}")
    unit = "Count change · fewer is better" if key == "net_extra" else "Count change · more is better"
    axis.set(yticks=range(len(labels)), yticklabels=labels if axis is axes[0] else [],
             title=title, xlabel=unit)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.22)
axes[-1].legend(frameon=False)
save(figure, "stage_comparison.png")
with (out / 'stage_pairs.csv').open('w', newline='', encoding='utf-8') as file:
    writer=csv.DictWriter(file,fieldnames=['pair','net_dead','net_all','net_extra']);writer.writeheader();writer.writerows(pairs)
