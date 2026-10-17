"""Plot fixed-batch readout fitting and audit endpoints; no generalization claim."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise ValueError('Refusing existing figure')
r = json.loads(a.input.read_text())['results']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
for ax, key, title, ymax in zip(axes, ('foreground_accuracy', 'best_final_recall50'),
                              ('A  Fixed-assignment foreground classification',
                               'B  Best final-instance recall (IoU > .5)'), (105, 65)):
    for batch, color, label in (('fit', '#26758D', 'Fit batch: 16 images'),
                                ('audit', '#C1603D', 'Audit batch: 16 images')):
        means = []
        for x, arm in enumerate(('A0', 'A1', 'A2')):
            vals = [100 * r[s][arm][batch][key] for s in ('19', '1')]
            means.append(sum(vals) / 2)
            ax.scatter([x-.035, x+.035], vals, marker='x', color=color, s=24)
        ax.plot(range(3), means, 'o-', lw=2, color=color, label=label)
        ax.annotate(f'{means[-1]:.1f}%', (2, means[-1]), xytext=(7, 1),
                    textcoords='offset points', color=color)
    ax.set(xticks=range(3), xticklabels=['A0\nFrozen', 'A1\nOriginal CE', 'A2\nBalanced CE'],
           ylim=(0, ymax), xlim=(-.2, 2.4), ylabel='Percent', title=title)
    ax.grid(axis='y', alpha=.2)
axes[0].legend(frameon=False, loc='upper left', fontsize=9)
fig.suptitle('The classifier fits one batch; the improvement does not transfer reliably', fontsize=13)
fig.supxlabel('Two fixed model seeds (crosses); fold1 only; 331 fit / 364 audit targets; no Dead targets.', fontsize=9)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=200)
plt.close(fig)
