"""Reproduce the D32 diagnostic-pattern figure from its published analysis JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    models = json.loads(Path(args.analysis).read_text())['models']
    labels = []
    names = {'C': 'control', 'M': 'full depth', 'B': 'early + late', 'G': 'group normalization'}
    for row in models:
        experiment, arm, seed = row['model'].split('_')
        labels.append(f'{experiment}: {names[arm]} / seed {seed[1:]}')
    fig, axes = plt.subplots(2, 1, figsize=(17, 13), constrained_layout=True)
    y = np.arange(len(models))
    left = np.zeros(len(models))
    categories = [('coarse_and_fine_negative', 'Coarse support and fine center region below threshold', '#6b7280'),
                  ('coarse_positive_fine_negative', 'Coarse support positive; fine center region below threshold', '#e69526'),
                  ('fine_positive_unmatched', 'Fine center region positive; object unmatched', '#6555a1')]
    for field, label, color in categories:
        values = np.asarray([m[field] for m in models])
        axes[0].barh(y, values, left=left, label=label, color=color)
        left += values
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.10), fontsize=10)
    axes[0].set_xlabel('Missed interior Dead object observations per model (145 annotated objects)')
    axes[0].set_title('Measured response patterns among missed nuclei', fontsize=17)
    axes[1].barh(y, [m['displaced_in_mask'] for m in models], color='#269d83')
    axes[1].set_xlabel('Misses accompanied by a retained prediction inside the same nucleus beyond radius 3')
    axes[1].set_title('Localization flag; overlaps with the response categories above', fontsize=17)
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
