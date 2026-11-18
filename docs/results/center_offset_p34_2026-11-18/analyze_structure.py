"""Structure analysis behind P34: exclusive miss classes, localization and loss theory.

Inputs are the published D32 object table and the D32 per-point tables. Outputs
are one JSON file and four PNG figures. No model is run.

    python analyze_structure.py --objects <objects.csv> --points <records dir> --out <dir>
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

CLASSES = ('matched', 'near_center_unassigned', 'displaced_in_mask', 'weak_coarse_positive', 'no_response')
LABELS = {'matched': 'Matched within 3 px',
          'near_center_unassigned': 'Score >= 0.5 within 3 px, not assigned',
          'displaced_in_mask': 'Score >= 0.5 inside nucleus, beyond 3 px',
          'weak_coarse_positive': 'Below 0.5 inside nucleus, 128-grid positive nearby',
          'no_response': 'Below 0.5 everywhere near the nucleus'}
# Fixed categorical order of the validated default palette (blue, orange, aqua, yellow, magenta).
COLORS = dict(zip(CLASSES, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')))
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'


def classify(row):
    if row['matched'] == 'True':
        return 'matched'
    if float(row['max_logit_r3']) >= 0:
        return 'near_center_unassigned'
    if float(row['mask_max_logit']) >= 0:
        return 'displaced_in_mask'
    if float(row['coarse_support_max_logit_r3']) >= 0:
        return 'weak_coarse_positive'
    return 'no_response'


def optimal_center_score(sigma, radius=12):
    """Score minimizing the expected Gaussian-weighted focal loss at the most likely pixel.

    The annotated center is uncertain to the model with an isotropic Gaussian
    spread sigma (pixels, discretized). At the central pixel the positive term
    has probability q; other center positions make this pixel a negative with
    the training weight (1 - exp(-d^2/2))^4 inside the 7x7 window and 1 outside.
    """
    d = np.arange(-radius, radius + 1)
    y, x = np.meshgrid(d, d, indexing='ij')
    prior = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    prior /= prior.sum()
    kernel = np.where((abs(x) <= 3) & (abs(y) <= 3), np.exp(-(x * x + y * y) / 2), 0.)
    q = prior[radius, radius]
    negative = float((prior * (1 - kernel) ** 4).sum())   # the center itself contributes zero
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(200):                                   # derivative is monotone in p
        p = (lo + hi) / 2
        grad = q * (2 * (1 - p) * math.log(p) - (1 - p) ** 2 / p) + negative * (-2 * p * math.log(1 - p) + p * p / (1 - p))
        lo, hi = (p, hi) if grad < 0 else (lo, p)
    within3 = float(prior[(x * x + y * y) <= 9].sum())
    return dict(sigma=float(sigma), center_probability=float(q), negative_weight=negative,
                optimal_score=float(p), point_estimate_within3=within3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', required=True)
    parser.add_argument('--points', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.objects, encoding='utf-8')))
    models = sorted({r['model'] for r in rows}, key=lambda m: (m[:3], m[4] != 'C', m))
    result = dict(scope='145 interior Dead nuclei in eleven development images and 656 in the selected '
                        'training images, each observed by twelve completed models', classes=LABELS)

    # 1. exclusive classes
    table = {role: {m: collections.Counter() for m in models} for role in ('development', 'training')}
    for r in rows:
        table[r['role']][r['model']][classify(r)] += 1
    result['exclusive_classes'] = {role: dict(per_model={m: {c: v[c] for c in CLASSES} for m, v in t.items()},
                                              total={c: sum(v[c] for v in t.values()) for c in CLASSES})
                                   for role, t in table.items()}
    dev = [r for r in rows if r['role'] == 'development']
    train = [r for r in rows if r['role'] == 'training']

    # 2. localization distances
    def nearest(group):
        return np.asarray([float(r['nearest_in_mask_point_px']) for r in group
                           if r['matched'] == 'True' and r['nearest_in_mask_point_px'] not in ('', 'None')])
    dist = dict(development=nearest(dev), training=nearest(train))
    displaced = np.asarray([float(r['mask_max_distance_px']) for r in dev if classify(r) == 'displaced_in_mask'])
    result['matched_point_distance'] = {k: dict(n=int(len(v)), median=float(np.median(v)),
        share_within_1px=float((v <= 1).mean()), share_between_2_and_3px=float(((v > 2) & (v <= 3)).mean()))
        for k, v in dist.items()}
    result['displaced_peak_distance'] = dict(n=int(len(displaced)), median=float(np.median(displaced)),
        share_3_to_4px=float((displaced < 4).mean()), share_below_6px=float((displaced < 6).mean()),
        maximum=float(displaced.max()),
        with_retained_point_inside_nucleus=int(sum(float(r['in_mask_displaced_points']) >= 1 for r in dev
                                                 if classify(r) == 'displaced_in_mask')))

    # 3. per-object consistency
    per = collections.defaultdict(list)
    for r in dev:
        per[(r['image'], r['gt_index'])].append(r)
    hits = {k: sum(r['matched'] == 'True' for r in v) for k, v in per.items()}
    images = collections.Counter(k[0] for k in per)
    top = max(images, key=images.get)
    never = [k for k, v in hits.items() if v == 0]
    result['object_consistency'] = dict(objects=len(per), never_matched=len(never),
        always_matched=sum(v == 12 for v in hits.values()),
        never_matched_in_largest_image=sum(k[0] == top for k in never), largest_image=int(top),
        objects_in_largest_image=images[top],
        never_matched_class_observations=dict(collections.Counter(classify(r) for k in never for r in per[k])),
        per_image={i: dict(objects=n, matched_rate=sum(hits[k] for k in per if k[0] == i) / (12 * n),
                           never_matched=sum(k[0] == i for k in never)) for i, n in images.items()})
    strata = collections.defaultdict(collections.Counter)
    for r in dev:
        size = 'under 80 px' if float(r['area']) < 80 else '80 to 199 px' if float(r['area']) < 200 else '200 px and above'
        strata[f"{size}; {'neighbor within 12 px' if float(r['neighbors12']) > 0 else 'isolated'}"][classify(r)] += 1
    result['area_neighbor_strata'] = {k: {c: v[c] for c in CLASSES} for k, v in sorted(strata.items())}
    arms = collections.defaultdict(collections.Counter)
    for r in dev:
        arms['last-12 control (6 models)' if '_C_' in r['model'] else r['model'].rsplit('_', 1)[0]][classify(r)] += 1
    result['arm_classes'] = {k: {c: v[c] for c in CLASSES} for k, v in sorted(arms.items())}
    result['upper_bound_if_displaced_recovered'] = {
        m: dict(matched=table['development'][m]['matched'],
                with_displaced=table['development'][m]['matched'] + table['development'][m]['displaced_in_mask'],
                rate=(table['development'][m]['matched'] + table['development'][m]['displaced_in_mask']) / 145)
        for m in models}

    # 4. all predictions on the 522 development images: where the unmatched points are
    extras = {}
    for m in models:
        file = Path(args.points) / f'{m}_points.csv'
        if not file.exists():
            continue
        pts = list(csv.DictReader(open(file, encoding='utf-8')))
        matched_masks = {(p['image'], p['mask_instance_id']) for p in pts
                         if p['matched'] == 'True' and p['inside_mask'] == 'True'}
        c = collections.Counter()
        for p in pts:
            if p['matched'] == 'True':
                c['matched'] += 1
            elif p['inside_mask'] != 'True':
                c['unmatched_outside_nuclei'] += 1
            elif (p['image'], p['mask_instance_id']) in matched_masks:
                c['unmatched_in_nucleus_holding_a_matched_point'] += 1
            else:
                c['unmatched_in_nucleus_without_matched_point'] += 1
        total = sum(c.values())
        extras[m] = dict(points=total, **c, precision=c['matched'] / total,
                         precision_if_in_nucleus_extras_resolved=(c['matched'] + c['unmatched_in_nucleus_without_matched_point'])
                         / (total - c['unmatched_in_nucleus_holding_a_matched_point']))
    result['development_predictions_all_classes'] = dict(
        note='A nucleus counts as holding a matched point when a matched point lies inside its mask; '
             'the last column is an arithmetic upper bound, not a model result.', per_model=extras)

    # 5. theory
    sigmas = np.round(np.arange(.25, 3.01, .05), 2)
    curve = [optimal_center_score(s) for s in sigmas]
    crossing = next(c['sigma'] for c in curve if c['optimal_score'] < .5)
    result['loss_theory'] = dict(
        model='isotropic Gaussian uncertainty of the annotated center; Gaussian-weighted focal loss, sigma 1, radius 3',
        largest_sigma_with_score_at_least_half=float(crossing - .05), curve=curve)
    (out / 'STRUCTURE_ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    # ------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURFACE,
                         'axes.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .8, 'axes.axisbelow': True,
                         'savefig.facecolor': SURFACE})

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw=dict(width_ratios=[1, 1]), sharey=True)
    for ax, role, total in ((axes[0], 'development', 145), (axes[1], 'training', 656)):
        left = np.zeros(len(models))
        for c in CLASSES:
            values = np.asarray([table[role][m][c] for m in models]) / total * 100
            ax.barh(range(len(models)), values, left=left, color=COLORS[c], edgecolor=SURFACE, linewidth=2,
                    height=.72, label=LABELS[c])
            for i, (v, l) in enumerate(zip(values, left)):
                if v >= 6:
                    ax.text(l + v / 2, i, f'{v:.0f}', ha='center', va='center', fontsize=11,
                            color=INK if c == 'weak_coarse_positive' else 'white')
            left += values
        ax.set_yticks(range(len(models)), [m.replace('_', ' ') for m in models])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('Share of interior Dead nuclei (%)')
        ax.set_title(f"{'Development evaluation images' if role == 'development' else 'Selected training images'}"
                     f' ({total} nuclei per model)', loc='left', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(.5, 0))
    fig.suptitle('What happens to each interior Dead nucleus: five exclusive outcomes, twelve completed models',
                 x=.01, ha='left', fontsize=16)
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(out / 'figures' / 'p34_exclusive_outcomes.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
    bins = np.arange(0, 3.26, .25)
    for key, color in (('training', '#2a78d6'), ('development', '#eb6834')):
        axes[0].hist(dist[key], bins=bins, weights=np.full(len(dist[key]), 100 / len(dist[key])), histtype='step',
                     linewidth=2.4, color=color, label=f'{key.capitalize()} images (n = {len(dist[key])})')
    axes[0].set_xlabel('Distance from the annotated center to the matched point (px)')
    axes[0].set_ylabel('Share of matched observations (%)')
    axes[0].set_title('Matched nuclei: training points sit on the center,\ndevelopment points spread toward the 3 px limit',
                      loc='left', fontsize=14)
    axes[0].legend(frameon=False, loc='upper right')
    axes[1].hist(displaced, bins=np.arange(3, 8.51, .5), color='#1baf7a', edgecolor=SURFACE, linewidth=2)
    axes[1].axvline(3, color=MUTED, linewidth=1.5, linestyle='--')
    axes[1].text(6.6, axes[1].get_ylim()[1] * .94, 'dashed line: matching radius 3 px', color=MUTED, fontsize=12)
    axes[1].set_xlabel('Distance from the annotated center to the highest score inside the nucleus (px)')
    axes[1].set_ylabel('Missed observations')
    axes[1].set_title(f'Displaced responses on development images (n = {len(displaced)}):\n'
                      f'{(displaced < 4).mean() * 100:.0f}% lie within one pixel beyond the limit', loc='left', fontsize=14)
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'p34_localization_distance.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.4))
    counts_top = np.bincount([v for k, v in hits.items() if k[0] == top], minlength=13)
    counts_rest = np.bincount([v for k, v in hits.items() if k[0] != top], minlength=13)
    ax.bar(range(13), counts_rest, color='#2a78d6', edgecolor=SURFACE, linewidth=2, width=.8,
           label=f'Ten other development images ({sum(counts_rest)} nuclei)')
    ax.bar(range(13), counts_top, bottom=counts_rest, color='#eb6834', edgecolor=SURFACE, linewidth=2, width=.8,
           label=f'Colon image {top} ({sum(counts_top)} nuclei)')
    for i in range(13):
        if counts_top[i] + counts_rest[i]:
            ax.text(i, counts_top[i] + counts_rest[i] + .5, str(counts_top[i] + counts_rest[i]), ha='center', fontsize=12)
    ax.set_xticks(range(13))
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Number of the twelve models that match the nucleus')
    ax.set_ylabel('Interior Dead nuclei')
    ax.set_title('Most nuclei are either found by every model or by none', loc='left', fontsize=15)
    ax.legend(frameon=False, loc='upper center')
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'p34_object_consistency.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), sharex=True)
    s = [c['sigma'] for c in curve]
    axes[0].plot(s, [c['optimal_score'] for c in curve], color='#2a78d6', linewidth=2.4)
    axes[0].axhline(.5, color=MUTED, linewidth=1.5, linestyle='--')
    axes[0].text(2.98, .52, 'decoder threshold 0.5', ha='right', color=MUTED, fontsize=12)
    axes[0].axvline(crossing - .025, color=MUTED, linewidth=1, linestyle=':')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Loss-optimal score at the most likely center pixel')
    axes[0].set_title(f'Exact-center loss: the optimal peak falls below 0.5\nonce center uncertainty exceeds about {crossing - .05:.2f} px',
                      loc='left', fontsize=14)
    axes[1].plot(s, [c['point_estimate_within3'] * 100 for c in curve], color='#eb6834', linewidth=2.4)
    axes[1].set_ylim(0, 102)
    axes[1].set_ylabel('Chance that the most likely pixel is within 3 px (%)')
    axes[1].set_title('Evaluation rule: a single point at the most likely pixel\nstill lands inside the 3 px matching radius',
                      loc='left', fontsize=14)
    for ax in axes:
        ax.set_xlabel('Uncertainty of the annotated center position, Gaussian sigma (px)')
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'p34_loss_theory.png', dpi=150)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ('matched_point_distance', 'displaced_peak_distance')}, indent=1))
    print(json.dumps(result['exclusive_classes']['development']['total']), json.dumps(result['exclusive_classes']['training']['total']))
    print(json.dumps({k: v for k, v in result['object_consistency'].items() if k != 'per_image'}))
    print(json.dumps(result['arm_classes']))
    print(json.dumps({m: [round(v['precision'], 4), round(v['precision_if_in_nucleus_extras_resolved'], 4), v['points'],
                          v.get('unmatched_in_nucleus_without_matched_point'), v.get('unmatched_in_nucleus_holding_a_matched_point'),
                          v.get('unmatched_outside_nuclei')] for m, v in extras.items()}))
    print('theory', crossing, [(c['sigma'], round(c['optimal_score'], 3), round(c['point_estimate_within3'], 3))
                               for c in curve if c['sigma'] in (.5, 1., 1.5, 2., 2.5, 3.)])


if __name__ == '__main__':
    main()
