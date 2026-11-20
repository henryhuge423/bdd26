"""P36 result analysis from the collected records of the eight frozen detectors.

Usage: python analyze_p36_results.py <P36 collected results> <output dir>
Writes P36_RESULTS.json and four figures.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

root, out = Path(sys.argv[1]), Path(sys.argv[2])
(out/'figures').mkdir(parents=True, exist_ok=True)
DET = ['P22_C_s19', 'P22_C_s1', 'P22_M_s19', 'P22_M_s1', 'P15_C_s19', 'P15_C_s1', 'P15_J_s19', 'P15_J_s1']
DET = [d for d in DET if (root/d/'summary.json').exists()]
RECIPE = {'P22_C': 'P22 last 12 layers', 'P22_M': 'P22 all 24 layers', 'P15_C': 'P15 last 12 layers',
          'P15_J': 'P15 trained with translation'}
SHORT = {'P22_C': 'P22\nlast 12', 'P22_M': 'P22\nall 24', 'P15_C': 'P15\nlast 12', 'P15_J': 'P15\ntranslation'}
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#52514e'
COLOR = {'P22_C': BLUE, 'P22_M': ORANGE, 'P15_C': AQUA, 'P15_J': YELLOW}
THRESHOLDS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
FILLS = ('white', 'mirror')
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b8b7b0',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'axes.grid': True,
                     'grid.color': '#e6e5df', 'grid.linewidth': .8, 'axes.axisbelow': True})
PERCENT = np.arange(0, 1.01, .2)


def recipe(det):
    return det.rsplit('_s', 1)[0]


def seed(det):
    return int(det.rsplit('_s', 1)[1])


def name(det):
    return f'{RECIPE[recipe(det)]}, seed {seed(det)}'


def rates(matched, points, nuclei):
    return dict(matched=int(matched), points=int(points), recall=matched / nuclei, precision=matched / points if points else None,
                f1=2 * matched / (points + nuclei))


def objects(det, role):
    with (root/det/f'{role}_objects.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    o = dict(cls=a['cls'].astype(int), interior=a['interior'] == 'True', border=a['border'].astype(int),
             valid=a['valid_phases'].astype(int), m0=a['matched_original'] == 'True', mc=a['matched_consensus'] == 'True',
             n12=a['neighbors12'].astype(int))
    for fill in FILLS:
        o[fill] = a[f'shifted_matched_{fill}'].astype(int)
        o['mask_' + fill] = a[f'phases_{fill}'].astype(int)
    o['central'] = (o['valid'] == 16) & (o['border'] >= 24)
    o['dead'] = (o['cls'] == 4) & o['interior']
    return o


result = dict(detectors={}, definition='central = valid at all sixteen alignments and at least 24 px from every border; '
                                       'shifted = mean of the fifteen alignments other than the trained one')
for det in DET:
    s = json.loads((root/det/'summary.json').read_text())
    entry = dict(recipe=RECIPE[recipe(det)], seed=seed(det), consensus_rule=s['rule'], selected_threshold=s['threshold'])
    for role in ('training', 'development'):
        r = s['roles'][role]
        n = r['nuclei']
        o = objects(det, role)
        c = o['central']
        e = dict(images=r['images'], nuclei=n, interior_dead=r['interior_dead'], central=int(c.sum()),
                 decoders={k: dict(v['counts'], recall=v['rates']['recall'], precision=v['rates']['precision'],
                                   dead_recall=v['rates']['dead_recall'],
                                   f1=2 * v['counts']['matched'] / (v['counts']['candidates'] + n))
                           for k, v in r['decoders'].items()},
                 trained_alignment=float(o['m0'][c].mean()),
                 shifted={fill: float(o[fill][c].mean() / 15) for fill in FILLS},
                 border_bands=r['objects']['border_bands'],
                 thresholds=[dict(threshold=t['threshold'], dead_hit=t['counts'][2], **rates(t['counts'][0], t['counts'][1], n))
                             for t in r['thresholds']],
                 shifted_thresholds={fill: [dict(threshold=t, dead_hit=int(x[2]), **rates(x[0], x[1], 15 * n))
                                            for t, x in zip(THRESHOLDS, r['shifted_thresholds'][f])]
                                     for f, fill in enumerate(FILLS)},
                 displacement={fill: dict(isolated_nuclei=r['objects'][fill]['isolated_objects_with_two_or_more_points'],
                                          changes_with_alignment_px2=r['objects'][fill]['alignment_jitter_px2'],
                                          shared_by_alignments_px2=r['objects'][fill]['stable_displacement_px2'])
                               for fill in FILLS},
                 shifted_histogram={fill: r['objects'][fill]['histogram'] for fill in FILLS},
                 stored_original_comparison=r['stored_original_comparison'],
                 forward_seconds_per_image=r['forward_seconds_per_image'])
        best = max(e['thresholds'], key=lambda t: t['f1'])
        e['best_threshold_on_this_role'] = dict(threshold=best['threshold'], f1=best['f1'])
        grid = np.asarray(r['consensus_grid'], float)
        f1 = 2 * grid[..., 0] / (grid[..., 1] + n)
        k = np.unravel_index(f1.argmax(), f1.shape)
        e['best_consensus_on_this_role'] = dict(fill=FILLS[k[0]], radius=[2., 3., 4.][k[1]], votes=int(k[2] + 1), f1=float(f1[k]))
        d = o['dead'] & (o['valid'] == 16)
        e['interior_dead_all_alignments_valid'] = dict(
            nuclei=int(d.sum()), trained_alignment=int(o['m0'][d].sum()), consensus=int(o['mc'][d].sum()),
            any_alignment_white=int(((o['mask_white'][d]) > 0).sum()))
        e['consensus_transitions'] = dict(gained=int((~o['m0'] & o['mc']).sum()), lost=int((o['m0'] & ~o['mc']).sum()),
                                          dead_gained=int((~o['m0'] & o['mc'] & o['dead']).sum()),
                                          dead_lost=int((o['m0'] & ~o['mc'] & o['dead']).sum()))
        entry[role] = e
    result['detectors'][det] = entry
if (root/'summary.json').exists():
    decisions = json.loads((root/'summary.json').read_text())['decisions']
    result['decisions'] = {k: dict(passed=v['pass'], reason=v.get('reason'), seeds={
        sd: dict(delta=d['delta'], ci95=d['ci95'], dead_recall=d['trainable']['dead_recall'], precision=d['trainable']['precision'],
                 recall=d['trainable']['recall'], reference_dead_recall=d['baseline']['dead_recall'],
                 reference_precision=d['baseline']['precision'], reference_recall=d['baseline']['recall'], passed=d['pass_seed'])
        for sd, d in v['details'].items()}) for k, v in decisions.items()}
(out/'P36_RESULTS.json').write_text(json.dumps(result, indent=1))
R = result['detectors']
x = np.arange(len(DET))
labels = [f'{SHORT[recipe(d)]}\nseed {seed(d)}' for d in DET]

# Figure 1: trained alignment against the shifted alignments, eight detectors.
fig, axes = plt.subplots(1, 2, figsize=(19, 6.4), sharey=True)
for ax, role, title in zip(axes, ('training', 'development'), ('240 sampled training images', '522 development images')):
    for k, (get, color, label) in enumerate(((lambda e: e['trained_alignment'], BLUE, 'Alignment used in training'),
                                             (lambda e: e['shifted']['mirror'], ORANGE, 'Mean of the 15 shifted alignments (mirrored fill)'))):
        v = [get(R[d][role]) for d in DET]
        ax.bar(x + (k - .5) * .38, v, .35, color=color, label=label)
        for a, b in zip(x + (k - .5) * .38, v):
            ax.text(a, b + .012, f'{100 * b:.0f}', ha='center', color=INK, fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(PERCENT)
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
    ax.grid(axis='x', visible=False)
    ax.set_title(f"{title}: {R[DET[0]][role]['central']:,} nuclei at least 24 px from every border", loc='left', color=INK)
axes[0].set_ylabel('Share of annotated nuclei matched (threshold 0.5)')
axes[0].legend(loc='upper center', bbox_to_anchor=(1.02, -0.2), ncol=2, frameon=False)
fig.suptitle('Eight frozen detectors: detection at the trained alignment and at shifted alignments', x=.01, ha='left',
             fontsize=14, color=INK)
fig.tight_layout(rect=(0, .03, 1, .95))
fig.savefig(out/'figures/p36_trained_versus_shifted.png', dpi=150)
plt.close(fig)

# Figure 2: F1 against the score threshold, and detection rate against precision (development images, trained alignment).
def curve(rec, key):
    members = [d for d in DET if recipe(d) == rec]
    values = [[np.nan if t[key] is None else t[key] for t in R[d]['development']['thresholds']] for d in members]
    return np.mean(values, axis=0), len(members)


percent = matplotlib.ticker.FuncFormatter(lambda v, _: f'{100 * v:.0f}%')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for rec in RECIPE:
    f1, n = curve(rec, 'f1')
    if not n:
        continue
    recall, precision = curve(rec, 'recall')[0], curve(rec, 'precision')[0]
    label = f'{RECIPE[rec]} ({n} seed{"s" if n > 1 else ""})'
    axes[0].plot(THRESHOLDS, f1, color=COLOR[rec], lw=2, marker='o', ms=6, mec='#fcfcfb', mew=1.5, label=label)
    axes[1].plot(recall, precision, color=COLOR[rec], lw=2, marker='o', ms=6, mec='#fcfcfb', mew=1.5, label=label)
    half = THRESHOLDS.index(.5)
    axes[1].plot([recall[half]], [precision[half]], 'o', ms=13, mfc='none', mec=INK, mew=1.5)
    for t in (.1, .9):
        i = THRESHOLDS.index(t)
        if not np.isnan(precision[i]):
            axes[1].annotate(f'{t:.1f}', (recall[i], precision[i]), xytext=(6, 6), textcoords='offset points', color=MUTED, fontsize=10)
axes[0].axvline(.5, color=MUTED, lw=1, ls='--')
axes[0].set_xticks(THRESHOLDS)
axes[0].set_xlabel('Score threshold')
axes[0].set_title('F1 score at each score threshold (dashed line = 0.5 used so far)', loc='left', color=INK)
axes[1].set_xlabel('Share of annotated nuclei matched')
axes[1].set_ylabel('Share of predicted points matched')
axes[1].set_title('Each dot is one threshold from 0.1 to 0.9; black ring = threshold 0.5', loc='left', color=INK)
for ax in axes:
    ax.yaxis.set_major_formatter(percent)
axes[1].xaxis.set_major_formatter(percent)
axes[0].legend(loc='lower left', frameon=False, fontsize=11)
fig.suptitle('Development images, trained alignment: effect of the score threshold on four recipes', x=.01,
             ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(out/'figures/p36_threshold_curves.png', dpi=150)
plt.close(fig)

# Figure 3: the three decoders on development images.
fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
series = (('original', BLUE, 'Original decoder (threshold 0.5)'), ('selected_threshold', ORANGE, 'Threshold selected on shifted training images'),
          ('consensus', AQUA, 'Consensus of 15 shifted alignments'))
for ax, key, title in zip(axes, ('recall', 'precision', 'dead_recall'), ('Share of annotated nuclei matched', 'Share of predicted points matched',
                                                                           'Share of interior Dead nuclei matched (145)')):
    for k, (decoder, color, label) in enumerate(series):
        v = [R[d]['development']['decoders'][decoder][key] for d in DET]
        ax.bar(x + (k - 1) * .27, v, .25, color=color, label=label)
        for a, b in zip(x + (k - 1) * .27, v):
            ax.text(a, b + .008, f'{100 * b:.0f}', ha='center', color=INK, fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(PERCENT)
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
    ax.grid(axis='x', visible=False)
    ax.set_title(title, loc='left', color=INK)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='lower center', ncol=3, frameon=False)
fig.suptitle('Development images: three decoders on each frozen detector', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, .07, 1, .95))
fig.savefig(out/'figures/p36_decoders.png', dpi=150)
plt.close(fig)

# Figure 4: displacement of the nearest point, split into the part shared by alignments and the part that changes.
fig, axes = plt.subplots(1, 2, figsize=(19, 6.2), sharey=True)
for ax, role, title in zip(axes, ('training', 'development'), ('Training sample (shifted alignments)', 'Development images (shifted alignments)')):
    for k, (key, color, label) in enumerate((('shared_by_alignments_px2', BLUE, 'Part shared by all alignments'),
                                             ('changes_with_alignment_px2', ORANGE, 'Part that changes with the alignment'))):
        v = [R[d][role]['displacement']['mirror'][key] for d in DET]
        ax.bar(x + (k - .5) * .38, v, .35, color=color, label=label)
        for a, b in zip(x + (k - .5) * .38, v):
            ax.text(a, b + .04, f'{b:.2f}', ha='center', color=INK, fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.grid(axis='x', visible=False)
    ax.set_title(title, loc='left', color=INK)
axes[0].set_ylabel('Mean squared displacement (px²)')
axes[0].legend(loc='upper center', bbox_to_anchor=(1.02, -0.2), ncol=2, frameon=False)
fig.suptitle('Displacement of the nearest predicted point from the annotated center (isolated nuclei, points within 6 px, mirrored fill)',
             x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, .03, 1, .95))
fig.savefig(out/'figures/p36_displacement.png', dpi=150)
plt.close(fig)

for d in DET:
    e = R[d]
    print('=====', d, 'rule', {k: e['consensus_rule'][k] for k in ('fill', 'radius', 'votes')}, 'f1 %.4f' % e['consensus_rule']['f1'],
          '| threshold', e['selected_threshold']['threshold'], 'f1 %.4f' % e['selected_threshold']['f1'])
    for role in ('training', 'development'):
        q = e[role]
        print(' ', role, 'nuclei', q['nuclei'], 'central', q['central'], 'trained %.4f' % q['trained_alignment'],
              'shifted', {k: round(v, 4) for k, v in q['shifted'].items()}, 'sec/img %.1f' % q['forward_seconds_per_image'])
        for k, v in q['decoders'].items():
            print('     %-19s m %5d pts %5d dead %3d/%3d R %.4f P %.4f D %.4f F1 %.4f' % (k, v['matched'], v['candidates'], v['dead_hit'],
                  v['dead_total'], v['recall'], v['precision'], v['dead_recall'], v['f1']))
        print('     thr trained:', ' '.join('%.1f:%.3f/%.3f/%.3f/%d' % (t['threshold'], t['recall'], t['precision'] or float('nan'), t['f1'], t['dead_hit']) for t in q['thresholds']))
        print('     thr shifted(mirror) R/P:', ' '.join('%.3f/%.3f' % (t['recall'], t['precision'] or float('nan')) for t in q['shifted_thresholds']['mirror']))
        print('     thr shifted(mirror) F1:', ' '.join('%.3f' % t['f1'] for t in q['shifted_thresholds']['mirror']),
              '| white:', ' '.join('%.3f' % t['f1'] for t in q['shifted_thresholds']['white']))
        print('     best here', q['best_threshold_on_this_role'], q['best_consensus_on_this_role'])
        print('     displacement', {f: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for f, x in q['displacement'].items()})
        print('     bands', {b: (v['objects'], round(v['original'], 3), round(v['white'], 3), round(v['mirror'], 3)) for b, v in q['border_bands'].items()})
        print('     dead', q['interior_dead_all_alignments_valid'], 'transitions', q['consensus_transitions'], 'stored', q['stored_original_comparison'] and
              (q['stored_original_comparison']['images_with_different_counts'], q['stored_original_comparison']['stored_counts']))
        print('     hist mirror', q['shifted_histogram']['mirror'])
for k, v in result.get('decisions', {}).items():
    print(k, v['passed'], v['reason'], {s: (round(d['delta'], 4), [round(c, 4) for c in d['ci95']], round(d['dead_recall'], 4),
                                           round(d['precision'], 4), round(d['recall'], 4)) for s, d in v['seeds'].items()})
