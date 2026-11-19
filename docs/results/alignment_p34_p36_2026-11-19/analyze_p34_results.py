"""P34 result analysis from the collected per-object records.

Usage: python analyze_p34_results.py <collected results dir> <output dir>
Reads the three completed detectors; writes P34_RESULTS.json and three figures.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, out = Path(sys.argv[1]), Path(sys.argv[2])
(out/'figures').mkdir(parents=True, exist_ok=True)
DET = ['P22_C_s19', 'P22_C_s1', 'P22_M_s19']
NAME = {'P22_C_s19': 'Last 12 layers, seed 19', 'P22_C_s1': 'Last 12 layers, seed 1', 'P22_M_s19': 'All 24 layers, seed 19'}
ROLES = ['training', 'training_shifted', 'development']
ROLE = {'training': 'Training images', 'training_shifted': 'Same training images moved by (5, 3) px',
        'development': 'Development images'}
READOUTS = ('same_augmentation', 'with_translation')
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
COLORS = (BLUE, ORANGE, AQUA)
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b8b7b0',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'axes.grid': True,
                     'grid.color': '#e6e5df', 'grid.linewidth': .8, 'axes.axisbelow': True})
PERCENT = np.arange(0, 1.01, .2)


def rows(det, name):
    with (root/det/name).open(newline='') as f:
        return list(csv.DictReader(f))


def flag(v):
    return v == 'True'


def number(v):
    return float(v) if v not in ('', 'None') else np.nan


def rates(block):
    return dict(block['counts'], recall=block['rates']['recall'], precision=block['rates']['precision'],
                dead_recall=block['rates']['dead_recall'])


result = dict(detectors={}, note='P22_M_s1 (array task 3) stopped at the first-epoch time guard of its second '
                                 'readout and has no evaluation.')
objects = {}
for det in DET:
    s = json.loads((root/det/'summary.json').read_text())
    entry = dict(roles={}, training=s['training'])
    for role in ROLES:
        r = s['roles'][role]
        o = rows(det, f'{role}_objects.csv')
        objects[(det, role)] = o
        e = dict(objects=len(o), original=rates(r['original']))
        m0 = np.array([flag(x['matched_original']) for x in o])
        dead = np.array([x['cls'] == '4' and flag(x['interior']) for x in o])
        before = np.array([number(x['nearest_inside_point_px']) for x in o])
        has = ~np.isnan(before)
        for rd in READOUTS:
            err = np.array([number(x[f'center_vote_error_px_{rd}']) for x in o])
            after = np.array([number(x[f'nearest_inside_point_refined_px_{rd}']) for x in o])
            m1 = np.array([flag(x[f'matched_merged_{rd}']) for x in o])
            e[rd] = dict(
                merged=rates(r['readouts'][rd]['merged']),
                offset_error_at_center=dict(median=float(np.median(err)), q90=float(np.quantile(err, .9)),
                                            within1=float((err <= 1).mean()), within3=float((err <= 3).mean())),
                transitions=dict(gained=int((~m0 & m1).sum()), lost=int((m0 & ~m1).sum()),
                                 dead_gained=int((~m0 & m1 & dead).sum()), dead_lost=int((m0 & ~m1 & dead).sum())),
                nuclei_with_inside_point=int(has.sum()),
                inside_point_far_before=int((has & (before > 3)).sum()),
                far_before_within3_after=int((has & (before > 3) & (after <= 3)).sum()),
                near_before_far_after=int((has & (before <= 3) & (after > 3)).sum()),
                median_point_distance_before=float(np.nanmedian(before)),
                median_point_distance_after=float(np.nanmedian(after)))
        if role == 'training_shifted':
            cy = np.array([int(x['center_y']) for x in o])
            cx = np.array([int(x['center_x']) for x in o])
            far = (cy >= 37) & (cx >= 35)
            e['original_recall_32px_from_filled_edges'] = dict(objects=int(far.sum()), recall=float(m0[far].mean()))
        entry['roles'][role] = e
    result['detectors'][det] = entry
(out/'P34_RESULTS.json').write_text(json.dumps(result, indent=1))

# Figure 1: detector recall and precision on the three image groups (original decoder).
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
titles = ('Share of annotated nuclei matched', 'Share of predicted points matched')
for ax, metric, title in zip(axes, ('recall', 'precision'), titles):
    w = .26
    for k, (role, color) in enumerate(zip(ROLES, COLORS)):
        v = [result['detectors'][d]['roles'][role]['original'][metric] for d in DET]
        ax.bar(np.arange(3) + (k - 1) * w, v, w * .92, color=color, label=ROLE[role])
        for x, y in zip(np.arange(3) + (k - 1) * w, v):
            ax.text(x, y + .012, f'{100 * y:.1f}', ha='center', color=INK, fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[d] for d in DET])
    ax.set_ylim(0, 1.1)
    ax.set_title(title, loc='left', color=INK)
    ax.set_yticks(PERCENT)
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
    ax.grid(axis='x', visible=False)
axes[0].legend(loc='upper center', bbox_to_anchor=(1.02, -0.08), ncol=3, frameon=False)
fig.suptitle('Frozen detectors, original decoder: a training image moved by (5, 3) pixels is detected '
             'like an unseen image', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, .04, 1, .95))
fig.savefig(out/'figures/p34_shift_sensitivity.png', dpi=150)
plt.close(fig)

# Figure 2: cumulative distribution of the offset error read at the annotated center.
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
grid = np.linspace(0, 10, 401)
titles = ("Readout trained with the detector's augmentation", 'Readout trained with added translation of up to 8 px')
for ax, rd, title in zip(axes, READOUTS, titles):
    for role, color, drop in zip(ROLES, COLORS, (.05, .13, .21)):
        err = np.sort([number(x[f'center_vote_error_px_{rd}']) for x in objects[('P22_M_s19', role)]])
        ax.plot(grid, np.searchsorted(err, grid, side='right') / len(err), color=color, lw=2, label=ROLE[role])
        inside = float(np.mean(err <= 3))
        ax.plot([3], [inside], 'o', color=color, ms=7, mec='#fcfcfb', mew=2)
        ax.text(5.2, .52 - drop * 1.2, f'{100 * inside:.1f}% within 3 px', color=INK, fontsize=11)
        ax.plot([4.8], [.53 - drop * 1.2], 's', color=color, ms=8)
    ax.axvline(3, color=MUTED, lw=1, ls='--')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Distance between predicted and annotated center, read at the center pixel (px)')
    ax.set_title(title, loc='left', color=INK)
    ax.set_yticks(PERCENT)
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
axes[0].set_ylabel('Share of annotated nuclei')
axes[0].legend(loc='lower right', frameon=False)
fig.suptitle('Offset readout on the all-24-layer detector (seed 19): error for every annotated nucleus; '
             'dashed line = matching radius', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(out/'figures/p34_offset_error.png', dpi=150)
plt.close(fig)

# Figure 3: development nuclei, nearest in-mask predicted point before and after the move.
fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), sharey=True)
for ax, det in zip(axes, DET):
    o = objects[(det, 'development')]
    before = np.array([number(x['nearest_inside_point_px']) for x in o])
    after = np.array([number(x['nearest_inside_point_refined_px_with_translation']) for x in o])
    keep = ~np.isnan(before)
    ax.scatter(before[keep], after[keep], s=7, color=BLUE, alpha=.25, linewidths=0)
    ax.plot([0, 12], [0, 12], color=MUTED, lw=1)
    ax.axvline(3, color=MUTED, lw=1, ls='--')
    ax.axhline(3, color=MUTED, lw=1, ls='--')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.set_title(NAME[det], loc='left', color=INK)
    t = result['detectors'][det]['roles']['development']['with_translation']
    ax.text(7.2, 1.0, f"moved inside 3 px:\n{t['far_before_within3_after']} of {t['inside_point_far_before']}",
            color=INK, fontsize=11)
    ax.text(.3, 10.2, f"moved outside 3 px: {t['near_before_far_after']}", color=INK, fontsize=11)
    ax.set_xlabel('Before: nearest in-nucleus point to center (px)')
axes[0].set_ylabel('After the offset move (px)')
fig.suptitle('Development images, translated-training readout: each dot is one annotated nucleus with a '
             'predicted point inside its mask', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(out/'figures/p34_point_movement.png', dpi=150)
plt.close(fig)

for d in DET:
    for r in ROLES:
        e = result['detectors'][d]['roles'][r]
        print(d, r, e['objects'], 'orig m/c/d', e['original']['matched'], e['original']['candidates'],
              e['original']['dead_hit'], 'R/P/D %.4f %.4f %.4f' % (e['original']['recall'], e['original']['precision'],
                                                                e['original']['dead_recall']),
              e.get('original_recall_32px_from_filled_edges', ''))
        for rd in READOUTS:
            x = e[rd]
            print('   ', rd, x['merged']['matched'], x['merged']['candidates'], x['merged']['dead_hit'],
                  {k: round(v, 3) for k, v in x['offset_error_at_center'].items()}, x['transitions'], 'far',
                  x['inside_point_far_before'], 'in', x['far_before_within3_after'], 'out', x['near_before_far_after'],
                  'med', round(x['median_point_distance_before'], 2), round(x['median_point_distance_after'], 2))
    print('   loss', {k: (round(v['first_epoch_loss'], 2), round(v['last_epoch_loss'], 2))
                      for k, v in result['detectors'][d]['training'].items()})
