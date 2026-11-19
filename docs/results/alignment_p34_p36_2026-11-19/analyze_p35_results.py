"""P35 result analysis from the collected per-object records.

Usage: python analyze_p35_results.py <P35 collected results> <P34 collected results> <output dir>
Writes P35_RESULTS.json and three figures. The P34 object table supplies the
number of annotated neighbors within 12 pixels (annotation only).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, p34, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
(out/'figures').mkdir(parents=True, exist_ok=True)
DET = ['P22_C_s19', 'P22_C_s1', 'P22_M_s19', 'P22_M_s1']
NAME = {'P22_C_s19': 'Last 12 layers\nseed 19', 'P22_C_s1': 'Last 12 layers\nseed 1',
        'P22_M_s19': 'All 24 layers\nseed 19', 'P22_M_s1': 'All 24 layers\nseed 1'}
CLASSES = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'}
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b8b7b0',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'axes.grid': True,
                     'grid.color': '#e6e5df', 'grid.linewidth': .8, 'axes.axisbelow': True})
PERCENT = np.arange(0, 1.01, .2)


def load(det, role):
    with (root/det/f'{role}_objects.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    o = dict(image=a['image'].astype(int), gt=a['gt_index'].astype(int), cls=a['cls'].astype(int),
             area=a['area'].astype(int), cy=a['center_y'].astype(int), cx=a['center_x'].astype(int),
             valid=a['valid_phases'].astype(int), k=a['phases_matched'].astype(int), m0=a['matched_original'] == 'True',
             other=a['matched_other_phases'].astype(int), smean=a['near_center_score_phase_mean'].astype(float),
             sp=a['near_center_score_mean_probability'].astype(float), mp=a['matched_mean_probability'] == 'True')
    o['border'] = np.minimum.reduce([o['cy'], o['cx'], 255 - o['cy'], 255 - o['cx']])
    o['central'] = (o['valid'] == 16) & (o['border'] >= 24)
    return o


neighbors = {}
with (p34/'P22_C_s19/development_objects.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        neighbors[(int(r['image']), int(r['gt_index']))] = int(r['neighbors12'])
tissue = {r['image']: r['tissue'] for r in json.loads((root/'P22_C_s19/development_original.json').read_text())}

result = dict(detectors={}, definition='central = valid at all sixteen alignments and at least 24 px from every border')
data = {}
for det in DET:
    s = json.loads((root/det/'summary.json').read_text())
    entry = dict(decoders={role: {n: dict(v['counts'], recall=v['rates']['recall'], precision=v['rates']['precision'],
                                          dead_recall=v['rates']['dead_recall']) for n, v in s['roles'][role]['decoders'].items()}
                           for role in ('development', 'training')})
    for role in ('development', 'training'):
        o = data[(det, role)] = load(det, role)
        c = o['central']
        h = np.bincount(o['k'][c], minlength=17)
        rate = o['k'][c] / 16
        m = rate.mean()
        always = c & (o['k'] == 16)
        entry[role] = dict(
            central=int(c.sum()), trained_alignment=float(o['m0'][c].mean()), other_alignments=float(o['other'][c].mean() / 15),
            histogram=h.tolist(), never=float(h[0] / c.sum()), sometimes=float(h[1:16].sum() / c.sum()),
            always=float(h[16] / c.sum()), at_least_one=float(1 - h[0] / c.sum()),
            between_nucleus_share=float((rate.var() / (m * (1 - m)) - 1 / 16) / (15 / 16)),
            always_matched_single_alignment_score_median=float(np.median(o['smean'][always])),
            always_matched_averaged_score_median=float(np.median(o['sp'][always])),
            always_matched_averaged_score_at_least_half=float((o['sp'][always] >= .5).mean()))
    result['detectors'][det] = entry

# Structure of the development nuclei: mean over the four detectors of the share of alignments matched.
o0 = data[(DET[0], 'development')]
c = o0['central']
n12 = np.array([neighbors[(i, g)] for i, g in zip(o0['image'], o0['gt'])])
tis = np.array([tissue[i] for i in o0['image']])
rate = np.mean([data[(d, 'development')]['k'] / 16 for d in DET], axis=0)
never = np.mean([data[(d, 'development')]['k'] == 0 for d in DET], axis=0)
groups = []


def add(family, label, mask):
    mask = mask & c
    if mask.sum() >= 30:
        groups.append(dict(family=family, label=label, nuclei=int(mask.sum()), mean_share_of_alignments=float(rate[mask].mean()),
                           never_matched=float(never[mask].mean())))


for lo, hi, label in ((0, 80, 'under 80 px'), (80, 150, '80-149 px'), (150, 300, '150-299 px'), (300, 600, '300-599 px'),
                      (600, 10 ** 6, '600 px or more')):
    add('Area of the nucleus', label, (o0['area'] >= lo) & (o0['area'] < hi))
for q, label in ((0, 'no neighbor'), (1, 'one neighbor')):
    add('Annotated centers within 12 px', label, n12 == q)
add('Annotated centers within 12 px', 'two or more', n12 >= 2)
for q, label in CLASSES.items():
    add('Class', label, o0['cls'] == q)
for t in sorted(set(tis), key=lambda t: -rate[c & (tis == t)].mean() if (c & (tis == t)).sum() >= 100 else 1):
    if (c & (tis == t)).sum() >= 100:
        add('Tissue (100 nuclei or more)', t.replace('_', ' '), tis == t)
result['development_structure'] = groups
(out/'P35_RESULTS.json').write_text(json.dumps(result, indent=1))

# Figure 1: trained alignment against the fifteen other alignments.
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, role, title in zip(axes, ('training', 'development'), ('48 selected training images', '522 development images')):
    for k, (key, color, label) in enumerate((('trained_alignment', BLUE, 'Alignment used in training'),
                                             ('other_alignments', ORANGE, 'Mean of the 15 shifted alignments'))):
        v = [result['detectors'][d][role][key] for d in DET]
        x = np.arange(4) + (k - .5) * .36
        ax.bar(x, v, .33, color=color, label=label)
        for a, b in zip(x, v):
            ax.text(a, b + .012, f'{100 * b:.1f}', ha='center', color=INK, fontsize=11)
    ax.set_xticks(range(4))
    ax.set_xticklabels([NAME[d] for d in DET])
    ax.set_ylim(0, 1.1)
    ax.set_yticks(PERCENT)
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
    ax.grid(axis='x', visible=False)
    n = result['detectors'][DET[0]][role]['central']
    ax.set_title(f'{title}: {n:,} nuclei at least 24 px from every border', loc='left', color=INK)
axes[0].set_ylabel('Share of annotated nuclei matched')
axes[0].legend(loc='upper center', bbox_to_anchor=(1.02, -0.13), ncol=2, frameon=False)
fig.suptitle('Frozen detectors at sixteen alignments to the 16-pixel patch grid', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, .03, 1, .95))
fig.savefig(out/'figures/p35_trained_versus_shifted.png', dpi=150)
plt.close(fig)

# Figure 2: number of alignments at which each development nucleus is matched.
fig, axes = plt.subplots(1, 4, figsize=(19, 5.8), sharey=True)
for ax, det in zip(axes, DET):
    e = result['detectors'][det]['development']
    h = np.array(e['histogram']) / e['central']
    ax.bar(range(17), h, .8, color=BLUE)
    for q in (0, 16):
        ax.text(q, h[q] + .008, f'{100 * h[q]:.1f}%', ha='center', color=INK, fontsize=11)
    ax.text(8, .2, f"1 to 15 alignments:\n{100 * e['sometimes']:.1f}% of nuclei", ha='center', color=INK, fontsize=11)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(0, .42)
    ax.set_yticks(np.arange(0, .41, .1))
    ax.set_yticklabels([f'{int(round(100 * t))}%' for t in np.arange(0, .41, .1)])
    ax.grid(axis='x', visible=False)
    ax.set_title(NAME[det].replace('\n', ', '), loc='left', color=INK)
    ax.set_xlabel('Alignments (of 16) at which the nucleus is matched')
axes[0].set_ylabel('Share of development nuclei')
fig.suptitle('Development images: for how many of the sixteen alignments is each nucleus matched? '
             '(7,014 nuclei at least 24 px from every border)', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(out/'figures/p35_alignment_counts.png', dpi=150)
plt.close(fig)

# Figure 3: which nuclei are matched at few alignments.
families = []
for g in groups:
    if g['family'] not in families:
        families.append(g['family'])
fig, axes = plt.subplots(1, 4, figsize=(20, 7.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
for ax, family in zip(axes, families):
    rows = [g for g in groups if g['family'] == family][::-1]
    y = np.arange(len(rows))
    v = [g['mean_share_of_alignments'] for g in rows]
    ax.barh(y, v, .62, color=BLUE)
    for a, g in zip(y, rows):
        ax.text(g['mean_share_of_alignments'] + .012, a, f"{100 * g['mean_share_of_alignments']:.0f}%  (n={g['nuclei']:,})",
                va='center', color=INK, fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels([g['label'] for g in rows])
    ax.set_xlim(0, 1.18)
    ax.set_xticks(PERCENT)
    ax.set_xticklabels([f'{int(round(100 * t))}%' for t in PERCENT])
    ax.grid(axis='y', visible=False)
    ax.set_title(family, loc='left', color=INK, fontsize=12)
axes[0].set_xlabel('Mean share of the 16 alignments at which a nucleus is matched')
fig.suptitle('Development images: share of alignments matched, by properties of the annotated nucleus '
             '(mean of the four detectors)', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(out/'figures/p35_nucleus_structure.png', dpi=150)
plt.close(fig)

for d in DET:
    for role in ('training', 'development'):
        e = result['detectors'][d][role]
        print(d, role, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != 'histogram'})
    print('   ', {n: (v['matched'], v['candidates'], v['dead_hit'], round(v['precision'], 4)) for n, v in result['detectors'][d]['decoders']['development'].items()})
for g in groups:
    print(g['family'], '|', g['label'], g['nuclei'], round(g['mean_share_of_alignments'], 3), round(g['never_matched'], 3))
