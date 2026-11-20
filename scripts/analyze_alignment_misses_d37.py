"""D37: fixed diagnostic partition of P36 misses from preserved point records.

No model forward pass or decoder selection. Use the stored 0.1 recording floor
and 0.5 decoder threshold, original one-to-one matching, and both data roles.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.distance import cdist
from evaluate_alignment_consensus_p36 import (FILLS, ORIGINAL, SHIFTED, retained,
    consensus_groups, consensus_points, match_points, score_points)

CATEGORIES = ('matched_original', 'matched_another_alignment', 'matched_only_below_0.5',
              'near_point_assigned_elsewhere', 'no_point_within_3px_at_0.1')
FIXED_IMAGES = (178, 776, 1037, 1482)


def partition(original, high, low, near):
    if original: return CATEGORIES[0]
    if high: return CATEGORIES[1]
    if low: return CATEGORIES[2]
    if near: return CATEGORIES[3]
    return CATEGORIES[4]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_objects(path):
    with path.open(newline='') as f: rows = list(csv.DictReader(f))
    grouped = {}
    for r in rows:
        for k in ('image', 'gt_index', 'cls', 'center_y', 'center_x', 'area', 'border', 'valid_phases', 'neighbors12',
                  'phases_white', 'phases_mirror'): r[k] = int(r[k])
        for k in ('interior', 'matched_original', 'matched_consensus'): r[k] = r[k] == 'True'
        grouped.setdefault(r['image'], []).append(r)
    for group in grouped.values():
        assert [r['gt_index'] for r in group] == list(range(len(group)))
    return grouped


def read_peaks(path, images):
    with np.load(path) as a:
        data = {k: a[k].copy() for k in a.files}
    assert all(len(v) == len(data['image']) for v in data.values())
    peaks = {i: {f: [[] for _ in range(16)] for f in FILLS} for i in images}
    for i, f, phase, y, x, score in zip(*(data[k] for k in ('image','fill','phase','y','x','score'))):
        peaks[int(i)][FILLS[int(f)]][int(phase)].append((int(y),int(x),float(score)))
    return peaks


def render_case(image, gt, points, labels, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    centers = np.asarray([g[1] for g in gt])
    for ax, pp, title in zip(axes.flat, points, labels):
        ax.imshow(image)
        ax.scatter(centers[:,1], centers[:,0], s=32, facecolors='none', edgecolors='#00c7e5', linewidths=.9)
        if pp:
            hit = {p for _,p in match_points(pp,gt)}
            for ok, color, marker in ((True,'#17aa4a','+'), (False,'#f82452','x')):
                aa=np.array([p for j,p in enumerate(pp) if (j in hit)==ok]).reshape(-1,2)
                if len(aa): ax.scatter(aa[:,1],aa[:,0],s=26,c=color,marker=marker,linewidths=1)
        for j,g in enumerate(gt):
            if g[0]==4 and g[2]: ax.text(g[1][1]+2,g[1][0]-2,str(j),fontsize=8,color='yellow',weight='bold')
        ax.set_title(title, fontsize=12); ax.set_xlim(-.5,255.5); ax.set_ylim(255.5,-.5); ax.set_axis_off()
    fig.text(.04,.02,'Cyan circles: annotated centers; green +: matched prediction; red x: unmatched prediction.\nYellow numbers: interior Dead nucleus index (coordinates in CASES.csv).',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(path,dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--images',type=Path)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False); (args.out/'figures').mkdir()
    started=time.monotonic(); result={'experiment':'D37','definition':CATEGORIES,'detectors':{},'inputs':{}}
    images=np.load(args.images,mmap_mode='r') if args.images else None
    cases=[]; all_records=[]
    detectors=sorted(p for p in args.root.iterdir() if p.is_dir() and (p/'complete.json').exists())
    if len(detectors)!=8: raise ValueError('D37 requires all eight P36 detectors')
    for folder in detectors:
        manifest=json.loads((folder/'preservation_manifest.json').read_text())
        used=['complete.json','summary.json','selected_rule.json']
        used += [f'{role}_{name}' for role in ('training','development') for name in
                 ('objects.csv','points.npz','original.json','selected_threshold.json','consensus.json')]
        for name in used:
            p=folder/name; h=digest(p)
            if name in manifest: assert h==manifest[name]['sha256'] and p.stat().st_size==manifest[name]['bytes'],p
            result['inputs'][folder.name+'/'+name]={'sha256':h,'bytes':p.stat().st_size}
        summary=json.loads((folder/'summary.json').read_text()); rule=summary['rule']; threshold=summary['threshold']['threshold']
        entry={}
        for role in ('training','development'):
            objects=read_objects(folder/f'{role}_objects.csv'); peaks=read_peaks(folder/f'{role}_points.npz',objects)
            original_rows={r['image']:r for r in json.loads((folder/f'{role}_original.json').read_text())}
            selected_rows={r['image']:r for r in json.loads((folder/f'{role}_selected_threshold.json').read_text())}
            consensus_rows={r['image']:r for r in json.loads((folder/f'{role}_consensus.json').read_text())}
            buckets={f:{subset:{c:0 for c in CATEGORIES} for subset in ('all','central','interior_dead')} for f in FILLS}
            selected_gain=selected_loss=0
            for image, rows in objects.items():
                gt=[(r['cls'],(r['center_y'],r['center_x']),r['interior'],r['area']) for r in rows]
                original=retained(peaks[image]['white'][ORIGINAL]); selected=retained(peaks[image]['white'][ORIGINAL],threshold)
                chosen=consensus_points(consensus_groups(peaks[image][rule['fill']],rule['radius']),rule['votes'])
                for pp, expected in ((original,original_rows[image]),(selected,selected_rows[image]),(chosen,consensus_rows[image])):
                    actual=score_points(pp,gt)
                    for k in ('matched','total','candidates','dead_hit','dead_total'): assert actual[k]==expected[k],(folder.name,role,image,k)
                original_hits={int(g) for g,_ in match_points(original,gt)}
                selected_hits={int(g) for g,_ in match_points(selected,gt)}
                selected_gain+=len(selected_hits-original_hits); selected_loss+=len(original_hits-selected_hits)
                records=[dict(model=folder.name,role=role,image=image,gt_index=r['gt_index'],cls=r['cls'],
                              center_y=r['center_y'],center_x=r['center_x'],area=r['area'],border=r['border'],
                              original=r['matched_original'],selected=r['gt_index'] in selected_hits,consensus=r['matched_consensus']) for r in rows]
                for fill in FILLS:
                    low=np.zeros(len(gt),dtype=np.uint32); near=np.zeros(len(gt),dtype=bool)
                    for phase, ps in enumerate(peaks[image][fill]):
                        hit={int(g) for g,_ in match_points(retained(ps,.1),gt)}
                        for g in hit: low[g] |= np.uint32(1<<phase)
                        if ps and gt:
                            dd=cdist(np.asarray([g[1] for g in gt]),np.asarray([(y,x) for y,x,_ in ps]))
                            near |= (dd<=3).any(1)
                    for j,r in enumerate(rows):
                        category=partition(r['matched_original'],r['phases_'+fill]>0,low[j]>0,near[j])
                        records[j]['category_'+fill]=category; records[j]['low_score_phases_'+fill]=int(low[j])
                        for subset, keep in (('all',True),('central',r['border']>=24 and r['valid_phases']==16),
                                             ('interior_dead',r['cls']==4 and r['interior'])):
                            if keep: buckets[fill][subset][category]+=1
                all_records.extend(records)
                if role=='development' and image in FIXED_IMAGES and folder.name in ('P22_M_s19','P15_J_s19'):
                    cases.extend(records)
                    if images is not None:
                        render_case(images[image],gt,[[],original,selected,chosen],['Annotated centers',
                            f'Original threshold 0.5: {len(original_hits)} matched',
                            f'Training-selected threshold {threshold:g}: {len(selected_hits)} matched',
                            f'Consensus: {consensus_rows[image]["matched"]} matched'],
                            args.out/'figures'/f'{folder.name}_development_{image:04d}.png')
            entry[role]={'partition':buckets,'selected_threshold_transitions':{'gained':selected_gain,'lost':selected_loss},
                         'images':len(objects),'nuclei':sum(len(r) for r in objects.values())}
            for fill in FILLS: assert sum(buckets[fill]['all'].values())==entry[role]['nuclei']
            print(folder.name,role,entry[role],flush=True)
        result['detectors'][folder.name]=entry
    for name, rows in (('OBJECTS.csv',all_records),('CASES.csv',cases)):
        with (args.out/name).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result['elapsed_seconds']=time.monotonic()-started
    (args.out/'D37_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
