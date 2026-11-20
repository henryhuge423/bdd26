"""Export measured scores and offsets for the fixed D37 image cases."""
import csv,json,sys
from pathlib import Path
import numpy as np
from analyze_alignment_misses_d37 import read_objects,read_peaks
from evaluate_alignment_consensus_p36 import ORIGINAL,retained,consensus_groups,consensus_points,match_points
root,out=map(Path,sys.argv[1:3]); records=[]
for model in ('P22_M_s19','P15_J_s19'):
    folder=root/model
    objects=read_objects(folder/'development_objects.csv')
    peaks=read_peaks(folder/'development_points.npz',set(np.load(folder/'development_points.npz')['image'].tolist()))
    summary=json.loads((folder/'summary.json').read_text()); rule=summary['rule']
    for image in (178,776,1037,1482):
        rows=objects[image]; gt=[(r['cls'],(r['center_y'],r['center_x']),r['interior'],r['area']) for r in rows]
        chosen=consensus_points(consensus_groups(peaks[image][rule['fill']],rule['radius']),rule['votes'])
        paired={int(g):int(p) for g,p in match_points(chosen,gt)}
        for j,r in enumerate(rows):
            c=np.array((r['center_y'],r['center_x']))
            ps=peaks[image]['white'][ORIGINAL]
            pp=np.array([(y,x) for y,x,s in ps]).reshape(-1,2)
            dd=((pp-c)**2).sum(1) if len(pp) else np.array([])
            scores=[s for y,x,s in ps]
            rec={k:r[k] for k in ('image','gt_index','cls','center_y','center_x','area','matched_original','matched_consensus','shifted_matched_white')}
            rec['model']=model
            for radius in (3,6): rec[f'original_max_score_{radius}px']=max((s for s,d in zip(scores,dd) if d<=radius**2),default=None)
            if j in paired:
                p=np.array(chosen[paired[j]])
                rec.update(consensus_y=float(p[0]),consensus_x=float(p[1]),consensus_distance=float(np.linalg.norm(p-c)))
            else: rec.update(consensus_y=None,consensus_x=None,consensus_distance=None)
            records.append(rec)
with out.open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
