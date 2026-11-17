"""Rebuild large P22/P25 overlays from preserved coordinates and shared original pixels.

python plot_stage_examples.py --root docs/results/p30_p31_2026-11-16
Statistics in the reports are compiled separately with PGFPlots.
"""
from pathlib import Path
import argparse,json,hashlib,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
states={(True,True):'retained',(False,True):'gained',(True,False):'lost',(False,False):'missed_both'}
colors={'retained':'#fffbe3','gained':'#65ffaf','lost':'#ffb947','missed_both':'#ff6191'}
objects=[];provenance={}
def read(path):
 b=path.read_bytes();provenance[path.relative_to(root).as_posix()]=hashlib.sha256(b).hexdigest();return json.loads(b)
for stage,arm,label in [('P22','M','All 24 blocks'),('P25','D','All 24, layerwise rates')]:
 for seed in (19,1):
  records=[{r['image']:r for r in read(root/f'early_files/{stage}/A_{x}_s{seed}/point_examples_audit/coordinates.json')['records']} for x in ('C',arm)]
  original={r['image']:r for r in read(root/f'p30_files/A_C_s{seed}/point_examples_audit/coordinates.json')['records']}
  for image,left in records[0].items():
   right=records[1][image];assert left['gt']==right['gt']==original[image]['gt']
   pc,pm=[dict(r['unique_pairs']) for r in (left,right)]
   for r,pmatch in zip((left,right),(pc,pm)):
    assert len(pmatch)==r['score']['matched'];assert all(math.dist(r['gt'][g][1],r['points'][i])<=3 for g,i in pmatch.items())
   rows=[]
   for i,(klass,(y,x),interior,area) in enumerate(left['gt']):
    if klass!=4 or not interior:continue
    row=dict(stage=stage,seed=seed,image=image,index=i,y=y,x=x,area=area,state=states[i in pc,i in pm],control_nearest_px=min((math.dist([y,x],q) for q in left['points']),default=None),intervention_nearest_px=min((math.dist([y,x],q) for q in right['points']),default=None))
    objects.append(row);rows.append(row)
   if image not in (776,1037):continue
   png=root/f'p30_files/A_C_s{seed}/point_examples_audit'/original[image]['png'];b=png.read_bytes();provenance[png.relative_to(root).as_posix()]=hashlib.sha256(b).hexdigest()
   with Image.open(png) as im:assert im.size==(1024,310);pixels=np.asarray(im.crop((0,42,256,298)))
   fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
   for ax,r,title in zip(axes,(left,right),('Last 12 blocks',label)):
    ax.imshow(pixels)
    if r['points']:
     y,x=np.asarray(r['points']).T;ax.scatter(x,y,marker='+',s=42,color='#6deaff',linewidths=1.1)
    for row in rows:
     ax.add_patch(plt.Circle((row['x'],row['y']),4,fill=False,color=colors[row['state']],linewidth=1.6))
     if row['state'] in ('gained','lost'):ax.annotate(str(row['index']),(row['x'],row['y']),xytext=(5,5),textcoords='offset points',color=colors[row['state']],fontsize=10,bbox=dict(facecolor='#292333',alpha=.9,edgecolor='none',pad=1))
    ax.set_title(f"{title}\nInterior Dead: {r['score']['dead_hit']}/{r['score']['dead_total']}");ax.set_axis_off()
   handles=[plt.Line2D([],[],marker='+',color='#6deaff',linestyle='none',label='Predicted center')]
   handles.extend(plt.Line2D([],[],marker='o',markerfacecolor='none',markeredgewidth=2,color=colors[s],linestyle='none',label=t) for s,t in [('retained','Both match'),('gained','Intervention only'),('lost','Control only'),('missed_both','Both miss')])
   fig.legend(handles=handles,loc='outside lower center',ncols=5,facecolor='#454052',labelcolor='white');fig.suptitle(f'{stage}: {left["tissue"]} image {image}, seed {seed}')
   fig.savefig(root/f'figures/{stage.lower()}_image_{image}_seed_{seed}.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
with (root/'early_fixed_objects.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=objects[0].keys());w.writeheader();w.writerows(objects)
(root/'EARLY_FIGURE_PROVENANCE.json').write_text(json.dumps(dict(input_sha256=provenance,objects=len(objects),image_source='Original native RGB panel from P30; identical frozen input and GT identities across rounds',matching='preserved one-to-one assignments; radius 3 native pixels'),indent=2))
print('EARLY_EXAMPLES',len(objects),'objects; 8 large images')
