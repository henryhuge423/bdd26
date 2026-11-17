"""Plot verified P30 results and fixed object matches; preserve partial-seed status."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--prior-stage-pairs',type=Path,required=True)
args=parser.parse_args()
out=args.root; raw=out/'p30_files'; figures=out/'figures';figures.mkdir(exist_ok=True)
analysis=json.loads((out/'p30_verified_analysis.json').read_text())
seeds=sorted({int(k.split('_')[-1]) for k in analysis['runs']},reverse=True)
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(figures/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
def records(arm,seed):
    r=json.loads((raw/f'A_{arm}_s{seed}/point_examples_audit/coordinates.json').read_text())
    return {v['image']:v for v in r['records']}
def pairs(row):
    p=dict(row['unique_pairs'])
    assert len(p)==row['score']['matched']
    assert all(math.dist(row['gt'][g][1],row['points'][i])<=3 for g,i in p.items())
    return p
states={(True,True):'retained',(False,True):'gained',(True,False):'lost',(False,False):'missed_both'}
colors={'retained':'#fffbe3','gained':'#65ffaf','lost':'#ffb947','missed_both':'#ff6191'}
objects=[];summary={}
for seed in seeds:
    c,b=records('C',seed),records('M',seed);assert c.keys()==b.keys()
    for image,left in c.items():
        right=b[image];assert left['gt']==right['gt'];pc,pb=pairs(left),pairs(right)
        for index,(klass,(y,x),interior,area) in enumerate(left['gt']):
            if klass!=4 or not interior:continue
            row=dict(seed=seed,image=image,tissue=left['tissue'],index=index,y=y,x=x,area=area,state=states[index in pc,index in pb])
            for arm,record,matched in [('c',left,pc),('m',right,pb)]:
                row[arm+'_matched_yx']=record['points'][matched[index]] if index in matched else None
                row[arm+'_nearest_px']=min((math.dist([y,x],p) for p in record['points']),default=None)
            objects.append(row)
    selected=[r for r in objects if r['seed']==seed];assert len(selected)==53
    changes=analysis['pairs'][str(seed)]['dead_images'];net=sum(r['net'] for r in changes)
    sensitivity=[dict(image=r['image'],gain=(net-r['net'])/(145-r['dead_total'])) for r in changes]
    summary[str(seed)]=dict(fixed=dict(Counter(r['state'] for r in selected)),leave_one=sensitivity)
    for image in [776,1025,1037]:
        shown=[r for r in selected if r['image']==image]
        fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
        for axis,arm,title in zip(axes,['C','M'],['Last 12 blocks','Middle 12 blocks']):
            record=records(arm,seed)[image]
            with Image.open(raw/f'A_{arm}_s{seed}/point_examples_audit'/record['png']) as panel:
                assert panel.size==(1024,310)
                im=np.asarray(panel.crop((0,42,256,298)))
            axis.imshow(im)
            if record['points']:
                y,x=np.asarray(record['points']).T;axis.scatter(x,y,marker='+',s=42,color='#6deaff',linewidths=1.1)
            for r in shown:
                axis.add_patch(plt.Circle((r['x'],r['y']),4,fill=False,color=colors[r['state']],linewidth=1.6))
                if r['state'] in ['gained','lost']:
                    axis.annotate(str(r['index']),(r['x'],r['y']),xytext=(5,5),textcoords='offset points',color=colors[r['state']],fontsize=10,bbox=dict(facecolor='#292333',alpha=.9,edgecolor='none',pad=1))
            axis.set_title(f"{title}\nInterior Dead: {record['score']['dead_hit']}/{record['score']['dead_total']}")
            axis.set_axis_off()
        handles=[plt.Line2D([],[],marker='+',color='#6deaff',linestyle='none',label='Predicted center')]
        handles.extend(plt.Line2D([],[],marker='o',markerfacecolor='none',markeredgewidth=2,color=colors[s],linestyle='none',label=t) for s,t in [('retained','Both match'),('gained','Middle only'),('lost','Control only'),('missed_both','Both miss')])
        fig.legend(handles=handles,loc='outside lower center',ncols=5,facecolor='#454052',labelcolor='white')
        fig.suptitle(f"{shown[0]['tissue']} image {image}, seed {seed}: fixed annotated objects")
        save(fig,f'p30_image_{image}_seed_{seed}.png')
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    ax.barh([f"{r['tissue']} {r['image']} (n={r['dead_total']})" for r in changes],[r['net'] for r in changes],color=['#27788a' if r['net']>=0 else '#ae513d' for r in changes]);ax.invert_yaxis();ax.axvline(0,color='#555',lw=1);ax.grid(axis='x',alpha=.2)
    ax.set(title=f'Seed {seed}: all 11 interior-Dead evaluation images',xlabel='Middle minus last-12 matched nuclei')
    save(fig,f'p30_dead_by_image_seed_{seed}.png')
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    for arm,label,color in [('C','Last 12','#156b82'),('M','Middle 12','#a94525')]:
        r=analysis['runs'][f'{arm}_{seed}'];axes[0].plot(range(1,131),r['loss_curve'],label=label,color=color)
        axes[1].plot([0,1],[r['fit']['recall'],r['audit']['recall']],marker='o',label=label,color=color)
    axes[0].set(xlabel='Epoch',ylabel='Training focal loss',yscale='log',title=f'Seed {seed}: full training trajectory')
    axes[1].set(xticks=[0,1],xticklabels=['Training images','Development images'],ylabel='All-nucleus center recall',ylim=[.5,1.02],title='Training and development coverage')
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,f'p30_training_seed_{seed}.png')
with (out/'fixed_objects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(objects[0]));w.writeheader();w.writerows(objects)
(out/'object_summary.json').write_text(json.dumps(summary,indent=2))
stage=list(csv.DictReader(args.prior_stage_pairs.open()))
for seed in seeds:
    c,b=[analysis['runs'][f'{a}_{seed}']['audit'] for a in ['C','M']]
    stage.append(dict(pair=f'P30_{seed}',net_dead=b['dead_hit']-c['dead_hit'],net_all=b['matched']-c['matched'],net_extra=(b['candidates']-b['matched'])-(c['candidates']-c['matched'])))
labels=['P20: input 512','P21: class-center supervision','P22: all 24 blocks','P23: parameter averaging','P24: higher late-block rate','P25: layerwise rates','P26: pretrained-weight anchor','P27: first 12 blocks','P28: alternating 12 blocks','P29: first 6 + last 6','P30: middle 12 blocks']
for key,title in [('net_dead','Matched interior Dead nuclei'),('net_all','All matched nuclei'),('net_extra','Unmatched predictions')]:
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    for seed,marker,color,shift in [(19,'o','#156b82',-.12),(1,'D','#a94525',.12)]:
        rows=[r for r in stage if r['pair'].endswith(f'_{seed}')]
        values=[int(r[key]) for r in rows];ax.scatter(values,np.arange(len(values))+shift,s=60,marker=marker,color=color,label=f'Seed {seed}')
    if len(seeds)==1:ax.text(0,9.18,'P30 seed 1 is still running',fontsize=11,color='#666')
    ax.axvline(0,color='#777',lw=1);ax.grid(axis='x',alpha=.22);ax.invert_yaxis()
    ax.set(yticks=np.arange(11),yticklabels=labels,xlabel='Intervention minus contemporaneous control: count change',title=title)
    ax.legend(frameon=False);save(fig,f'stage_{key}.png')
with (out/'stage_pairs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['pair','net_dead','net_all','net_extra']);w.writeheader();w.writerows(stage)
print(json.dumps(summary,indent=2))
