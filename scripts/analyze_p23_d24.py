"""Reproduce P23 aggregate/paired checks and descriptive D24 partial analysis.

Inputs are retained compact evidence, not a new checkpoint evaluation.
Run: python analyze_p23_d24.py INPUT_DIRECTORY OUTPUT_DIRECTORY
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, sys, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
def dump(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
p23=json.loads((src/'p23_preliminary_analysis.json').read_text())
records=[json.loads(s) for s in (src/'d24_partial.jsonl').read_text().splitlines()]
assert len(records)==4 and {r['tag'] for r in records}=={'P22_'+a+'_'+s for a in ('C','M') for s in ('19','1')}
bytag={r['tag']:r for r in records};pairs={};bootstrap={}
for seed in ('19','1'):
    c,m=[bytag['P22_'+a+'_'+seed] for a in ('C','M')]
    keyed=[{(v['image'],v['gt_index']):v for v in r['dead_objects']} for r in (c,m)]
    assert len(keyed[0])==len(keyed[1])==145 and keyed[0].keys()==keyed[1].keys()
    transitions=Counter();tissues=defaultdict(lambda:Counter());rows=[]
    for k in sorted(keyed[0]):
        a,b=keyed[0][k],keyed[1][k]
        assert all(a[n]==b[n] for n in ('center','area','kind','interior','tissue'))
        assert a['kind']==4 and a['interior']
        ca,cb=a['reason']=='matched',b['reason']=='matched'
        transitions[a['reason']+' -> '+b['reason']]+=1
        state='retained' if ca and cb else 'gained' if cb else 'lost' if ca else 'still_missed'
        tissues[a['tissue']][state]+=1
        rows.append(dict(image=k[0],gt_index=k[1],tissue=a['tissue'],area=a['area'],state=state,C=a['radius3_max'],M=b['radius3_max']))
    states=Counter(r['state'] for r in rows)
    assert states['gained']-states['lost']==m['counts']['interior_dead']['matched']-c['counts']['interior_dead']['matched']
    pairs[seed]=dict(transitions=dict(transitions),states=dict(states),tissues={k:dict(v) for k,v in tissues.items()},objects=rows,
        low_response_share_of_misses={a:r['counts']['interior_dead']['low_response']/(145-r['counts']['interior_dead']['matched']) for a,r in [('C',c),('M',m)]})
    positive=p23['pairs'][seed]['dead_images'];assert len(positive)==11 and sum(r['dead_total'] for r in positive)==145
    groups=defaultdict(list)
    for i,r in enumerate(positive):groups[r['tissue']].append(i)
    rng=np.random.default_rng(20261018);draws=[]
    for _ in range(1000):
        ids=np.concatenate([rng.choice(v,len(v),replace=True) for _,v in sorted(groups.items())])
        draws.append(sum(positive[i]['H_hits']-positive[i]['C_hits'] for i in ids)/sum(positive[i]['dead_total'] for i in ids))
    ci=np.percentile(draws,[2.5,97.5]).tolist()
    np.testing.assert_allclose(ci,p23['pairs'][seed]['ci95'],rtol=0,atol=1e-14)
    sums=[p23['summaries'][f'trajectory_s{seed}/{arm}/evaluation.json'] for arm in ('C','E')]
    for arm,r in zip(('C','E'),sums):
        for key,num,den in [('recall','matched','total'),('precision','matched','candidates'),('dead_recall','dead_hit','dead_total')]:
            assert abs(r[key]-r[num]/r[den])<1e-14
    csum,esum=sums;delta=esum['dead_recall']-csum['dead_recall']
    checks=dict(support=esum['dead_images']>=10 and esum['dead_total']>=100,
        absolute_dead_recall=esum['dead_recall']>=.60,gain=delta>=.05,
        positive_bootstrap_lower=ci[0]>0,absolute_precision=esum['precision']>=.75,
        precision_noninferiority=esum['precision']>=csum['precision']-.02,
        overall_recall_noninferiority=esum['recall']>=csum['recall']-.02,nondegeneracy=esum['nondeg']>=.95)
    assert checks==p23['pairs'][seed]['checks']
    bootstrap[seed]=dict(ci95=ci,delta=delta,checks=checks,passed=all(checks.values()))
result=dict(status='REPRODUCED_FROM_RETAINED_COMPACT_EVIDENCE',numpy=np.__version__,P23=bootstrap,D24_P22=pairs,
    limitations=['Repeated fold1 development; not PQ/test or patient-independent inference.','D24 four P22 models retrieved; missing P23 object rows are not imputed.','P23 522-image raw files not freshly retrieved in this run; arithmetic and bootstrap reconstructed from retained summaries and eleven positive-image rows.'],
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()})
dump('analysis.json',result)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
colors=['#2d5d7b','#c66b3d']
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,metric,title,gate in zip(axes,['dead_recall','precision','recall'],['Interior Dead recall','Unique precision','Overall recall'],[.60,.75,None]):
    for j,arm in enumerate(('C','E')):
        vals=[p23['summaries'][f'trajectory_s{s}/{arm}/evaluation.json'][metric] for s in ('19','1')]
        bars=ax.bar(np.arange(2)+(j-.5)*.34,vals,.32,color=colors[j],label='Online C' if arm=='C' else 'EMA E (.999)')
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    if gate:ax.axhline(gate,color='#444',ls='--',lw=1,label=f'Original gate {gate:.2f}')
    ax.set(xticks=[0,1],xticklabels=['Seed 19','Seed 1'],ylim=(0,.86),title=title)
axes[0].legend(fontsize=8,loc='upper left');fig.suptitle('P23: small EMA gains do not pass the original gate\nFinal epoch 130; fold1 development audit; 145 interior Dead objects / 11 positive images',fontsize=12)
fig.savefig(out/'p23_metrics.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,seed in zip(axes,('19','1')):
    bottom=np.zeros(2)
    for reason,color,label in [('matched','#2d5d7b','Matched'),('low_response','#c66b3d','Radius-3 maximum < .5'),('localization_or_nms','#b9b7b0','Localization / NMS')]:
        vals=[bytag[f'P22_{a}_{seed}']['counts']['interior_dead'].get(reason,0) for a in ('C','M')]
        bars=ax.bar([0,1],vals,bottom=bottom,color=color,label=label)
        ax.bar_label(bars,labels=[str(v) if v else '' for v in vals],label_type='center',color='white' if reason!='localization_or_nms' else 'black')
        bottom+=vals
    ax.set(xticks=[0,1],xticklabels=['C: last12','M: full24'],ylim=(0,160),title=f'Seed {seed}',ylabel='Interior Dead objects (n=145)')
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=1,fontsize=8)
fig.suptitle('D24 partial diagnostics: weak local response dominates P22 misses\nFixed threshold .5 / radius 3; descriptive labels do not establish a causal mechanism',fontsize=12)
fig.savefig(out/'d24_miss_reasons.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
palette={'retained':'#2d5d7b','gained':'#1b8d73','lost':'#c44b49','still_missed':'#b5b7ba'}
for ax,seed in zip(axes,('19','1')):
    rows=pairs[seed]['objects']
    for state,color in palette.items():
        subset=[r for r in rows if r['state']==state]
        ax.scatter([r['C'] for r in subset],[r['M'] for r in subset],s=19,alpha=.7,color=color,label=f'{state.replace("_"," ")}: {len(subset)}')
    ax.plot([0,1],[0,1],color='#666',lw=.7);ax.axhline(.5,color='#777',ls='--',lw=.8);ax.axvline(.5,color='#777',ls='--',lw=.8)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='C: maximum probability within radius 3',ylabel='M: maximum probability within radius 3',title=f'Seed {seed}');ax.legend(fontsize=8,loc='upper left')
fig.suptitle('D24 / P22: object-paired response and unique-match transitions\nSame 145 interior Dead objects; matching can also depend on localization / NMS',fontsize=12)
fig.savefig(out/'d24_paired_response.png');plt.close(fig)
print(json.dumps(dict(P23=bootstrap,D24={s:{k:v for k,v in p.items() if k!='objects'} for s,p in pairs.items()}),indent=2))
