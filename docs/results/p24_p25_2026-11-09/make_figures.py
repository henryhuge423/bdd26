"""Rebuild descriptive P24 figures from verified p24_analysis.json; no model inference."""
import argparse,json,pathlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('analysis',type=pathlib.Path);ap.add_argument('out',type=pathlib.Path);a=ap.parse_args()
    d=json.loads(a.analysis.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.facecolor':'white'})
    colors=['#45657b','#d18138'];x=np.arange(2);width=.34
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,title,gate in zip(axs,['dead_recall','precision','recall'],['Interior Dead recall','Unique point precision','Overall point recall'],[.60,.75,None]):
        for j,arm in enumerate(('C','L')):
            vals=[d['runs'][f'{arm}_{s}']['audit'][key] for s in (19,1)]
            ax.bar(x+(j-.5)*width,vals,width,label=f'{arm}: '+('encoder1e-5' if arm=='C' else 'encoder3e-5'),color=colors[j])
            for xx,v in zip(x+(j-.5)*width,vals):ax.text(xx,v+.012,f'{v:.3f}',ha='center',fontsize=9)
        if gate is not None:ax.axhline(gate,color='#a93838',ls='--',lw=1,label=f'Absolute gate {gate:.2f}')
        ax.set(xticks=x,xticklabels=['Seed19','Seed1'],ylim=(0,.86),title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].legend(fontsize=8,loc='lower left');fig.suptitle('P24: stable precision gain, seed-dependent Dead recall',fontsize=14);fig.tight_layout();fig.savefig(out/'p24_metrics.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
    for ax,s in zip(axs,('19','1')):
        rows=d['pairs'][s]['dead_images'];labels=[f"{r['image']} {r['tissue']} (n={r['dead_total']})" for r in rows]
        vals=[r['net'] for r in rows];ax.barh(np.arange(len(rows)),vals,color=[colors[1] if v>=0 else '#8c4851' for v in vals]);ax.axvline(0,c='black',lw=.7)
        ax.set(yticks=np.arange(len(rows)),yticklabels=labels,title=f"Seed{s}: net{sum(vals):+d} /145 Dead",xlabel='L minus C matched objects');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('All 11 Dead-positive audit images; descriptive paired changes',fontsize=13);fig.tight_layout();fig.savefig(out/'p24_dead_images.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    labels=['C19','L19','C1','L1'];keys=['C_19','L_19','C_1','L_1'];xx=np.arange(4)
    for ax,key,title in zip(axs,['dead_recall','recall','precision'],['Interior Dead: fit vs audit','Overall recall: fit vs audit','Precision: fit vs audit']):
        for scope,offset,color in [('fit',-.17,'#57856e'),('audit',.17,'#b16a3e')]:
            vals=[d['runs'][k][scope][key] for k in keys];ax.bar(xx+offset,vals,.34,label=scope,color=color)
        ax.set(xticks=xx,xticklabels=labels,ylim=(0,1.08),title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].legend(fontsize=8);fig.suptitle('Strong fitting coverage does not establish development transfer',fontsize=13);fig.tight_layout();fig.savefig(out/'p24_fit_audit.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    ax=axs[0]
    for i,arm in enumerate(('C','L')):
        pred=[d['runs'][f'{arm}_{s}']['audit']['candidates'] for s in (19,1)];hits=[d['runs'][f'{arm}_{s}']['audit']['matched'] for s in (19,1)]
        positions=x+(i-.5)*width;ax.bar(positions,hits,width,color=colors[i],label=f'{arm} matched');ax.bar(positions,np.array(pred)-hits,width,bottom=hits,color=colors[i],alpha=.35,label=f'{arm} unmatched')
    ax.set(xticks=x,xticklabels=['Seed19','Seed1'],ylabel='Points',title='Fewer unmatched proposals');ax.legend(fontsize=8,ncol=2)
    b=np.arange(24);rates=3e-5*.95**(23-b)
    axs[1].plot(b,rates*1e5,'o-',color='#536e63',label='P25 D all24 layerwise');axs[1].plot(np.arange(12,24),np.ones(12),'s--',color=colors[0],label='P25 C last12 uniform');axs[1].set(xlabel='UNI block',ylabel='Initial LR /1e-5',title='P25 fixed exploratory recipe');axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'p24_candidates_p25_rates.png');plt.close(fig)
    with (out/'paired_dead_images.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','image','tissue','Dead_objects','C_hits','L_hits','net'])
        for seed,pair in d['pairs'].items():
            for r in pair['dead_images']:w.writerow([seed,r['image'],r['tissue'],r['dead_total'],r['C_hits'],r['H_hits'],r['net']])
    print(json.dumps({'png':4,'csv':1,'status':'PASS'}))

if __name__=='__main__':main()
