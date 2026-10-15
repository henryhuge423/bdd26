from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args();p=args.input
c=pd.read_csv(p/'tissue_class_by_split_seed.csv');s=pd.read_csv(p/'unique_fold_support.csv')
ts=list(pd.read_csv(p/'tissue_summary.csv').tissue)
cs=['Neoplastic','Inflammatory','Connective','Dead','Epithelial']
base=c.groupby(['tissue','cls']).PQ.mean().unstack().reindex(index=ts,columns=cs)
change=c.groupby(['tissue','cls']).delta_PQ.mean().unstack().reindex(index=ts,columns=cs)
support=s[s.fold.isin([1,3])].groupby(['tissue','cls']).positive_images.sum().unstack().reindex(index=ts,columns=cs)
fig,axes=plt.subplots(1,2,figsize=(12,10),layout='constrained')
for ax,df,title,cmap,norm in [(axes[0],base,'Frozen x1 + M/S baseline PQ','viridis',None),(axes[1],change,'Frozen EF-P2 minus baseline PQ','RdBu',TwoSlopeNorm(vmin=-.04,vcenter=0,vmax=.04))]:
 cm=plt.get_cmap(cmap).copy();cm.set_bad('#d8d8d8')
 im=ax.imshow(df.to_numpy(),aspect='auto',cmap=cm,norm=norm,**({'vmin':0,'vmax':.75} if norm is None else {}))
 ax.set_yticks(range(19),[t.replace('_',' ') for t in ts]);ax.set_xticks(range(5),['Neo','Inflam','Connect','Dead','Epi']);ax.set_title(title,fontsize=13,pad=13)
 for i in range(19):
  for j in range(5):
   val=df.iloc[i,j]
   if pd.isna(val):txt='NA';color='#555555'
   else:
    txt=(f'{val:.2f}' if norm is None else f'{val:+.3f}')+('*' if support.iloc[i,j]<20 else '')
    color=('white' if val<.35 else 'black') if norm is None else ('white' if abs(val)>.025 else 'black')
   ax.text(j,i,txt,ha='center',va='center',fontsize=8,color=color)
 fig.colorbar(im,ax=ax,shrink=.65,label='PQ' if norm is None else 'Change in PQ')
fig.suptitle('PanNuke tissue × class audit: three splits, three fixed seeds',fontsize=16)
fig.supxlabel('Gray: no GT support in official test folds. *: fewer than 20 unique GT-positive test images.\nDescriptive historical test reuse; folds 1/3 counted once for support; no unseen-tissue experiment.',fontsize=10)
fig.savefig(p/'tissue_class_heatmaps.png',dpi=180);plt.close(fig)
