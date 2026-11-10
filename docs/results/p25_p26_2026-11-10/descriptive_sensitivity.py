"""Post-result sensitivity descriptions; never change the frozen admission decision."""
import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).parent;analysis=json.loads((root/'analysis.json').read_text());result={'post_hoc':True,'changes_original_gate':False,'leave_one_positive_image_out':{},'fixed_examples':{}}
for seed,pair in analysis['pairs'].items():
    rows=[]
    for image in pair['dead_images']:
        denominator=145-image['dead_total'];net=pair['net_dead_hits']-image['net']
        rows.append(dict(excluded_image=image['image'],excluded_tissue=image['tissue'],remaining_Dead=denominator,delta=net/denominator))
    result['leave_one_positive_image_out'][seed]=dict(rows=rows,min=min(r['delta'] for r in rows),max=max(r['delta'] for r in rows))
    a=json.loads((root/f'raw/A_C_s{seed}/point_examples_audit/coordinates.json').read_text())['records']
    b=json.loads((root/f'raw/A_D_s{seed}/point_examples_audit/coordinates.json').read_text())['records']
    assert [r['image'] for r in a]==[r['image'] for r in b]
    totals={k:0 for k in ('Dead_objects','retained','gained','lost','missed_both')};records=[]
    for c,l in zip(a,b):
        assert c['gt']==l['gt'];dead={i for i,g in enumerate(c['gt']) if g[0]==4 and g[2]}
        cm={p[0] for p in c['unique_pairs']}&dead;lm={p[0] for p in l['unique_pairs']}&dead
        assert len(cm)==c['score']['dead_hit'] and len(lm)==l['score']['dead_hit']
        row=dict(image=c['image'],Dead_objects=len(dead),retained=len(cm&lm),gained=len(lm-cm),lost=len(cm-lm),missed_both=len(dead-(cm|lm)))
        assert sum(row[k] for k in ('retained','gained','lost','missed_both'))==len(dead)
        for k in totals:totals[k]+=row[k]
        records.append(row)
    result['fixed_examples'][seed]=dict(audit_images=8,totals=totals,records=records,scope='predetermined GT-stratified examples only; not full522 audit')
(root/'descriptive_sensitivity.json').write_text(json.dumps(result,indent=2))
with (root/'figures/tissue_changes.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['seed','tissue','C_matched','D_matched','C_candidates','D_candidates','recall_delta','precision_delta','Dead_objects','Dead_net'])
    for seed in ('19','1'):
        c=analysis['runs']['C_'+seed]['tissues'];l=analysis['runs']['D_'+seed]['tissues'];assert c.keys()==l.keys()
        for tissue in c:
            x,y=c[tissue],l[tissue];assert x['total']==y['total'] and x['dead_total']==y['dead_total']
            w.writerow([seed,tissue,x['matched'],y['matched'],x['candidates'],y['candidates'],y['recall']-x['recall'],y['precision']-x['precision'],x['dead_total'],y['dead_hit']-x['dead_hit']])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,seed in zip(axs,('19','1')):
    rows=result['leave_one_positive_image_out'][seed]['rows'];ax.plot(range(len(rows)),[r['delta'] for r in rows],'o-',c='#3d6c77');ax.axhline(analysis['pairs'][seed]['delta'],c='#d18138',ls='--',label='Original all11 images');ax.axhline(0,c='black',lw=.6);ax.set(xticks=range(len(rows)),xticklabels=[r['excluded_image'] for r in rows],xlabel='Excluded positive image',ylabel='Descriptive Dead recall change',title='Seed'+seed);ax.tick_params(axis='x',rotation=65);ax.legend(fontsize=8)
fig.suptitle('Sensitivity to one image: descriptive, original gate unchanged');fig.tight_layout();fig.savefig(root/'figures/p25_leave_one_image.png',dpi=150);plt.close(fig)
print(json.dumps(result,indent=2))
