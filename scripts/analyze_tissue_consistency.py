"""Descriptive audit of frozen PanNuke evaluations; no model/threshold selection."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from nucseg.constants import CLASS_NAMES, TISSUES, SPLITS

warnings.filterwarnings('ignore', category=RuntimeWarning)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); root=a.root; a.out.mkdir(parents=True, exist_ok=False)
    inputs={}; checks=[]; rows=[]; cells=[]; errors=[]; global_errors=[]; arrays={}; summaries=[]
    tissues={f:np.load(root/f'data/pannuke/fold{f}/tissue.npy') for f in (1,2,3)}
    for f in tissues: inputs[f'data/pannuke/fold{f}/tissue.npy']=digest(root/f'data/pannuke/fold{f}/tissue.npy')
    for split in (1,2,3):
        fold=SPLITS[split][2]; ts=tissues[fold]; per=[]; delta=[]; cp=[]; dc=[]
        for seed in (19,1,2):
            pair=f'split{split}_seed{seed}'
            base=root/f'runs/analysis/matched_seed_20261004/p0/{pair}/test/base/ms/eval'
            ef=root/f'runs/analysis/existence_ef_20261005/p2ef/{pair}/test/eval'
            zs=[]; ss=[]
            for d in (base,ef):
                for name in ('summary.json','per_image.npz','gt_records.csv.gz'):
                    inputs[str((d/name).relative_to(root))]=digest(d/name)
                z=dict(np.load(d/'per_image.npz')); s=json.loads((d/'summary.json').read_text())
                assert len(z['mPQ'])==len(ts)
                reconstructed=np.nanmean([np.nanmean(z['mPQ'][ts==t]) for t in TISSUES])
                assert abs(reconstructed-s['official']['mPQ'])<1e-12
                for ti in TISSUES:
                    for c,cn in enumerate(CLASS_NAMES):
                        v=np.nanmean(z['class_PQ'][ts==ti,c]); expected=s['tissue_class_PQ'][ti][cn]
                        assert (np.isnan(v) and np.isnan(expected)) or abs(v-expected)<1e-12
                zs.append(z); ss.append(s)
            g=pd.read_csv(base/'gt_records.csv.gz'); ge=pd.read_csv(ef/'gt_records.csv.gz')
            cols=['image','cls','area','n_touch','tissue']
            pd.testing.assert_frame_equal(g[cols],ge[cols])
            assert np.array_equal(g.tissue.to_numpy(),ts[g.image.to_numpy()])
            for cl in range(1,6):
                q=g[g.cls==cl];m=q[q.status=='matched']
                global_errors.append(dict(split=split,seed=seed,cls=CLASS_NAMES[cl-1],matched_rate=(q.status=='matched').mean(),low_iou_rate=(q.max_iou_x1000<100).mean(),typing_given_match=(m.pred_cls==cl).mean()))
            per.append(zs[0]['mPQ']); delta.append(zs[1]['mPQ']-zs[0]['mPQ'])
            cp.append(zs[0]['class_PQ']); dc.append(zs[1]['class_PQ']-zs[0]['class_PQ'])
            for ti in TISSUES:
                gg=g[g.tissue==ti]
                r={'split':split,'seed':seed,'test_fold':fold,'tissue':ti,'n_images':int((ts==ti).sum())}
                for metric in ('mPQ','bPQ'):
                    r[metric]=ss[0]['per_tissue'][ti][metric]
                    r['delta_'+metric]=ss[1]['per_tissue'][ti][metric]-r[metric]
                rows.append(r)
                for c,cn in enumerate(CLASS_NAMES):
                    q=gg[gg.cls==c+1]; qm=q[q.status=='matched']; qe=ge[(ge.tissue==ti)&(ge.cls==c+1)]
                    cells.append(dict(split=split,seed=seed,test_fold=fold,tissue=ti,cls=cn,n_gt=len(q),positive_images=q.image.nunique(),PQ=ss[0]['tissue_class_PQ'][ti][cn],delta_PQ=ss[1]['tissue_class_PQ'][ti][cn]-ss[0]['tissue_class_PQ'][ti][cn]))
                    if len(q):
                        errors.append(dict(split=split,seed=seed,tissue=ti,cls=cn,n_gt=len(q),matched_rate=(q.status=='matched').mean(),missed_bg_rate=(q.status=='missed_bg').mean(),merged_rate=(q.status=='merged').mean(),typing_given_match=(qm.pred_cls==c+1).mean(),ef_matched_rate=(qe.status=='matched').mean()))
            summaries.append({'split':split,'seed':seed,'baseline':ss[0]['official'],'ef':ss[1]['official'],'Dead':ss[0]['per_class_PQ']['Dead'],'ef_Dead':ss[1]['per_class_PQ']['Dead']})
        arrays[split]={'m':np.nanmean(per,axis=0),'d':np.nanmean(delta,axis=0),'c':np.nanmean(cp,axis=0),'dc':np.nanmean(dc,axis=0),'t':ts}
        checks.append(f'split{split}: three seeds; official and all 95 tissue/class means reproduce; baseline/EF GT rows align')
    tab=pd.DataFrame(rows); cell=pd.DataFrame(cells); err=pd.DataFrame(errors)
    tab.to_csv(a.out/'tissue_by_split_seed.csv',index=False);cell.to_csv(a.out/'tissue_class_by_split_seed.csv',index=False);err.to_csv(a.out/'error_rates.csv',index=False)
    pd.DataFrame(global_errors).to_csv(a.out/'global_class_errors.csv',index=False)
    mean=tab.groupby('tissue').mean(numeric_only=True).reindex(TISSUES)
    # Equal class weights, only the three broadly represented common classes.
    common=cell[cell.cls.isin(CLASS_NAMES[:3])].groupby(['tissue','cls']).PQ.mean().unstack()
    mean['common3_PQ']=common.mean(axis=1)
    mean['bottom5_splits']=sum((tab.groupby(['split','tissue']).mPQ.mean().loc[s].rank()<=5).astype(int) for s in (1,2,3))
    mean.to_csv(a.out/'tissue_summary.csv')
    # Paired image bootstrap, shared image draws for splits 1/2 because both test fold3.
    # Seed predictions stay paired/averaged. No training/patient uncertainty claimed.
    rng=np.random.default_rng(20261015); B=2000; bt=[]; bcm=[]; bd=[]; bcd=[]
    for t in TISSUES:
        mv=[]; cv=[]; dv=[]; dcv=[]
        draws={f:rng.integers(0,int((tissues[f]==t).sum()),size=(B,int((tissues[f]==t).sum()))) for f in (1,3)}
        for split in (1,2,3):
            z=arrays[split]; mask=z['t']==t; ix=draws[SPLITS[split][2]]
            mv.append(np.nanmean(z['m'][mask][ix],axis=1)); dv.append(np.nanmean(z['d'][mask][ix],axis=1))
            # Conditional class means use GT-positive images only. Keep the
            # supported split set fixed, avoiding rare-class absent resamples.
            cv.append(np.full((B,5),np.nan));dcv.append(np.full((B,5),np.nan))
        for c in range(5):
            class_draws={}
            for split in (1,2,3):
                z=arrays[split]; mask=(z['t']==t)&np.isfinite(z['c'][:,c]); n=int(mask.sum()); f=SPLITS[split][2]
                if n==0:continue
                if f not in class_draws:class_draws[f]=rng.integers(0,n,size=(B,n))
                ix=class_draws[f]
                cv[split-1][:,c]=np.mean(z['c'][mask,c][ix],axis=1)
                dcv[split-1][:,c]=np.mean(z['dc'][mask,c][ix],axis=1)
        bt.append(np.mean(mv,axis=0));bd.append(np.mean(dv,axis=0));bcm.append(np.nanmean(cv,axis=0));bcd.append(np.nanmean(dcv,axis=0))
    bt=np.array(bt);bd=np.array(bd);bcm=np.array(bcm);bcd=np.array(bcd)
    intervals=[]
    for k,t in enumerate(TISSUES):
        for name,z in [('mPQ',bt[k]),('delta_mPQ',bd[k])]:
            lo,hi=np.nanpercentile(z,[2.5,97.5]);intervals.append(dict(tissue=t,metric=name,low=lo,high=hi))
        for c,cn in enumerate(CLASS_NAMES):
            for name,z in [('PQ',bcm[k,:,c]),('delta_PQ',bcd[k,:,c])]:
                lo,hi=np.nanpercentile(z,[2.5,97.5]); intervals.append(dict(tissue=t,cls=cn,metric=name,low=lo,high=hi))
    pd.DataFrame(intervals).to_csv(a.out/'image_bootstrap_intervals.csv',index=False)
    # Ground-truth support, once per unique fold (not nine times).
    support=[]
    for f in (1,2,3):
        d=root/('runs/cellvit_uni/split1/eval_fold2' if f==2 else f'runs/analysis/matched_seed_20261004/p0/split{3 if f==1 else 1}_seed19/test/base/ms/eval')
        p=d/'gt_records.csv.gz'; inputs[str(p.relative_to(root))]=digest(p);g=pd.read_csv(p)
        assert np.array_equal(g.tissue.to_numpy(),tissues[f][g.image.to_numpy()])
        for t in TISSUES:
            for c,cn in enumerate(CLASS_NAMES):
                q=g[(g.tissue==t)&(g.cls==c+1)]
                support.append(dict(fold=f,tissue=t,cls=cn,n_images=int((tissues[f]==t).sum()),n_gt=len(q),positive_images=q.image.nunique()))
    pd.DataFrame(support).to_csv(a.out/'unique_fold_support.csv',index=False)
    # Original reproduced model comparison, independent of later M/S changes.
    models=[]
    for model in ('cellvit_uni','hovernet'):
        for split in (1,2,3):
            fold=SPLITS[split][2];d=root/f'runs/{model}/split{split}/eval_test_fold{fold}'
            if not (d/'summary.json').exists(): d=root/f'runs/{model}/split{split}/eval_fold{fold}'
            p=d/'summary.json'; inputs[str(p.relative_to(root))]=digest(p);s=json.loads(p.read_text())
            for t in TISSUES: models.append(dict(model=model,split=split,tissue=t,**s['per_tissue'][t]))
    pd.DataFrame(models).to_csv(a.out/'original_model_comparison.csv',index=False)
    result={'scope':'descriptive audit of existing test artifacts; no threshold/model selection; future development must use train/val',
      'bootstrap':{'draws':B,'seed':20261015,'unit':'tissue-stratified image; class PQ conditional on GT-positive images; shared fold3 draws across splits1/2','limits':'conditional on trained seeds and GT-positive class support; no patient ids; pointwise exploratory intervals, no multiplicity correction'},
      'checks':checks,'split_seed_scores':summaries,'tissue_summary':mean.reset_index().to_dict('records'),
      'inputs_sha256':inputs,'script_sha256':digest(Path(__file__))}
    (a.out/'audit.json').write_text(json.dumps(result,indent=2))
    print(mean[['mPQ','bPQ','common3_PQ','delta_mPQ','bottom5_splits']].sort_values('mPQ').to_string())
    print('classes',cell.groupby('cls')[['PQ','delta_PQ']].mean().to_string())
    print('support Dead',pd.DataFrame(support).query("cls == 'Dead'").groupby('fold')[['n_gt','positive_images']].sum().to_string())
    print('CHECKS',checks)

if __name__=='__main__': main()
