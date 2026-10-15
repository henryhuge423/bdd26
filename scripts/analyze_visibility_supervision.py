#!/usr/bin/env python
"""Frozen visibility dev mechanism analysis; fold2 only, no training or model selection.

C2 geometry and decoding reuse the corrected E4 runner. Original-image predictions
are read from each completed run. Final checkpoints only; paired tissue-stratified
image bootstrap averages the two fixed seeds, not independent nucleus/seed draws.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'runs/visibility_supervision_20261014'
SEEDS = (19, 1)
BOOT_SEED = 20261013

def validate_request(round_dir, fold):
    if Path(round_dir).resolve() != ROUND.resolve() or fold != 2:
        raise ValueError('Only the frozen visibility round and validation fold2 are allowed')

def paired_primary(rows, n_boot=1000, left='A1', right='A2'):
    from nucseg.constants import TISSUES
    chosen = [r for r in rows if r['cls']==4 and r['group']=='truncated'
              and r['arm'] in (left,right)]
    maps = {}
    for s in SEEDS:
        for arm in (left,right):
            rr = [r for r in chosen if r['seed']==s and r['arm']==arm]
            m = {(r['image'],r['gt_id']):r for r in rr}
            if len(m)!=len(rr):
                raise ValueError('duplicate object identity')
            maps[s,arm] = m
    keys = sorted(maps[19,left])
    if not keys or any(set(m)!=set(keys) for m in maps.values()):
        raise ValueError('paired object identities differ across arms/seeds')
    images = sorted({k[0] for k in keys})
    pos = {im:i for i,im in enumerate(images)}
    counts = np.zeros(len(images),int)
    diffs = {s:np.zeros(len(images),float) for s in SEEDS}
    tissue = {}
    rates = {}
    for s in SEEDS:
        rates[str(s)] = {a:float(np.mean([maps[s,a][k]['miss'] for k in keys])) for a in (left,right)}
    for key in keys:
        j = pos[key[0]]
        counts[j] += 1
        ts = {m[key]['tissue'] for m in maps.values()}
        if len(ts)!=1: raise ValueError('tissue identity mismatch')
        tissue[key[0]] = ts.pop()
        for s in SEEDS:
            diffs[s][j] += int(maps[s,left][key]['miss'])-int(maps[s,right][key]['miss'])
    if any(t not in TISSUES for t in tissue.values()):
        raise ValueError('unknown tissue')
    point = {str(s):float(diffs[s].sum()/counts.sum()) for s in SEEDS}
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(idx,len(idx),replace=True)
            for t in TISSUES if len(idx:=np.array([pos[i] for i in images if tissue[i]==t],int))])
        boot.append(float(np.mean([diffs[s][pick].sum()/counts[pick].sum() for s in SEEDS])))
    lo,hi = np.percentile(boot,[2.5,97.5])
    improvement = float(np.mean(list(point.values())))
    return dict(left=left,right=right,n_objects_per_seed=len(keys),n_images=len(images),
                rates=rates,by_seed_improvement=point,improvement=improvement,
                lo=float(lo),hi=float(hi),n_boot=n_boot,bootstrap_seed=BOOT_SEED,
                **{'pass':bool(all(v>0 for v in point.values()) and improvement>=.03 and lo>0)})

def provenance(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return dict(path=str(path),sha256=h.hexdigest())

def run_one(run, out, fold):
    import cv2
    import torch
    sys.path.insert(0,str(ROOT/'scripts'))
    from probe_visibility import predict_frames, gt_class_map, WINDOW_SEED
    from nucseg.cellvit.engine import TrainConfig, build_model
    from nucseg.visibility_probe import (sample_windows,crop_group,original_full,
        max_iou_table,bin_area_native,stratified_rate_gap_bootstrap)
    from nucseg.metrics.instance import overlap
    from nucseg.metrics.errors import classify_errors
    from nucseg.metrics.pannuke_eval import instance_classes
    cfg = json.loads((run/'config.json').read_text())
    arm, seed = run.name.split('_seed'); seed=int(seed)
    assert cfg['split']==1 and cfg['visibility_arm']==arm and cfg['seed']==seed
    assert cfg['upscale']==1 and cfg['epochs']==130
    assert not cfg['dead_expert'] and not cfg['widen']
    n=len(fold); windows=sample_windows(n,192,WINDOW_SEED)
    frozen=json.loads((ROOT/'runs/analysis/visibility_probe_20261014/probe.json').read_text())
    assert windows.tolist()==frozen['windows']['192'], 'E4 fixed windows differ'
    model=build_model(TrainConfig(split=1,out_dir=str(run)),pretrained=False).cuda()
    model.load_state_dict(torch.load(run/'final.pth',map_location='cpu',weights_only=False)['model'])
    model.eval(); torch.manual_seed(0)
    records=[]; c0records=[]
    sec={c:dict(split=0,merged=0,n_gt=0,n_pred=0,matched=0,typing_correct=0) for c in ('C0','C2')}
    base=dict(np.load(run/'pred_fold2.npz'))
    assert base['inst'].shape==(n,256,256)
    def diagnostic(gt,pi,pt,classes,cond):
        ov=overlap(gt,pi); gs,_,_=classify_errors(ov)
        ss=sec[cond]; ss['split']+=int((gs==2).sum()); ss['merged']+=int((gs==1).sum())
        ss['n_gt']+=int((np.unique(gt)>0).sum()); ss['n_pred']+=int((np.unique(pi)>0).sum())
        ids=np.unique(gt); ids=ids[ids>0]
        _,pc=instance_classes(pi,pt)
        gi,pj=np.nonzero(ov.iou>.5)
        ss['matched']+=len(gi)
        ss['typing_correct']+=sum(classes[int(ids[g])]==int(pc[p]) for g,p in zip(gi,pj))
    for start in range(0,n,256):
        end=min(start+256,n)
        frames=np.stack([cv2.resize(np.ascontiguousarray(fold.images[j][y:y+192,x:x+192]),
                        (256,256),interpolation=cv2.INTER_LINEAR)
                        for j,(y,x) in enumerate(windows[start:end],start)])
        pi,pt,hv=predict_frames(model,frames,batch=16,workers=8)
        for b,j in enumerate(range(start,end)):
            gt=np.asarray(fold.inst[j]); classes=gt_class_map(gt,np.asarray(fold.type[j]))
            full=original_full(gt); areas=np.bincount(gt.ravel()); tissue=str(fold.tissue[j])
            u0=max_iou_table(gt,base['inst'][j])
            for i,u in u0.items():
                c0records.append(dict(arm=arm,seed=seed,image=j,tissue=tissue,gt_id=i,cls=classes[i],
                    group='interior' if full[i] else 'border',area=int(areas[i]),
                    miss=u<.1,matched=u>.5,max_iou=u))
            diagnostic(gt,base['inst'][j],base['type'][j],classes,'C0')
            y,x=windows[j]; crop=gt[y:y+192,x:x+192]
            pred=cv2.resize(pi[b],(192,192),interpolation=cv2.INTER_NEAREST).astype(np.int32)
            typ=cv2.resize(pt[b],(192,192),interpolation=cv2.INTER_NEAREST)
            uc=max_iou_table(crop,pred)
            for i,g in crop_group(gt,(int(y),int(x),192)).items():
                if not full[i] or g['group']=='outside': continue
                records.append(dict(arm=arm,seed=seed,image=j,tissue=tissue,gt_id=i,cls=classes[i],
                    group='truncated' if g['group']=='truncated' else 'stays_full',
                    area=g['area'],miss=uc[i]<.1,matched=uc[i]>.5,max_iou=uc[i]))
            diagnostic(crop,pred,typ,classes,'C2')
        print(f'{run.name}: C2 {end}/{n}',flush=True)
        del pi,pt,hv,frames
    dead=[r for r in c0records if r['cls']==4]
    h1=stratified_rate_gap_bootstrap(
        np.array([r['miss'] if r['group']=='border' else np.nan for r in dead]),
        np.array([r['miss'] if r['group']=='interior' else np.nan for r in dead]),
        np.array([bin_area_native(r['area']) for r in dead]),np.array([r['image'] for r in dead]),
        np.array([r['tissue'] for r in dead]),1000,BOOT_SEED)
    interior=[r for r in dead if r['group']=='interior']
    summary=dict(arm=arm,seed=seed,n_images=n,h1_area_stratified_dead=h1,
        interior_dead=dict(n=len(interior),matched_rate=float(np.mean([r['matched'] for r in interior])),
                           miss_rate=float(np.mean([r['miss'] for r in interior]))),
        split_merge_typing=sec,provenance={k:provenance(p) for k,p in {
            'checkpoint':run/'final.pth','config':run/'config.json','pred_c0':run/'pred_fold2.npz',
            'eval_c0':run/'eval_fold2/summary.json','windows':ROOT/'runs/analysis/visibility_probe_20261014/probe.json',
            'gt':fold.dir/'inst.npy','gt_type':fold.dir/'type.npy','images':fold.dir/'images.npy',
            'script':Path(__file__),'e4_script':ROOT/'scripts/probe_visibility.py',
            'geometry':ROOT/'src/nucseg/visibility_probe.py'}.items()},
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated()))
    with gzip.open(out/f'{run.name}_C2.json.gz','wt') as f: json.dump(records,f)
    with gzip.open(out/f'{run.name}_C0.json.gz','wt') as f: json.dump(c0records,f)
    (out/f'{run.name}_summary.json').write_text(json.dumps(summary,indent=2))
    del model; torch.cuda.empty_cache()
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round',type=Path,default=ROUND)
    p.add_argument('--fold',type=int,default=2)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); validate_request(args.round,args.fold)
    if args.out.exists(): raise SystemExit('output exists; no overwrite/replay')
    args.out.mkdir(parents=True)
    from nucseg.data.pannuke import PanNukeFold
    fold=PanNukeFold(2); allrows=[]; start=time.time()
    for seed in SEEDS:
        for arm in ('A0','A1','A2','A3'):
            import shutil, subprocess
            free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',__import__('os').environ['CUDA_VISIBLE_DEVICES']], text=True).strip())
            if free < 12000 or shutil.disk_usage(ROOT).free < 5*1024**3:
                raise RuntimeError('Resource admission failed; stop without retry')
            (args.out/'status.json').write_text(json.dumps(dict(state='running',run=f'{arm}_seed{seed}',gpu_free_mib=free)))
            allrows.extend(run_one(args.round/f'{arm}_seed{seed}',args.out,fold))
    results={f'{l}-{r}':paired_primary(allrows,left=l,right=r) for l,r in [('A1','A2'),('A0','A2'),('A3','A2')]}
    (args.out/'paired_primary.json').write_text(json.dumps(dict(
        comparisons=results,admission_clause1=results['A1-A2']['pass'],
        admission_overall=False,reason='Original-image clauses 2 and 3 already fail; no confirmation',
        wall_seconds=time.time()-start),indent=2))
    (args.out/'status.json').write_text(json.dumps(dict(state='complete')))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
