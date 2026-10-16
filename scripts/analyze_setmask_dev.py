#!/usr/bin/env python
"""Frozen E6 fold2 endpoints and admission. No model selection or test-fold access."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(19,1)
BOOT_SEED=20261015


def object_rates(max_iou):
    values=np.asarray(max_iou,float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Invalid geometric IoU')
    return {'matched':values>.5,'miss':values<.1}


def paired_comparison(rows_by_run,n_boot=1000):
    from nucseg.visibility_probe import resample_rows
    tables={};keys=None
    for seed in SEEDS:
        for arm in ('B','Q'):
            rr=[r for r in rows_by_run[(arm,seed)] if r['cls']==4 and r['interior']]
            table={(r['image'],r['gt_id']):r for r in rr}
            if len(table)!=len(rr):raise ValueError('Duplicate object identity')
            if keys is None:keys=set(table)
            if set(table)!=keys:raise ValueError('Paired object identities differ')
            tables[(arm,seed)]=table
    if not keys or n_boot<1:raise ValueError('Empty paired population or bootstrap')
    keys=sorted(keys);reference=tables[('B',19)]
    for table in tables.values():
        if any(table[k]['tissue']!=reference[k]['tissue'] for k in keys):
            raise ValueError('Paired tissue identity differs')
    images=np.array([k[0] for k in keys],int)
    tissues=np.array([reference[k]['tissue'] for k in keys])
    matched={};miss={}
    for seed in SEEDS:
        rates={arm:object_rates([tables[(arm,seed)][k]['max_iou'] for k in keys]) for arm in ('B','Q')}
        matched[seed]=rates['Q']['matched'].astype(float)-rates['B']['matched']
        miss[seed]=rates['B']['miss'].astype(float)-rates['Q']['miss']
    rng=np.random.default_rng(BOOT_SEED);boot=np.empty((n_boot,2))
    for j in range(n_boot):
        picked=resample_rows(images,tissues,rng)
        boot[j]=[np.mean([diff[s][picked].mean() for s in SEEDS]) for diff in (matched,miss)]
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    point={str(s):float(matched[s].mean()) for s in SEEDS}
    miss_point={str(s):float(miss[s].mean()) for s in SEEDS}
    return {'n_objects':len(keys),'n_images':len(np.unique(images)),
            'per_seed_matched_improvement':point,'matched_improvement':float(np.mean(list(point.values()))),
            'lo':float(lo[0]),'hi':float(hi[0]),'per_seed_miss_improvement':miss_point,
            'miss_improvement':float(np.mean(list(miss_point.values()))),'miss_lo':float(lo[1]),'miss_hi':float(hi[1]),
            'n_boot':n_boot,'bootstrap_seed':BOOT_SEED}


def admission(comparison,endpoints,cost):
    deltas={key:{str(s):float(endpoints[f'Q_seed{s}'][key]-endpoints[f'B_seed{s}'][key]) for s in SEEDS}
            for key in ('mPQ','bPQ','strict_mPQ','strict_dead_PQ','dead_PQ','merge_rate')}
    primary=all(comparison['per_seed_matched_improvement'][str(s)]>0 for s in SEEDS)
    primary=primary and comparison['matched_improvement']>=.02 and comparison['lo']>0
    dead=all(v>=0 for v in deltas['dead_PQ'].values()) and np.mean(list(deltas['dead_PQ'].values()))>=.005
    floors=all(v>=-.002 for k in ('mPQ','bPQ','strict_dead_PQ') for v in deltas[k].values())
    strict=all(v>=0 for v in deltas['strict_mPQ'].values())
    merge=all(v<=.002 for v in deltas['merge_rate'].values())
    latency=all(str(s) in cost and np.isfinite(cost[str(s)]) and 0<cost[str(s)]<=2 for s in SEEDS)
    finite=all(np.isfinite(v) for values in deltas.values() for v in values.values())
    criteria={'interior_dead_matched':bool(primary),'official_dead_pq':bool(dead),
              'quality_floors':bool(floors),'strict_mpq_nondecrease':bool(strict),
              'merge_guard':bool(merge),'latency_each_seed':bool(latency),'finite':bool(finite)}
    return {'admission':all(criteria.values()),'criteria':criteria,'deltas':deltas,'cost_ratios':cost}


def verify_metadata(root):
    """Reject incompatible/incomplete rounds before opening any fold arrays."""
    from nucseg.setmask.engine import load_config,make_provenance,sha_file
    root=Path(root);configs={};sources={};first=None;logs={}
    for seed in SEEDS:
        for arm in ('B','Q'):
            name=f'{arm}_seed{seed}';run=root/name;cfg=load_config(run)
            if cfg.arm!=arm or cfg.seed!=seed or cfg.workers!=8 or Path(cfg.out_dir).resolve()!=run.resolve():
                raise ValueError('Round/config identity differs')
            p=json.loads((run/'provenance.json').read_text())
            if first is None:
                first=p
                if make_provenance(cfg)!=p:raise ValueError('Current source/input identity differs')
            if p!=first:raise ValueError('Unpaired source/input identity')
            log=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
            if [r['epoch'] for r in log]!=list(range(130)):raise ValueError('Incomplete or repeated epoch log')
            logs[(arm,seed)]=log
            receipt=json.loads((run/'completion.json').read_text())
            paths=('config.json','provenance.json','log.jsonl','final.pth','pred_fold2.npz','eval_fold2/summary.json')
            hashes={path:sha_file(run/path) for path in paths}
            if receipt.get('hashes')!=hashes or receipt.get('stage_exit_codes')!={'train':0,'predict':0,'evaluate':0}:
                raise ValueError('Completion provenance is missing or changed')
            sources[name]=hashes;configs[name]=cfg
        if [r['data_digest'] for r in logs[('B',seed)]]!=[r['data_digest'] for r in logs[('Q',seed)]]:
            raise ValueError('B/Q augmented sample streams differ')
    return configs,sources,first


def subgroup_support(rows_by_run):
    """Descriptive five-class/tissue support and geometric deltas, without admission gates."""
    output={}
    reference=rows_by_run[('B',19)]
    for group in ('cls','tissue'):
        output[group]={}
        for value in sorted({r[group] for r in reference},key=str):
            rates={};support=None
            for (arm,seed),rr in rows_by_run.items():
                subset=[r for r in rr if r[group]==value];n=len(subset)
                if support is None:support=n
                if n!=support:raise ValueError('Subgroup support differs')
                rate=object_rates([r['max_iou'] for r in subset])
                rates[f'{arm}_seed{seed}']={k:float(v.mean()) for k,v in rate.items()}
            delta={str(s):{k:rates[f'Q_seed{s}'][k]-rates[f'B_seed{s}'][k] for k in ('matched','miss')} for s in SEEDS}
            output[group][str(value)]={'n_objects':support,'rates':rates,'Q_minus_B':delta}
    return output


def analyze_round(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Analysis output exists; no overwrite')
    configs,sources,provenance=verify_metadata(root)
    from nucseg.data.pannuke import PanNukeFold
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_fusion_arms import load_run
    from nucseg.setmask.engine import sha_file
    latency_path=root/'latency.json';latency=json.loads(latency_path.read_text())
    if latency['run_hashes']!={name:{k:v for k,v in hashes.items() if k in ('config.json','provenance.json','final.pth')}
                               for name,hashes in sources.items()}:
        raise ValueError('Benchmark/checkpoint provenance differs')
    fold=PanNukeFold(2);rows={};endpoints={}
    for name,cfg in configs.items():
        result=load_run(root/name,fold);rows[(cfg.arm,cfg.seed)]=result['rows'];endpoints[name]=result['endpoints']
        if result['endpoints']['interior_dead_n']!=741:raise ValueError('Frozen741 interior Dead population changed')
    comparison=paired_comparison(rows);verdict=admission(comparison,endpoints,latency['ratios'])
    payload={'round':'E6','split':1,'evaluation_fold':2,'seeds':list(SEEDS),'comparison':comparison,
             'verdict':verdict,'endpoints':endpoints,'descriptive_subgroups':subgroup_support(rows),
             'sources':sources,'training_provenance':provenance,'latency_sha256':sha_file(latency_path),
             'limitations':['Development fold2 reused; this is not held-out test performance.',
                            'Two fixed seeds; no seed-population inference or threshold/subgroup selection.',
                            'Any benefit belongs to the entire recipe, not an isolated removal of HV.']}
    out.mkdir(parents=True)
    (out/'endpoints.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze_round(a.root,a.out)['verdict'],indent=2))
