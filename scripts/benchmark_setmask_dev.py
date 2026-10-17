#!/usr/bin/env python
"""E6 fixed-frame, batch1 end-to-end GPU latency; setup/disk IO excluded."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from nucseg.data.pannuke import PanNukeFold
from nucseg.setmask.engine import load_final,infer_batch,sha_file

FRAME_IDS=tuple(range(64))+(1063,1515)


def comparable_load(observations):
    # A stable nonzero competing workload does not establish equal execution load.
    # Conservatively admit only observations with no foreign GPU processes.
    return bool(observations) and all(not row for row in observations)


def foreign_workload(uuid):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                  '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in text.strip().splitlines():
        fields=[s.strip() for s in line.split(',')]
        if fields[0]==uuid and int(fields[1])!=os.getpid():
            rows.append({'pid':int(fields[1]),'used_memory_mib':fields[2]})
    return rows


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader'],text=True)


def benchmark(root,device='cuda'):
    root=Path(root);output=root/'latency.json'
    if output.exists():raise ValueError('Latency output exists; no overwrite')
    if torch.device(device).type!='cuda' or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one admitted GPU required')
    # Fixed train images, copied before timing. No training labels or fold3 access.
    fold=PanNukeFold(1);frames=np.array(fold.images[list(FRAME_IDS)],copy=True)
    del fold
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not physical or ',' in physical:raise ValueError('Explicit single physical GPU identity required')
    uuid=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    results={};hashes={};before=gpu_snapshot();loads=[foreign_workload(uuid)]
    for seed in (19,1):
        for arm in ('B','Q'):
            name=f'{arm}_seed{seed}';run=root/name
            cfg,model=load_final(run,device)
            if cfg.arm!=arm or cfg.seed!=seed:raise ValueError('Benchmark identity differs')
            hashes[name]={path:sha_file(run/path) for path in ('config.json','provenance.json','final.pth')}
            for image in frames[:16]:infer_batch(model,image[None],arm,device)
            torch.cuda.synchronize();values=[];load_before=gpu_snapshot();loads.append(foreign_workload(uuid))
            for repeat in range(3):
                lap=[]
                for image in frames:
                    loads.append(foreign_workload(uuid))
                    torch.cuda.synchronize();start=time.perf_counter()
                    result=infer_batch(model,image[None],arm,device)
                    torch.cuda.synchronize();lap.append(time.perf_counter()-start)
                    del result
                    loads.append(foreign_workload(uuid))
                values.append(lap)
            results[name]={'seconds':values,'mean_seconds':float(np.mean(values)),
                           'median_seconds':float(np.median(values)),
                           'load_before':load_before,'load_after':gpu_snapshot()}
            del model;gc.collect();torch.cuda.empty_cache()
    ratios={str(seed):results[f'Q_seed{seed}']['mean_seconds']/results[f'B_seed{seed}']['mean_seconds'] for seed in (19,1)}
    median_ratios={str(seed):results[f'Q_seed{seed}']['median_seconds']/results[f'B_seed{seed}']['median_seconds'] for seed in (19,1)}
    payload={'frame_ids':FRAME_IDS,'fold':1,'batch_size':1,'warmup':16,'repeats':3,'results':results,
             'ratios':ratios,'median_ratios':median_ratios,'run_hashes':hashes,
             'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),'device':torch.cuda.get_device_name(),
             'load_before':before,'load_after':gpu_snapshot(),'selected_gpu_uuid':uuid,
             'foreign_workload_observations':loads,'load_comparable':comparable_load(loads),
             'scope':'normalization,H2D,forward,full native decoding,overlap and CPU output; disk/model loading excluded'}
    with output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(benchmark(a.root)['ratios']))
