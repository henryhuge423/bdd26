"""Descriptive all-Dead-positive training cohort audit using frozen E7 query models."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--snapshot', type=Path, required=True)
p.add_argument('--helper', type=Path, required=True)
p.add_argument('--run-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
os.chdir(a.snapshot); sys.path.insert(0, str(a.snapshot/'src'))
from nucseg.setmask.engine import load_config, build_model, move_batch, sha_file
from nucseg.setmask.data import PairedDataset
from nucseg.setmask.matching import match_queries
from nucseg.setmask.decode import decode_set
from nucseg.visibility_probe import original_full
from nucseg.metrics.instance import overlap
assert sha_file(a.helper) == 'f2158f283d082f23c9fbdcd19e8beee74170752d86689617671cf5c39006409c'
spec = importlib.util.spec_from_file_location('frozen_alignment_helper', a.helper)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
torch.cuda.set_per_process_memory_fraction(.30)
a.out.mkdir(exist_ok=False)
ds = PairedDataset(1, False, 19)
ids = np.flatnonzero(ds.class_presence()[:, 3]).tolist()
assert len(ids) == 65
source = json.loads((a.snapshot/'MANIFEST.json').read_text())
for group in ('sources', 'vendor'):
    for name, digest in source[group].items():
        assert sha_file(a.snapshot/name) == digest, name
manifest = {'fold': 1, 'images': ids, 'batch_size': 1, 'augmentation': False,
            'script_sha256': sha_file(__file__), 'helper_sha256': sha_file(a.helper),
            'source_commit': source['commit'], 'checkpoints': {},
            'limits': 'GT-selected training cohort; descriptive; no fitting, no validation or test results'}
results = {}; started = time.monotonic()
for seed in (19, 1):
    run = a.run_root/f'Q_seed{seed}'; cfg = load_config(run)
    assert cfg.round == 'E7' and cfg.arm == 'Q' and cfg.seed == seed
    receipt = json.loads((run/'completion.json').read_text())
    assert sha_file(run/'final.pth') == receipt['hashes']['final.pth']
    state = torch.load(run/'final.pth', map_location='cpu', weights_only=False, mmap=True)
    assert state['config'] == asdict(cfg)
    assert state['provenance']['torch'] == torch.__version__ and state['provenance']['numpy'] == np.__version__
    for name in ('images', 'inst', 'type', 'tissue'):
        assert sha_file(ds.fold.dir/(name+'.npy')) == state['provenance']['inputs']['fold1/'+name+'.npy']
    model = build_model(cfg, pretrained=False).eval().to('cuda')
    model.load_state_dict(state['model']); del state
    names = ('n','assigned_raw_good','any_raw_good','accepted_raw_good','typed_accepted_raw_good',
             'canonical_bg','canonical_kept','canonical_correct_type','raw_good_rejected_bg',
             'raw_good_rejected_lowconf','raw_good_kept_wrong_type','final_good','final_typed_good','final_miss')
    groups = {str(c): {n:0 for n in names} for c in range(1,6)}
    groups.update({key:{n:0 for n in names} for key in ('Dead_interior','Dead_border')})
    confusion = np.zeros((5,6), dtype=np.int64); rows = []; n_pred = 0
    for image in ids:
        batch = move_batch(default_collate([ds[image]]), 'cuda', 'Q')
        cache = helper.cache_batch(model, batch); pred = cache['pred']; target = cache['targets'][0]
        qi,ti = match_queries(pred['class_logits'],pred['mask_embed'],pred['mask_features'],[target])[0]
        by_target = dict(zip(ti.tolist(),qi.tolist()))
        score,guess = pred['class_logits'][0].float().softmax(-1).max(-1)
        kept = (guess>0)&(score>=.5); inst,_ = decode_set(pred); pi=inst[0]
        pids=np.unique(pi);pids=pids[pids>0];n_pred+=len(pids)
        iou=overlap(target['inst'].numpy(),pi).iou
        assert iou.shape==(len(target['ids']),len(pids))
        interior=original_full(target['inst'].numpy())
        for t,(oid,cls) in enumerate(zip(target['ids'].tolist(),target['classes'].tolist())):
            q=by_target[t]; mat=cache['ious'][0][:,t]
            raw=float(mat.max());assigned=float(mat[q]);accepted=float(mat[kept].max()) if bool(kept.any()) else 0.
            typed=kept&(guess==cls);typed_raw=float(mat[typed].max()) if bool(typed.any()) else 0.
            assert raw+1e-7>=accepted>=typed_raw-1e-7
            best=float(iou[t].max()) if len(pids) else 0.
            pc=int(guess[int(pids[iou[t].argmax()])-1]) if best>.5 else 0
            label=int(guess[q]);is_kept=bool(kept[q]);confusion[cls-1,label]+=1
            values={'n':1,'assigned_raw_good':int(assigned>.5),'any_raw_good':int(raw>.5),
                    'accepted_raw_good':int(accepted>.5),'typed_accepted_raw_good':int(typed_raw>.5),
                    'canonical_bg':int(label==0),'canonical_kept':int(is_kept),
                    'canonical_correct_type':int(label==cls),
                    'raw_good_rejected_bg':int(assigned>.5 and label==0),
                    'raw_good_rejected_lowconf':int(assigned>.5 and label>0 and not is_kept),
                    'raw_good_kept_wrong_type':int(assigned>.5 and is_kept and label!=cls),
                    'final_good':int(best>.5),'final_typed_good':int(best>.5 and pc==cls),'final_miss':int(best<.1)}
            for group in ([str(cls),'Dead_interior' if interior[oid] else 'Dead_border'] if cls==4 else [str(cls)]):
                for name in names:groups[group][name]+=values[name]
            if cls==4:rows.append({'image':image,'gt_id':oid,'interior':bool(interior[oid]),
                                  'assigned_raw_iou':assigned,'best_raw_iou':raw,'best_accepted_iou':accepted,
                                  'best_typed_accepted_iou':typed_raw,'canonical_class':label,'canonical_kept':is_kept,
                                  'final_max_iou':best,'final_class_if_matched':pc})
        del cache,pred,batch
        if time.monotonic()-started>1100:raise RuntimeError('Audit wall budget')
    assert groups['4']['n']==967 and len(rows)==967 and int(confusion[3].sum())==967
    assert groups['Dead_interior']['n']+groups['Dead_border']['n']==967
    result={'groups':groups,'canonical_confusion_rows_gt1to5_cols_pred0to5':confusion.tolist(),
            'n_pred':int(n_pred),'dead_rows':rows}
    with (a.out/f'seed{seed}.json').open('x') as f:json.dump(result,f,indent=2)
    results[str(seed)]=result;manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth']
    print(json.dumps({'seed':seed,'Dead':groups['4'],'interior':groups['Dead_interior']}),flush=True)
    del model;torch.cuda.empty_cache()
with (a.out/'results.json').open('x') as f:
    json.dump({'manifest':manifest,'results':results,'elapsed_seconds':time.monotonic()-started},f,indent=2)
print('AUDIT_COMPLETE',flush=True)
