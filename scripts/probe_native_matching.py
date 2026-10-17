"""E14: exact native-resolution Hungarian matching on stable mask basis."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json,math,os
from pathlib import Path
import shutil,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

BASE_SHA='f002647604a44cd95064dfa805ec389c9835ad2decb619b68a86ec3c690ef2f8'
MEASURE_SHA='8432b771d2fe1b69b1633d748659a668ddad7095146f597ddd5a0e06d3df51f1'
CONTROL_SHA='0e6d668eb0463784d3200c87bdeae2d945373777f39153e1007984bb3e3d0648'
CONTROL_DESIGN_SHA='6fabdfb7563e3670e23739bab6e3039b79c99ea781d1717586b40293d5c10bd9'


def staged_forward(base,head,features):
    """Frozen E10 tensor route plus three auxiliary3D predictions; no state changes."""
    from nucseg.setmask.model import spatial_position
    if len(head.layers)!=3:raise ValueError('E11 requires initial plus three query stages')
    memory=head.memory_projection(F.avg_pool2d(features,8))
    mpos=spatial_position(*memory.shape[-2:],head.dim,memory.device,memory.dtype)
    mask_features=head.mask_projection(features)
    with torch.no_grad():
        low=F.interpolate(mask_features.float(),size=memory.shape[-2:],mode='bilinear',align_corners=False)
    memory=memory.flatten(2).transpose(1,2)
    q=head.query_content.weight[None].expand(features.shape[0],-1,-1)
    qpos=head.query_position.weight[None]
    def predict(query):
        return {'class_logits':head.class_head(query),'mask_embed':head.mask_head(query),'mask_features':mask_features}
    stages=[predict(q)]
    for layer in head.layers:
        with torch.no_grad(),torch.autocast(features.device.type,enabled=False):
            embed=head.mask_head(q.float())
            logits=torch.einsum('bqc,bck->bqk',embed,low.flatten(2))/math.sqrt(embed.shape[-1])
            blocked,_=base.attention_mask(logits,layer.cross.num_heads)
        t=layer.norm1(q)
        q=q+layer.drop(layer.cross(t+qpos,memory+mpos,memory,attn_mask=blocked,need_weights=False)[0])
        t=layer.norm2(q)
        q=q+layer.drop(layer.self_attention(t+qpos,t+qpos,t,need_weights=False)[0])
        q=q+layer.drop(layer.ff(layer.norm3(q)))
        stages.append(predict(q))
    return stages


def auxiliary_loss(stages,batch,tissue_logits):
    if len(stages)!=4:raise ValueError('Exactly four stages required; no silent stage reweighting')
    from nucseg.setmask.matching import setmask_loss
    losses=[];rows=[]
    for index,pred in enumerate(stages):
        loss,terms=setmask_loss({**pred,'tissue':tissue_logits},batch)
        losses.append(loss);rows.append({'stage':index,'loss':float(loss.detach()),'terms':terms})
    loss=torch.stack(losses).mean()
    return loss,{'stages':rows,'stage_weight':.25,'loss':float(loss.detach())}


def allocation(base,rows):
    mapped={s:{'A0':r['A0'],'G':r['P'],'M':r['R']} for s,r in rows.items()}
    result=base.allocation(mapped)
    for detail in result['per_seed'].values():
        detail['r_minus_p']=detail.pop('m_minus_g');detail['r_minus_a0']=detail.pop('m_minus_a0')
    return result


@torch.no_grad()
def native_cost(classes,embed,features,target):
    truth=(target['inst'][None]==target['ids'][:,None,None]).to(features.device).flatten(1).float()
    foreground=target['classes'].to(features.device)
    native=features.float().flatten(1);pixels=native.shape[1]
    pieces=[]
    with torch.autocast(features.device.type,enabled=False):
        for left in range(0,len(embed),16):
            right=min(len(embed),left+16)
            logits=embed[left:right].float()@native/math.sqrt(embed.shape[-1])
            bce=F.softplus(logits).mean(1)[:,None]-logits@truth.T/pixels
            p=logits.sigmoid()
            dice=1-(2*(p@truth.T)+1)/(p.sum(1)[:,None]+truth.sum(1)[None,:]+1)
            pieces.append(-2*classes[left:right].float().softmax(-1)[:,foreground]+5*bce+5*dice)
    cost=torch.cat(pieces,dim=0)
    if not bool(torch.isfinite(cost).all()):raise ValueError('Nonfinite native matching cost')
    return cost

@torch.no_grad()
def native_match_queries(class_logits,mask_embed,mask_features,targets):
    result=[]
    for b,target in enumerate(targets):
        count=len(target['ids'])
        if count>class_logits.shape[1]:raise ValueError('Ground truth exceeds query capacity')
        if count==0:
            result.append((torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)));continue
        cost=native_cost(class_logits[b],mask_embed[b],mask_features[b],target)
        qi,ti=linear_sum_assignment(cost.cpu().numpy())
        result.append((torch.from_numpy(qi).long(),torch.from_numpy(ti).long()))
    return result

def trainable_head(head):
    head.mask_projection.requires_grad_(False)
    return [p for p in head.parameters() if p.requires_grad]


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@torch.no_grad()
def check_equivalence(base,head,row):
    feature=row['feature'].to('cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=base.query_forward(head,feature,True)
        stages=staged_forward(base,head,feature)
    assert all(torch.equal(expected[k],stages[-1][k]) for k in expected),'Auxiliary calls changed E10 final bf16 prediction'
    return {'image':row['image'],'all_final_tensors_bit_exact':True}


def run(a):
    snapshot=a.snapshot;os.chdir(snapshot);sys.path.insert(0,str(snapshot/'src'))
    from torch.utils.data import default_collate
    from nucseg.setmask.engine import load_config,build_model,sha_file
    from nucseg.setmask.data import PairedDataset
    import nucseg.setmask.matching as matching
    matching.match_queries=native_match_queries
    assert sha_file(a.base)==BASE_SHA and sha_file(a.measure)==MEASURE_SHA
    assert sha_file(a.control)==CONTROL_SHA and sha_file(a.control_design)==CONTROL_DESIGN_SHA
    base=load_module(a.base,'e10_frozen_forward');helper=load_module(a.measure,'e9_frozen_measurement')
    helper.query_forward=base.query_forward
    reference=json.loads(a.control.read_text())
    assert sha_file(a.aux_control)=='8e9c81ba393f7fcec19a14692191edb4382a666eaf870cec6a24a502ea0d55ed'
    auxiliary=json.loads(a.aux_control.read_text())
    assert sha_file(a.stable_control)=='63da90ec0de54ed04fc548d2365d981ada663204a4d1de33484a56f47e7baee8'
    stable=json.loads(a.stable_control.read_text())
    assert not stable['allocation']['promising']
    assert stable['manifest']['groups']==auxiliary['manifest']['groups']
    assert stable['manifest']['checkpoints']==auxiliary['manifest']['checkpoints']
    for seed in ('19','1'):
        for arm in ('A0','G','M','N'):assert stable['results'][seed][arm]==auxiliary['results'][seed][arm]
    assert auxiliary['manifest']['groups']==reference['manifest']['groups']
    assert auxiliary['manifest']['checkpoints']==reference['manifest']['checkpoints']
    for seed in ('19','1'):
        for arm in ('A0','G','M'):assert auxiliary['results'][seed][arm]==reference['results'][seed][arm]
    assert reference['manifest']['script_sha256']==BASE_SHA and reference['manifest']['design_sha256']==CONTROL_DESIGN_SHA
    assert reference['manifest']['steps']==300 and reference['manifest']['batch_size']==2
    assert torch.cuda.device_count()==1 and shutil.disk_usage(a.out.parent).free>20*1024**3
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    a.out.mkdir(exist_ok=False);start=time.monotonic()
    source=json.loads((snapshot/'MANIFEST.json').read_text())
    assert source['commit']==reference['manifest']['source_commit']
    for group in ('sources','vendor'):
        for name,digest in source[group].items():assert sha_file(snapshot/name)==digest,name
    ds=PairedDataset(1,False,19);present=ds.class_presence()[:,3]
    groups=helper.cohorts(np.flatnonzero(present).tolist(),np.flatnonzero(~present).tolist())
    assert groups==reference['manifest']['groups'] and len(groups['fit'])==64 and len(groups['audit'])==66
    for arms in reference['results'].values():
        for arm in ('A0','G','M'):
            for group,images in groups.items():base.validate_measurement(arms[arm][group],images)
        assert [x['step'] for x in arms['M']['optimization']]==[1,100,300]
    manifest={'fold':1,'source_commit':source['commit'],'groups':groups,'steps':300,'batch_size':2,
              'script_sha256':sha_file(__file__),'design_sha256':sha_file(a.design),'base_sha256':BASE_SHA,
              'measure_sha256':MEASURE_SHA,'control_sha256':CONTROL_SHA,'control_design_sha256':CONTROL_DESIGN_SHA,
              'checkpoints':{},'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'stage_weights':[.25]*4,
              'control_reuse':'E13 P, exact A0 reconstruction before training; original A0/G/M/N retained',
              'aux_control_sha256':sha_file(a.aux_control),'stable_control_sha256':sha_file(a.stable_control),'frozen_projection':True,'matching':'exact full native pixels',
              'limitations':'Train-only cohort previously seen in E7; no validation/test claim; no saved weights'}
    base.save(a.out/'manifest.json',manifest)
    results={};gate_rows={}
    for seed in (19,1):
        run_dir=a.run_root/f'Q_seed{seed}';cfg=load_config(run_dir)
        assert cfg.round=='E7' and cfg.arm=='Q' and cfg.seed==seed
        receipt=json.loads((run_dir/'completion.json').read_text())
        for name in ('config.json','provenance.json','log.jsonl','final.pth'):
            assert sha_file(run_dir/name)==receipt['hashes'][name],name
        assert receipt['hashes']['final.pth']==reference['manifest']['checkpoints'][str(seed)]
        state=torch.load(run_dir/'final.pth',map_location='cpu',weights_only=False,mmap=True)
        assert state['config']==asdict(cfg)
        prov=state['provenance'];assert torch.__version__==prov['torch'] and np.__version__==prov['numpy']
        for name in ('images','inst','type','tissue'):
            assert sha_file(ds.fold.dir/(name+'.npy'))==prov['inputs']['fold1/'+name+'.npy']
        assert sha_file(cfg.uni_ckpt)==prov['inputs']['UNI']
        model=build_model(cfg,pretrained=False).eval().to('cuda');model.load_state_dict(state['model']);del state
        caches={k:[] for k in groups};captured=[]
        hook=model.query_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        with torch.no_grad():
            for group,images in groups.items():
                for image in images:
                    batch=default_collate([ds[image]])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(batch['img'].to('cuda'));feature=captured.pop()
                        original=base.query_forward(model.query_head,feature,False);captured.clear()
                    assert all(torch.equal(pred[k],original[k]) for k in original)
                    caches[group].append({'image':image,'feature':feature.cpu(),'batch':batch,'tissue_logits':pred['tissue'].cpu()})
                    del pred,feature,original
                    helper.deadline(start)
        hook.remove();original_head=deepcopy(model.query_head).eval();del model;torch.cuda.empty_cache()
        a0={k:helper.measure(original_head,v,False,start) for k,v in caches.items()}
        assert a0==reference['results'][str(seed)]['A0'],'A0 differs; do not reuse controls'
        before=helper.feature_hash(caches['fit']+caches['audit'])
        torch.manual_seed(seed);head=deepcopy(original_head).eval()
        assert all(torch.equal(v,original_head.state_dict()[k]) for k,v in head.state_dict().items())
        equivalence={'before':{k:check_equivalence(base,head,v[0]) for k,v in caches.items()}}
        diagnostics={'before':{k:base.diagnose(head,v[0]) for k,v in caches.items()}}
        print(json.dumps({'seed':seed,'a0_exact_reproduction':True,'bf16_final_equivalence':equivalence['before']}),flush=True)
        projection_before={k:v.clone() for k,v in head.mask_projection.state_dict().items()}
        optimizer=torch.optim.AdamW(trainable_head(head),lr=3e-4,betas=(.85,.95),weight_decay=1e-4)
        logs=[]
        for step in range(300):
            rows=[caches['fit'][(step*2+j)%64] for j in range(2)]
            feature=torch.cat([r['feature'] for r in rows]).to('cuda')
            batch={k:torch.cat([r['batch'][k] for r in rows]) for k in ('inst_map','tp_map','tissue')}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):stages=staged_forward(base,head,feature)
            tissue=torch.cat([r['tissue_logits'] for r in rows]).to('cuda')
            loss,record=auxiliary_loss(stages,batch,tissue);assert bool(torch.isfinite(loss))
            loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.requires_grad)
            optimizer.step()
            if step in (0,99,299):logs.append({'step':step+1,**record})
            del stages,feature,loss,tissue
            helper.deadline(start)
        assert all(torch.equal(v,projection_before[k]) for k,v in head.mask_projection.state_dict().items())
        equivalence['after']={k:check_equivalence(base,head,v[0]) for k,v in caches.items()}
        treatment={k:helper.measure(head,v,True,start) for k,v in caches.items()}
        diagnostics['after']={k:base.diagnose(head,v[0]) for k,v in caches.items()}
        treatment.update(optimization=logs,bf16_equivalence=equivalence,attention_diagnostics=diagnostics,frozen_projection_exact=True)
        assert helper.feature_hash(caches['fit']+caches['audit'])==before
        for group,images in groups.items():base.validate_measurement(treatment[group],images)
        arms={'A0':a0,'G':reference['results'][str(seed)]['G'],'M':reference['results'][str(seed)]['M'],'N':auxiliary['results'][str(seed)]['N'],'P':stable['results'][str(seed)]['P'],'R':treatment}
        results[str(seed)]=arms
        gate_rows[str(seed)]={arm:{**r['audit'],'fit_recall':r['fit']['recall']} for arm,r in arms.items() if arm!='G'}
        manifest['checkpoints'][str(seed)]=receipt['hashes']['final.pth']
        base.save(a.out/f'seed{seed}.json',arms)
        print(json.dumps({'seed':seed,'R':{g:{k:v for k,v in treatment[g].items() if k!='per_image'} for g in ('fit','audit')}}),flush=True)
        del optimizer,head,original_head,caches;torch.cuda.empty_cache()
    payload={'manifest':manifest,'results':results,'allocation':allocation(base,gate_rows),
             'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    base.save(a.out/'results.json',payload);print(json.dumps(payload['allocation']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','base','measure','control','control-design','aux-control','stable-control','design','run-root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    run(a)
