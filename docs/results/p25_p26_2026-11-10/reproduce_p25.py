"""Recheck published raw last-checkpoint P25 results and original gate. Run from repository root."""
import argparse,pathlib,sys,json,hashlib
root=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'scripts'))
from proposal_p3_utils import admission
from analyze_proposal_p18 import aggregate,bootstrap_and_gate
from train_proposal_p25 import run_config
from p5_contracts import validate_cohort,COHORT_SHA

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=pathlib.Path,default=pathlib.Path(__file__).parent);a=ap.parse_args();d=a.directory
    expected=json.loads((d/'analysis.json').read_text());hashes=json.loads((d/'artifact_hashes.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h,name
    raw={};summary={}
    for seed in (19,1):
        for arm in ('C','D'):
            tag=f'A_{arm}_s{seed}';run=d/'raw'/tag;load=lambda name:json.loads((run/name).read_text())
            manifest=load('portable_manifest.json');assert manifest['config']==run_config(arm,seed)
            assert validate_cohort(manifest['fit'],manifest['audit'])==COHORT_SHA
            rows=load('evaluation.json');fit=load('fit.json');done=load('complete.json')
            assert len(rows)==522 and len(fit)==2134 and done['epochs']==130 and done['exit_code']==0
            assert [r['image'] for r in rows]==manifest['audit'] and [r['image'] for r in fit]==manifest['fit']
            logs=[json.loads(s) for s in (run/'loss.jsonl').read_text().splitlines()];assert [r['epoch'] for r in logs]==list(range(1,131))
            assert all(r['optimizer_steps']==134 and r['effective_batch']==16 and r['microbatch']==4 for r in logs)
            for scope in ('audit','fit'):
                ex=load(f'point_examples_{scope}/coordinates.json');assert [r['image'] for r in ex['records']]==manifest['fixed_examples'][scope]
            key=f'{arm}_{seed}';raw[key]=rows;summary[key]=dict(audit=aggregate(rows),fit=aggregate(fit))
            assert summary[key]['audit']==expected['runs'][key]['audit'] and summary[key]['fit']==expected['runs'][key]['fit']
    gate=admission({str(s):dict(F1=raw[f'C_{s}'],T1=raw[f'D_{s}']) for s in (19,1)})
    assert gate==expected['original_gate']
    for s in (19,1):
        pair=bootstrap_and_gate(raw[f'C_{s}'],raw[f'D_{s}'])
        for k in ('delta','ci95','pass_seed'):assert pair[k]==gate['details'][str(s)][k]
    print(json.dumps(dict(status='PASS',files=len(hashes),original_gate=gate,source_manifest_sha256='b419ac061fffc7d36f0b87614d4ec3f7cb1ad93994a35947bd0a9f6e71153354'),indent=2))

if __name__=='__main__':main()
