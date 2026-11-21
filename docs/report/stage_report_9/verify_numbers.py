"""Validate revised report evidence through cross-artifact checks."""
from pathlib import Path
import json,math,csv,re
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'assets').exists() else root/'outputs/report9_revision_20261121'
data={n:json.loads((out/'assets'/(n+'.json')).read_text()) for n in ['D32_ANALYSIS','D33_ANALYSIS','P34_RESULTS','P35_RESULTS','P36_RESULTS','D37_RESULTS','P38_RESULTS']}
checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
d=data['D32_ANALYSIS']
check('D32 repeated objects',sum(x['objects'] for x in d['models'])==1740)
check('D32 original misses',sum(x['missed'] for x in d['models'])==875)
check('D32 interpolation flags',sum(x['coarse_positive_fine_negative'] for x in d['models'])==334)
check('D32 in-mask displaced misses',sum(x['displaced_in_mask'] for x in d['models'])==183)
check('D32 shift crossings',sum(x['shift_threshold_crossings'] for x in d['models'])==744)
check('D32 shift observations',sum(x['shift_observations'] for x in d['models'])==6048)
for x in data['D33_ANALYSIS']['models']:
 check(x['model']+' D33 net match',x['alternative']['matched']-x['original']['matched']==x['net_matched'])
 check(x['model']+' D33 net extras',(x['alternative']['candidates']-x['alternative']['matched'])-(x['original']['candidates']-x['original']['matched'])==x['net_extra'])
 check(x['model']+' D33 net Dead',x['alternative']['dead_hit']-x['original']['dead_hit']==x['dead_gain']-x['dead_loss'])
check('D33 repeated extra observations',sum(x['net_extra'] for x in data['D33_ANALYSIS']['models'])==813)
for m,x in data['P34_RESULTS']['detectors'].items():
 for role,r in x['roles'].items():
  for method in ['same_augmentation','with_translation']:
   check(m+' '+role+' '+method+' transition',r[method]['merged']['matched']-r['original']['matched']==r[method]['transitions']['gained']-r[method]['transitions']['lost'])
for m,x in data['P36_RESULTS']['detectors'].items():
 y=data['P38_RESULTS']['detectors'][m]
 for role in ['training','development']:
  for a,b in [('original','original'),('selected_threshold','p36_threshold'),('consensus','p36_consensus')]:
   cc=x[role]['decoders'][a];dd=y['roles'][role]['decoders'][b]['counts']
   for k in ['matched','total','candidates','dead_hit','dead_total']:check(m+' '+role+' '+a+' '+k,cc[k]==dd[k])
  r=data['D37_RESULTS']['detectors'][m][role]
  for fill,subsets in r['partition'].items():
   for subset,counts in subsets.items():
    den=x[role]['nuclei'] if subset=='all' else x[role]['central'] if subset=='central' else x[role]['interior_dead']
    check(m+' '+role+' '+fill+' '+subset+' partition',sum(counts.values())==den)
per_image_path=out/'assets/PER_IMAGE_COUNTS.json'
if per_image_path.exists():
 per_image=json.loads(per_image_path.read_text())
 for m,roles in per_image.items():
  for role,decoders in roles.items():
   for decoder,rows in decoders.items():
    counts=data['P38_RESULTS']['detectors'][m]['roles'][role]['decoders'][decoder]['counts']
    check(m+' '+role+' '+decoder+' image coverage',len(rows)==(240 if role=='training' else 522) and len({r['image'] for r in rows})==len(rows))
    for k in ['matched','total','candidates','dead_hit','dead_total']:
     check(m+' '+role+' '+decoder+' per-image '+k,sum(r[k] for r in rows)==counts[k])
case=json.loads((out/'assets/CASE_RESPONSES.json').read_text())
expect=[('P15_J_s19',1482,141,65,.91579,1.89,True),('P15_J_s19',1482,81,44,.30811,1.69,False),('P22_M_s19',1037,222,125,.19840,.25,False),('P22_M_s19',1482,73,19,.39413,None,True)]
for m,im,y,x,q,dist,keep in expect:
 row=next(v for v in case if v['model']==m and v['image']==im and v['center_y']==y and v['center_x']==x)
 check(f'{m} image{im} center{y},{x} quality',abs(row['nearest_group_score']-q)<.0000051)
 if dist is not None:check(f'{m} image{im} center{y},{x} distance',abs(row['nearest_group_distance_px']-dist)<.0051)
 check(f'{m} image{im} center{y},{x} selection',row['calibrated']==keep)
for m,deltaextra,deltamatch in [('P15_J_s19',-676,-166),('P15_J_s1',-377,4),('P22_M_s19',984,489),('P22_M_s1',175,117)]:
 ds=data['P38_RESULTS']['detectors'][m]['roles']['development']['decoders'];a=ds['support_only']['counts'];b=ds['calibrated']['counts']
 check(m+' P38 match difference',b['matched']-a['matched']==deltamatch)
 check(m+' P38 extra difference',b['candidates']-b['matched']-a['candidates']+a['matched']==deltaextra)
baseline = json.loads((out/'assets/P34_SEED1_BASELINE.json').read_text())
check('P34 all-layer seed-1 raw preflight identity', baseline['zero_offset_refined_equals_original_per_image_for_both_readouts'] is True)
for role, expected in [('development', 522), ('training', 48), ('training_shifted', 48)]:
 check('P34 all-layer seed-1 '+role+' preflight images',baseline['roles'][role]['images']==expected)
p36_original = data['P36_RESULTS']['detectors']['P22_M_s1']['development']['decoders']['original']
for key, value in baseline['roles']['development']['counts'].items():
 check('P34 all-layer seed-1 original agrees with P36 '+key,value==p36_original[key])
record={'status':'PASS','checks':len(checks),'check_names':checks,'scope':'Cross-artifact P34/P36/P38 original and replay counts, exclusive partitions, transitions and fixed-case geometry; no new model inference or experiments.'}
(out/'NUMBER_CHECKS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('PASS:',len(checks),'cross-artifact numerical and case checks')
