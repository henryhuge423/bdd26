"""Check P31 published values against the retained source records and manifests."""
from pathlib import Path
import hashlib, json, math, csv
p=Path(__file__).resolve().parent
expected={(19,'C'):(7274,10536,65),(19,'G'):(7134,10089,73),(1,'C'):(7256,10437,68),(1,'G'):(7262,10354,70)}
for (seed,arm),(matches,predictions,dead) in expected.items():
 d=p/'evidence'/'p31_files'/f'A_{arm}_s{seed}'
 m=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'))
 for f in d.rglob('*'):
  if f.is_file() and f.name!='preservation_manifest.json':
   k=f.relative_to(d).as_posix(); b=f.read_bytes(); assert len(b)==m[k]['bytes'] and hashlib.sha256(b).hexdigest()==m[k]['sha256'],k
 done=json.loads((d/'complete.json').read_text()); assert done['epochs']==130 and done['exit_code']==0
 rows=json.loads((d/'evaluation.json').read_text()); assert len(rows)==522
 assert sum(r['matched'] for r in rows)==matches
 assert sum(r['candidates'] for r in rows)==predictions
 assert sum(r['dead_hit'] for r in rows)==dead
 print(seed,arm,matches,predictions,dead,'OK')
for seed in (19,1):
 records=[]
 for arm in ('C','G'):
  d=p/'evidence'/'p31_files'/f'A_{arm}_s{seed}'/'point_examples_audit'
  records.append({r['image']:r for r in json.loads((d/'coordinates.json').read_text())['records']})
 for im in (776,1025,1037):
  l,r=[x[im] for x in records]; assert l['gt']==r['gt']
  for q in (l,r):
   pairs=dict(q['unique_pairs']); assert len(pairs)==q['score']['matched']
   assert all(math.dist(q['gt'][g][1],q['points'][j])<=3 for g,j in pairs.items())
print('Fixed object matches and selected inputs verified.')
comparisons={27:('p27_analysis.json','E'),28:('p28_analysis.json','I'),29:('p29_verified_analysis.json','B'),30:('p30_verified_analysis.json','M')}
rows={v['pair']:v for v in csv.DictReader((p/'evidence'/'stage_pairs.csv').open(encoding='utf-8'))}
for number,(filename,arm) in comparisons.items():
 runs=json.loads((p/'evidence'/filename).read_text(encoding='utf-8'))['runs']
 for seed in (19,1):
  c=runs[f'C_{seed}']['audit'];t=runs[f'{arm}_{seed}']['audit'];row=rows[f'P{number}_{seed}']
  assert int(row['net_dead'])==t['dead_hit']-c['dead_hit']
  assert int(row['net_all'])==t['matched']-c['matched']
  assert int(row['net_extra'])==(t['candidates']-t['matched'])-(c['candidates']-c['matched'])
for seed in (19,1):
 c=expected[(seed,'C')];t=expected[(seed,'G')];row=rows[f'P31_{seed}']
 assert int(row['net_dead'])==t[2]-c[2] and int(row['net_all'])==t[0]-c[0]
 assert int(row['net_extra'])==(t[1]-t[0])-(c[1]-c[0])
print('All ten P27–P31 paired chart rows verified.')
