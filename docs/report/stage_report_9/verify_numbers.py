import csv,json,pathlib
root=pathlib.Path(__file__).parent
if not (root/'assets').exists(): root=root/'report9'
a=root/'assets'
p=json.loads((a/'P36_RESULTS.json').read_text()); d=json.loads((a/'D37_RESULTS.json').read_text())
assert len(p['detectors'])==len(d['detectors'])==8
checks=0
for model,e in p['detectors'].items():
    for role in ('training','development'):
        x=e[role]; q=d['detectors'][model][role]
        assert q['images']==(240 if role=='training' else 522)
        for fill in ('white','mirror'):
            assert sum(q['partition'][fill]['all'].values())==q['nuclei']==x['nuclei'];checks+=1
        for name,v in x['decoders'].items():
            assert abs(v['recall']-v['matched']/v['total'])<1e-12
            assert abs(v['precision']-v['matched']/v['candidates'])<1e-12
            assert abs(v['f1']-2*v['matched']/(v['total']+v['candidates']))<1e-12;checks+=3
        t=q['selected_threshold_transitions']; dec=x['decoders']
        assert t['gained']-t['lost']==dec['selected_threshold']['matched']-dec['original']['matched'];checks+=1
r=list(csv.DictReader((a/'CASE_RESPONSES.csv').open()))
case=next(x for x in r if x['model']=='P15_J_s19' and x['image']=='1482' and x['gt_index']=='7')
assert abs(float(case['original_max_score_3px'])-.4859921634197235)<1e-12
assert abs(float(case['consensus_distance'])-1.900292375165233)<1e-12
assert case['matched_original']=='False' and case['matched_consensus']=='True';checks+=3
for name in ('stage_report_9_en.md','stage_report_9_zh.md','summary_plain_zh.md'):
    text=(root/name).read_text(encoding='utf-8')
    assert '0.486' in text and '1.90' in text
    assert not any(term in text for term in ('肺图','拟合','审计','局部响应','不代表','测试表现不好'))
result={'status':'PASS','checks':checks,'detectors':8,'roles':['training','development'],'case_score':float(case['original_max_score_3px']),'case_distance':float(case['consensus_distance']),'language_scan':'PASS','review':'self-reviewed'}
(root/'NUMBER_CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
