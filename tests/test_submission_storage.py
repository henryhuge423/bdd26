import json
import subprocess
from pathlib import Path
import pytest


def test_all_three_scopes_are_checked_by_sealed_helper():
    from submission_storage import admit_storage
    calls=[]
    def run(cmd,**kwargs):
        calls.append((cmd,kwargs));return subprocess.CompletedProcess(cmd,0,json.dumps({'status':'PASS'})+'\n','')
    records=admit_storage('/p/python','/sealed/check_lustre_quota.py',run=run)
    assert set(records)=={'scratch_user','project_user','project_group'}
    assert [c[0][c[0].index('--scope')+1] for c in calls]==['user','user','group']
    assert [c[0][c[0].index('--identity')+1] for c in calls]==['xjin10','xjin10','jhu159']
    assert all(c[0][:2]==['/p/python','/sealed/check_lustre_quota.py'] and '--min-free-files' in c[0] and c[1]['timeout']==600 for c in calls)


@pytest.mark.parametrize('code,text',[(1,'{"status":"FAIL","quota_free_files":0}\n'),(1,'{"status":"FAIL","quota_free_bytes":0}\n'),(0,'{"status":"QUERY_FAILED"}\n'),(0,'not json\n')])
def test_failed_or_ambiguous_admission_prevents_submission(code,text):
    from submission_storage import admit_storage
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd);return subprocess.CompletedProcess(cmd,code,text,'')
    with pytest.raises((RuntimeError,ValueError)):
        admit_storage('/p/python','/sealed/check_lustre_quota.py',run=run)
    assert len(calls)==1
