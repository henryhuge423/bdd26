import hashlib
import pytest
from submission_readback import readback, verify_source

def test_readback_creates_fsynced_verified_evidence(tmp_path):
    r=readback(tmp_path,'probe')
    assert r['bytes']==65536 and hashlib.sha256((tmp_path/'probe').read_bytes()).hexdigest()==r['sha256']
    with pytest.raises(FileExistsError):readback(tmp_path,'probe')

def test_fsync_failure_stops_admission(tmp_path,monkeypatch):
    def fail(_):raise OSError('fsync failed')
    monkeypatch.setattr('os.fsync',fail)
    with pytest.raises(OSError,match='fsync failed'):readback(tmp_path,'probe')

def test_readback_mismatch_stops_admission(tmp_path,monkeypatch):
    monkeypatch.setattr(type(tmp_path),'read_bytes',lambda _:b'corrupt')
    with pytest.raises(ValueError,match='readback'):readback(tmp_path,'probe')

def test_source_identity_rejects_changed_missing_and_added_files(tmp_path):
    (tmp_path/'a.py').write_bytes(b'original')
    expected={'a.py':hashlib.sha256(b'original').hexdigest()}
    verify_source(tmp_path,expected,['a.py'])
    (tmp_path/'a.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source'):verify_source(tmp_path,expected,['a.py'])
    with pytest.raises(ValueError,match='inventory'):verify_source(tmp_path,expected,['a.py','new.py'])
    with pytest.raises(ValueError,match='inventory'):verify_source(tmp_path,expected,[])

def test_source_identity_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError,match='path'):verify_source(tmp_path,{'../a.py':'x'},['../a.py'])
