"""Behavioral regression tests for transient Lustre quota-query failures."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_lustre_quota.py'
HEADER = 'Disk quotas for usr example (uid 1000):\n Filesystem kbytes quota limit grace files quota limit grace\n/test\n'
VALID = HEADER + ' 100 0 1048576 - 10 0 2000 -\n'
BROKEN = HEADER + ' [100] 0 1048576 - 10 0 2000 -\nSome errors happened; data in [] is inaccurate.\n'


def run_guard(tmp_path, replies, minimum=1):
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'replies': replies, 'calls': 0}))
    fake = tmp_path / 'lfs'
    fake.write_text('#!' + sys.executable + '\n' + '''import json, os, pathlib, sys
p=pathlib.Path(os.environ['QUOTA_TEST_STATE']);s=json.loads(p.read_text())
r=s['replies'][min(s['calls'],len(s['replies'])-1)];s['calls']+=1;p.write_text(json.dumps(s))
print(r[1],end='');sys.exit(r[0])
''')
    fake.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'], QUOTA_TEST_STATE=str(state))
    result = subprocess.run([sys.executable, str(SCRIPT), '--filesystem', str(tmp_path),
        '--scope', 'user', '--identity', 'example', '--min-free-gib', str(minimum / 1024),
        '--min-free-files', '1000', '--attempts', '3', '--delay-seconds', '0',
        '--timeout-seconds', '2'], env=env, text=True, capture_output=True)
    return result, json.loads(state.read_text())['calls']


def test_transient_exit5_then_valid_query_recovers(tmp_path):
    r, calls = run_guard(tmp_path, [[5, BROKEN], [0, VALID]])
    assert r.returncode == 0, r.stderr
    assert calls == 2
    assert json.loads(r.stdout.splitlines()[-1])['status'] == 'PASS'


def test_inaccurate_zero_exit_is_never_accepted(tmp_path):
    r, calls = run_guard(tmp_path, [[0, BROKEN]])
    assert r.returncode != 0
    assert calls == 3
    assert 'PASS' not in r.stdout


def test_exhaustion_stops_after_three_queries(tmp_path):
    r, calls = run_guard(tmp_path, [[5, BROKEN]])
    assert r.returncode != 0
    assert calls == 3


@pytest.mark.parametrize('row', [
    '1048500 0 1048576 - 10 0 2000 -',
    '100 0 1048576 - 1500 0 2000 -',
    '100 110 1048576 - 10 0 2000 -',
])
def test_insufficient_block_inode_or_soft_limit_fails_without_retry(tmp_path, row):
    r, calls = run_guard(tmp_path, [[0, HEADER + row + '\n']])
    assert r.returncode != 0
    assert calls == 1


def test_missing_or_malformed_data_cannot_pass(tmp_path):
    r, calls = run_guard(tmp_path, [[0, 'unexpected quota response\n']])
    assert r.returncode != 0
    assert calls == 3


def test_nonzero_exit_with_plausible_numbers_does_not_pass(tmp_path):
    r, calls = run_guard(tmp_path, [[5, VALID]])
    assert r.returncode != 0
    assert calls == 3


def test_valid_response_needs_only_one_query(tmp_path):
    r, calls = run_guard(tmp_path, [[0, VALID]])
    assert r.returncode == 0, r.stderr
    assert calls == 1


def test_unlimited_inode_quota_is_allowed(tmp_path):
    r, calls = run_guard(tmp_path, [[0, HEADER + '100 0 1048576 - 10 0 0 -\n']])
    assert r.returncode == 0, r.stderr
    assert calls == 1
