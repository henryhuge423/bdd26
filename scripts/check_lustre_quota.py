#!/usr/bin/env python3
"""Fail-closed Lustre quota check with bounded retries for query failures.

Uses machine-readable (non -h) lfs output. Invalid/incomplete query data may be
retried; verified insufficient quota is a terminal failure. No files are removed
or written on the queried filesystem. Diagnostics are JSON lines on stdout.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time


def parse_quota(text):
    """Return byte/inode headroom, respecting the smaller nonzero soft/hard limit."""
    if '[' in text or ']' in text or 'inaccurate' in text.lower():
        raise ValueError('quota query contains inaccurate values')
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith('/'):
            fields = fields[1:]
        if len(fields) == 8 and all(fields[i].rstrip('*').isdigit() for i in (0, 1, 2, 4, 5, 6)):
            rows.append(fields)
    if len(rows) != 1:
        raise ValueError('expected one complete quota totals row')
    row = rows[0]
    available = []
    for offset, scale in ((0, 1024), (4, 1)):
        used, soft, hard = (int(row[offset + i].rstrip('*')) for i in range(3))
        limits = [v for v in (soft, hard) if v > 0]
        available.append((min(limits) - used) * scale if limits else None)
    return dict(quota_free_bytes=available[0], quota_free_files=available[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filesystem', required=True)
    parser.add_argument('--scope', choices=('user', 'group'), required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--min-free-gib', type=float, required=True)
    parser.add_argument('--min-free-files', type=int, default=1000)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--delay-seconds', type=float, default=10)
    parser.add_argument('--timeout-seconds', type=float, default=45)
    args = parser.parse_args(argv)
    if (not 1 <= args.attempts <= 3 or not 0 <= args.delay_seconds <= 30 or
            not 0 < args.timeout_seconds <= 60 or not math.isfinite(args.min_free_gib) or
            args.min_free_gib < 0 or args.min_free_files < 0):
        parser.error('invalid thresholds or retry bounds')
    required = math.ceil(args.min_free_gib * 1024**3)
    command = ['lfs', 'quota', '-u' if args.scope == 'user' else '-g', args.identity, args.filesystem]
    for attempt in range(1, args.attempts + 1):
        record = dict(attempt=attempt, command=command, status='QUERY_FAILED')
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=args.timeout_seconds)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode != 0:
                raise ValueError('lfs query returned nonzero status')
            headroom = parse_quota(result.stdout + '\n' + result.stderr)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            record['error'] = str(error)
            print(json.dumps(record), flush=True)
            if attempt < args.attempts:
                time.sleep(args.delay_seconds)
                continue
            print('quota query exhausted; storage admission denied', file=sys.stderr)
            return 1
        # Capacity failures are terminal, not transient-query retry candidates.
        try:
            filesystem_free = shutil.disk_usage(args.filesystem).free
        except OSError as error:
            record.update(status='FAIL', error=str(error))
            print(json.dumps(record), flush=True)
            return 1
        passed = (filesystem_free >= required and
                  (headroom['quota_free_bytes'] is None or headroom['quota_free_bytes'] >= required) and
                  (headroom['quota_free_files'] is None or headroom['quota_free_files'] >= args.min_free_files))
        record.update(headroom, filesystem_free_bytes=filesystem_free,
                      required_bytes=required, required_files=args.min_free_files,
                      status='PASS' if passed else 'FAIL')
        print(json.dumps(record), flush=True)
        return 0 if passed else 1
    return 1


if __name__ == '__main__':
    sys.exit(main())
