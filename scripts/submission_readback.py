"""Fail closed on stale tested source or failed storage write/fsync/readback."""
from pathlib import Path, PurePosixPath
import hashlib
import os
import datetime


def readback(directory, name):
    if Path(name).name != name:
        raise ValueError('readback name must be a basename')
    path=Path(directory)/name
    data=os.urandom(65536)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes()!=data:
        raise ValueError('storage readback mismatch')
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def verify_source(root, expected, inventory):
    if set(inventory)!=set(expected):
        raise ValueError('tested source inventory mismatch')
    for name,digest in expected.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name:
            raise ValueError('invalid source path')
        path=Path(root)/name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('tested source mismatch: '+name)
    return True
