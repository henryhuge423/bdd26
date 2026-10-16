"""Owned atomic checkpoints and complete stochastic state at epoch boundaries."""
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA device visibility differs')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(path,**state):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
