"""Small stdlib-only identity/learning gates, also usable by Expanse login Python."""
import hashlib
import json
import math
from pathlib import Path

TISSUE_SHA='64746d127bc28e88be6fdc4639af4f21a91fb1fda00e68869ea5fe61a574f615'
COHORT_SHA='55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22'


def validate_cohort(fit,audit):
    digest=hashlib.sha256(json.dumps(dict(fit=fit,audit=audit),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=COHORT_SHA:raise ValueError('frozen P3/P4 cohort identity mismatch')
    return digest


def require_positive_gradients(values):
    if set(values)!=set(('encoder','refinement','output')) or not all(math.isfinite(v) and v>0 for v in values.values()):
        raise ValueError('actual finite nonzero encoder/refinement/output gradients required')


def verify_hashes(root,hashes):
    for name,expected in hashes.items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('artifact identity mismatch: '+name)


def validate_precision_trace(native,trace):
    expected='torch.bfloat16' if native else 'torch.float32'
    if not trace['enabled'] or trace['amp_dtype']!='torch.bfloat16' or trace['logits_dtype']!=expected or trace['layers']!=dict(encoder='torch.bfloat16',refinement='torch.bfloat16',output='torch.bfloat16'):
        raise ValueError('CUDA BF16 autocast operation policy mismatch')
