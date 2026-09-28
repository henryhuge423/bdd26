"""Phase-2 upscale plumbing: resolution tags, run-config resolution, the per-instance table
invariant (table ids must survive the nearest downsampling to the native map), and encoder
gradient checkpointing actually taking effect inside UNIEncoder.forward."""
import json

import cv2
import numpy as np
import torch

from nucseg.cellvit.engine import instance_table_rows, res_tag, run_upscale
from nucseg.cellvit.model import UNIEncoder


def test_res_tag_empty_at_native_resolution():
    assert res_tag(1) == ""
    assert res_tag(2) == "_x2"


def test_run_upscale_override_then_config_then_default(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"upscale": 2}))
    assert run_upscale(tmp_path, None) == 2               # run's config.json
    assert run_upscale(tmp_path, 3) == 3                  # explicit CLI override wins
    assert run_upscale(tmp_path, 1) == 1
    bare = tmp_path / "bare"
    bare.mkdir()
    assert run_upscale(bare, None) == 1                   # no config -> native
    nokey = tmp_path / "nokey"
    nokey.mkdir()
    (nokey / "config.json").write_text(json.dumps({"split": 1}))
    assert run_upscale(nokey, None) == 1                  # config without upscale key


def _thin_nucleus_maps():
    """8x8 model-resolution map: nucleus 1 spans even rows/cols (survives 2x nearest), nucleus 2
    lives only on odd row/col and vanishes from the 4x4 downsample."""
    inst_model = np.zeros((8, 8), np.int32)
    inst_model[0:4, 0:4] = 1
    inst_model[5, 5] = 2
    inst_final = cv2.resize(inst_model, (4, 4), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return inst_model, inst_final


def test_instance_table_rows_drop_ids_lost_in_downsample():
    inst_model, inst_final = _thin_nucleus_maps()
    assert 2 in np.unique(inst_model) and 2 not in np.unique(inst_final)  # precondition
    prob = np.random.default_rng(0).random((8, 8, 6))
    img, ids, probs = instance_table_rows(inst_final, inst_model, prob, img_idx=7)
    assert list(ids) == [1]                       # phantom id 2 must not appear in the table
    assert img.tolist() == [7]
    assert np.allclose(probs[0], prob[inst_model == 1].mean(0))  # averaged at model resolution


def test_instance_table_rows_keep_all_ids_without_downsample():
    inst_model, _ = _thin_nucleus_maps()
    prob = np.random.default_rng(1).random((8, 8, 6))
    _, ids, probs = instance_table_rows(inst_model, inst_model, prob, img_idx=0)
    assert list(ids) == [1, 2]
    assert np.allclose(probs[1], prob[inst_model == 2].mean(0))


class _Counting(torch.nn.Module):
    """Shape-preserving stub whose backward needs an intermediate tensor created inside the block
    (what checkpointing discards and must recompute); a bare Identity would never re-run."""
    calls = 0

    def forward(self, x):
        _Counting.calls += 1
        h = x * 2.0
        return h * torch.sigmoid(h)


def test_uni_encoder_grad_checkpointing_recomputes_blocks():
    enc = UNIEncoder()
    enc.blocks = torch.nn.ModuleList([_Counting() for _ in enc.blocks])  # shape-preserving stubs
    enc.set_grad_checkpointing()
    assert enc.grad_checkpointing
    enc.train()
    cls, feats = enc(torch.zeros(1, 3, 224, 224))
    n_after_forward = _Counting.calls
    (cls.sum() + sum(f.sum() for f in feats)).backward()
    # with checkpointing each block is re-executed during backward; without it stays at n
    assert _Counting.calls > n_after_forward
