"""Full-image HoVer HV targets with an explicit working-pixel area cutoff.

Adapted from vqdang/hover_net models/hovernet/targets.py. The full-image caller
already zero-pads the labels; the official center rounding and normalization
are preserved. Only the minimum area is configurable (historical default30).

MIT License
Copyright (c) 2020 vqdang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from numbers import Integral

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.morphology import remove_small_objects

from .official import fix_mirror_padding, get_bounding_box


def validate_hv_min_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("hv_min_size must be a positive integer in working pixels")
    return int(value)


def instance_hv_map(ann: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Compute HV on full, zero-padded labels; small instances receive zero HV."""
    min_size = validate_hv_min_size(min_size)
    fixed_ann = fix_mirror_padding(ann.copy())
    selected = remove_small_objects(fixed_ann, min_size=min_size)
    x_map = np.zeros(ann.shape, dtype=np.float32)
    y_map = np.zeros(ann.shape, dtype=np.float32)

    for inst_id in np.unique(selected):
        if inst_id == 0:
            continue
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        box = get_bounding_box(inst_map)
        box[0] -= 2
        box[2] -= 2
        box[1] += 2
        box[3] += 2
        inst_map = inst_map[box[0]:box[1], box[2]:box[3]]
        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue
        com = [int(v + 0.5) for v in center_of_mass(inst_map)]
        xs = np.arange(1, inst_map.shape[1] + 1) - com[1]
        ys = np.arange(1, inst_map.shape[0] + 1) - com[0]
        inst_x, inst_y = np.meshgrid(xs, ys)
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")
        for coords in (inst_x, inst_y):
            if np.min(coords) < 0:
                coords[coords < 0] /= -np.amin(coords[coords < 0])
            if np.max(coords) > 0:
                coords[coords > 0] /= np.amax(coords[coords > 0])
        x_box = x_map[box[0]:box[1], box[2]:box[3]]
        y_box = y_map[box[0]:box[1], box[2]:box[3]]
        x_box[inst_map > 0] = inst_x[inst_map > 0]
        y_box[inst_map > 0] = inst_y[inst_map > 0]
    return np.dstack([x_map, y_map])
