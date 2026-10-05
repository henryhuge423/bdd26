import numpy as np
import pytest

from nucseg.postproc.existence import (candidate_geometry, distance_to_base, distance_to_base_map,
                                       existence_pass)


def test_existence_pass_matches_the_frozen_ef_menu_semantics():
    # a30: plain area floor at the boundary
    assert existence_pass({"area": 29, "class": 3}, min_area=30) is False
    assert existence_pass({"area": 30, "class": 3}, min_area=30) is True
    # a30d: Dead additions are exempt from the floor, non-Dead are not
    assert existence_pass({"area": 4, "class": 4}, min_area=30, dead_exempt=True) is True
    assert existence_pass({"area": 4, "class": 3}, min_area=30, dead_exempt=True) is False
    # off: no floor keeps everything
    assert existence_pass({"area": 1, "class": 1}, min_area=0) is True


def disk(centre, r, shape=(64, 64)):
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= r * r


def test_geometry_disk_is_round_sliver_is_not():
    d = candidate_geometry(disk((32, 32), 6))
    assert d["area"] == 113
    assert d["circularity"] >= 0.8
    assert d["extent"] >= 0.6  # rasterised disk fills ~2/3 of its bbox at this radius
    assert d["border"] is False
    sliver = np.zeros((64, 64), bool)
    sliver[np.arange(5, 25), np.arange(5, 25)] = True  # diagonal 1-px line: area 20, bbox 20x20
    s = candidate_geometry(sliver)
    assert s["area"] == 20
    assert s["circularity"] <= 0.65
    assert s["extent"] <= 0.1


def test_geometry_border_contact_and_single_pixel():
    edge = np.zeros((64, 64), bool)
    edge[0, 10:20] = True
    g = candidate_geometry(edge)
    assert g["border"] is True
    one = np.zeros((64, 64), bool)
    one[40, 40] = True
    p = candidate_geometry(one)
    assert p["area"] == 1 and 0.0 <= p["circularity"] <= 1.0


def test_distance_to_base_gaps_and_contact():
    base = np.zeros((64, 64), bool)
    base[:, :20] = True
    dist = distance_to_base_map(base)
    assert distance_to_base(disk((25, 32), 3), dist) == pytest.approx(3)  # leftmost pixel at x=22, base ends x=19
    assert distance_to_base(disk((18, 20), 3), dist) == 0.0  # centre inside the base band


def test_distance_map_error_on_empty_base():
    with pytest.raises(ValueError):
        distance_to_base_map(np.zeros((8, 8), bool))
