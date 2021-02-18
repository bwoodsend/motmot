# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from motmot._polygon_walk import connected, slug
from motmot import Mesh
from tests import data

pytestmark = pytest.mark.order(4)


def test_connected():
    polygon_map = np.array([
        [3, 1],
        [2, 3],
        [-1, -1],
        [-1, 4],
        [-1, 4],
    ], dtype=np.intp)

    assert connected(polygon_map, 0).tolist() == [0, 3, 1, 4, 2]
    assert connected(polygon_map, 1).tolist() == [1, 2, 3, 4]
    assert connected(polygon_map, 2).tolist() == [2]
    assert connected(polygon_map, 3).tolist() == [3, 4]
    assert connected(polygon_map, 4).tolist() == [4]

    mask = np.ones(5, bool)
    mask[2] = False
    assert 2 not in connected(polygon_map, 0, mask)
    mask[0] = False
    assert connected(polygon_map, 0, mask).tolist() == [0, 3, 1, 4]


def test_connected_polygons():
    self = Mesh(data.rabbit_path)
    upper_half = self.centers[:, 2] > self.z.mean()
    self.z[upper_half] += 100

    assert not upper_half[self.connected_polygons(upper_half.argmin())].any()
    assert upper_half[self.connected_polygons(upper_half.argmax())].all()

    ids, count = self.group_connected_polygons()
    assert count == 2
    assert ids[0] == 0

    assert np.all(ids[~upper_half] == upper_half[0])
    assert np.all(ids[upper_half] == 1 - upper_half[0])

    from rockhopper import RaggedArray
    grouped = RaggedArray.group_by(self.vectors, ids, count)
    halves = [Mesh(i) for i in grouped]
    assert (self[ids == 0].vectors == halves[0].vectors).all()
    assert (self[ids == 1].vectors == halves[1].vectors).all()
