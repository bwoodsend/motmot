# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from motmot._polygon_walk import connected, slug
from motmot import Mesh
from tests import data, square_grid

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


def test_connected_per_edge_mask():
    polygon_map = np.array([
        [-1, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, -1],
    ], dtype=np.intp)

    assert connected(polygon_map, 0).tolist() == [0, 1, 2, 3, 4]

    mask = np.ones((5, 2), bool)
    assert connected(polygon_map, 0, mask).tolist() == [0, 1, 2, 3, 4]

    mask[2, 1] = False
    assert connected(polygon_map, 0, mask).tolist() == [0, 1, 2]
    assert connected(polygon_map, 4, mask).tolist() == [4, 3, 2, 1, 0]

    mask[:, 0] = False
    assert connected(polygon_map, 0, mask).tolist() == [0, 1, 2]
    assert connected(polygon_map, 4, mask).tolist() == [4]
    assert connected(polygon_map, 2, mask).tolist() == [2]


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


def test_closed_vertex_map():
    """Test Mesh.vertex_map on the simplest possible closed mesh - a
    tetrahedron."""
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    ids = np.array([[2, 1, 0], [1, 2, 3], [0, 3, 2], [3, 0, 1]])
    self = Mesh(vertices, ids)

    # In a tetrahedron, each vertex is connected to every other vertex except
    # itself. So the vertex_map RaggedArray will have 4 rows all of length 3.
    assert (self.vertex_map.ends - self.vertex_map.starts).tolist() == [3] * 4

    # And each row each should contain 0, 1, 2, 3 minus its own row number...
    map = self.vertex_map.to_rectangular_arrays()[0].tolist()
    # ... however, the order is arbitrary so we'll need to sort them first.
    [i.sort() for i in map]
    assert map == [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    _check_symmetry_and_duplicity(self.vertex_map)


def test_non_closed_vertex_map():
    """Test Mesh.vertex_map on a non-closed mesh. See the comments in
    Mesh.vertex_map for why non-closed meshes require special handling.
    Also briefly check Mesh.connected_vertices().
    """
    self = square_grid(2)
    assert len(self.vertex_map) == len(self.vertices)

    for (id, neighbours) in enumerate(self.vertex_map):
        point = self.vertices[id]

        # Each point should have a neighbour above, below, to the left and to
        # the right...
        expected = point + \
              np.array([[-.01, 0, 0], [.01, 0, 0], [0, -.01, 0], [0, .01, 0]])

        # ... except the points at the boundaries which will have some missing.
        # Chop off any points that stick out beyond the mesh boundaries.
        expected = expected[(self.min <= expected).all(1)]
        expected = expected[(expected <= self.max).all(1)]

        # Sanity check.
        assert 2 <= len(expected) <= 4

        # Ideally we'd check that ``self.vertices[neighbours] == expected`` but
        # the order is not guaranteed so this will fail without some
        # pre-sorting. Sorting vertices is fiddly. Instead look up their ids,
        # sort those, then compare to ``sorted(neighbours)``.
        expected_ids = self.vertex_table[expected]
        assert sorted(neighbours) == sorted(expected_ids)

        # Check connected_vertices() for any major bogies.
        assert np.array_equal(self.connected_vertices(point),
                              self.vertices[neighbours])

    with pytest.raises(NotImplementedError, match="Must take a single"):
        self.connected_vertices(self.vertices)

    _check_symmetry_and_duplicity(self.vertex_map)


def _check_symmetry_and_duplicity(vertex_map):
    """Validate a Mesh.vertex_map.

    1. Check that if A is connected to B then B should also be connected to A.
    2. That there are no duplicate neighbours (A is connected to B and B).
    3. There are no self references (A is connected to A).

    """
    for (i, neighbours) in enumerate(vertex_map):
        # No self references (3).
        assert i not in neighbours
        for j in neighbours:
            # Symmetry and uniqueness (1) and (2).
            assert (vertex_map[j] == i).sum() == 1
