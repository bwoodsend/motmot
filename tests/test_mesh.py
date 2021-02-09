# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pytest

from motmot import Mesh
from tests import data

pytestmark = pytest.mark.order(2)


def test_basics():
    """Test the basics of a mesh, generating vertices/ids from a vectors mesh
    and vectors from a vertices/ids mesh."""
    # Read a mesh from an STL file. This is automatically a vectors mesh.
    mesh = Mesh(data.rabbit_path, name="cat")
    assert mesh.name == "cat"

    # Test generating vertices/ids from a vectors mesh.
    # The work is all handled by hoatzin and therefore doesn't need to be tested
    # extensively here.
    assert np.array_equal(mesh.vectors, mesh.vertices[mesh.ids])
    _assert_first_appearances_are_sorted(mesh.ids.flat, len(mesh.vertices))

    # A vertices/ids mesh.
    self = Mesh(mesh.vertices, mesh.ids, name="octopus")
    assert self.ids is mesh.ids
    assert np.array_equal(self.vertices, mesh.vertices)
    assert self.name == "octopus"

    # Length and per_polygon should be the same for both types of mesh.
    assert len(mesh) == len(self) == len(mesh.vectors)
    assert mesh.per_polygon == self.per_polygon == mesh.vectors.shape[1]

    # Test generating vectors from vertices/ids.
    assert np.array_equal(self.vectors, mesh.vectors)
    assert self._vertex_table.unique_count == len(self.vertices)

    # Test the duplicity warning for duplicate vertices in a vertices/ids mesh.
    vertices_with_duplicate = mesh.vertices[np.arange(-1, len(mesh.vertices))]
    contains_duplicate = Mesh(vertices_with_duplicate, mesh.ids)
    with pytest.warns(UserWarning):
        contains_duplicate._vertex_table


def _assert_first_appearances_are_sorted(ids, max):
    """Assert that the 1st appearance of each id is the sequence range(0, max).
    """
    seen = np.zeros(max)
    highest = -1
    for id in ids:
        assert 0 <= id < max
        if not seen[id]:
            assert id == highest + 1
            highest = id
            seen[id] = True
    assert highest == max - 1


def test_slices():
    self = Mesh(np.zeros((10, 4, 3)))
    assert self.x.shape == (10, 4)

    y = np.arange(40).reshape((10, 4))
    self.y = y
    assert np.array_equal(self.vectors[:, :, 1], y)
