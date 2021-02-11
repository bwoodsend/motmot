# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pytest

from motmot import Mesh
from tests import data, ids_mesh, vectors_mesh

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


def test_bounds():
    """Test the min, max, bounds and dims attributes of Mesh."""
    self = Mesh(np.empty((10, 2, 3), dtype=np.uint8))
    self.x = np.arange(20).reshape((10, 2))
    self.y = np.arange(50, 70).reshape((10, 2))
    self.z = 200
    assert self.bounds.tolist() == [[0, 50, 200], [19, 69, 200]]
    assert self.dims.tolist() == [19, 19, 0]
    # min and max should be aliases for bounds[0] and bounds[1].
    assert np.shares_memory(self.min, self.bounds[0])
    assert np.shares_memory(self.max, self.bounds[1])


def test_normals():
    from motmot import geometry
    mesh = Mesh(data.rabbit_path)

    # ``inner_product(normals, vectors)`` should give the same value for each
    # corner.
    n_projections = geometry.inner_product(mesh.normals[:, np.newaxis],
                                           mesh.vectors)

    # Get the inner product at the 1st corner and all the corners.
    x, y = np.broadcast_arrays(n_projections[:, 0, np.newaxis], n_projections)

    # pytest.approx() seems super slow here for some reason.
    assert np.allclose(x, y, atol=y.ptp() * 1e-5)

    normals = geometry.magnitude(mesh.normals, keepdims=True) * mesh.units
    assert np.allclose(normals, mesh.normals)
    assert np.allclose(geometry.magnitude_sqr(mesh.units), 1)

    assert np.allclose(geometry.inner_product(mesh.normals, mesh.units),
                       mesh.areas * 2)
    assert mesh.area > 0


def test_vertex_normals():
    from motmot.geometry import magnitude
    # Create a mesh with deliberate missing vertices.
    self = Mesh(np.random.random((10, 3)), np.random.randint(0, 8, (30, 3)))

    # Bit of a non-test.
    assert self.vertex_normals.shape == self.vertices.shape

    # Vertices which appear in no polygons (and therefore have no meaningful
    # definition of normals) must have vertex_normals [nan, nan, nan].
    # All other vertices should have finite vertex_normals with magnitude 1.
    invalid = self.vertex_counts == 0
    assert magnitude(self.vertex_normals[~invalid]) == pytest.approx(1)
    assert np.isnan(self.vertex_normals[invalid]).all()
