# -*- coding: utf-8 -*-
"""
"""
import re
import copy
import pickle

import numpy as np
import pytest

from motmot import Mesh
from tests import data, ids_mesh, vectors_mesh, assert_mesh_equal, closed_mesh

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


def test_translate():
    self = ids_mesh(5)
    old = self.vertices.copy()
    self.translate([1, 2, 3])
    assert np.all(self.vertices == old + [1, 2, 3])

    self = vectors_mesh(5)
    old = self.vectors.copy()
    self.translate([1, 2, 3])
    assert np.all(self.vectors == old + [1, 2, 3])


def test_rotate():
    m = Mesh.rotation_matrix([0, 0, 1], np.pi / 2).round(3)

    self = ids_mesh(5)
    old = self.vertices.copy()
    self.rotate_using_matrix(m)
    assert np.all(self.vertices.T == [old[:, 1], -old[:, 0], old[:, 2]])
    vectors = self.vectors.copy()
    self.rotate_using_matrix(m, [1, 0, 0])
    assert np.all(self.vertices.T == [1 - old[:, 0], 1 - old[:, 1], old[:, 2]])

    self = Mesh(old[self.ids])
    self.rotate_using_matrix(m)
    assert np.all(self.vectors == vectors)


def test_reset():
    self = ids_mesh(10)
    old = self.vectors
    self.vertices[:] += 1
    assert np.array_equal(self.vectors, old)
    self.reset()
    assert np.array_equal(self.vectors, old + 1)


def test_repr():
    r = repr(Mesh(np.empty((6, 4, 3))))
    assert re.fullmatch(
        r'<Vectors Mesh at 0x[0-9a-fA-F]+ \| 6 4-sided polygons>', r)

    r = repr(Mesh(np.empty((10, 3)), np.empty((6, 4))))
    assert re.fullmatch(
        r'<IDs Mesh at 0x[0-9a-fA-F]+ \| 10 vertices \| 6 4-sided polygons>', r)


@pytest.mark.parametrize("make_mesh", [ids_mesh, vectors_mesh])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("crop_at", [np.min, np.mean, np.max])
def test_cropped(make_mesh, in_place, crop_at):
    mesh: Mesh = make_mesh(10)
    threshold = crop_at(mesh.x[:, 0])
    placebo = np.empty(len(mesh))
    mask = mesh.x[:, 0] > threshold
    cropped = mesh.crop(mask, in_place=in_place)
    assert (cropped is mesh) is in_place
    assert len(cropped) == len(placebo[mask])
    assert len(cropped) == len(cropped.vectors)
    assert (cropped.x[:, 0] > threshold).all()


pickle_copy = lambda mesh: pickle.loads(pickle.dumps(mesh))
deep_copy_methods = [pickle_copy, copy.deepcopy, Mesh.copy]
shallow_copy_methods = [lambda mesh: mesh.copy(deep=False), copy.copy]


@pytest.mark.parametrize("copy_method",
                         shallow_copy_methods + deep_copy_methods)
@pytest.mark.parametrize("make_mesh", [ids_mesh, vectors_mesh])
def test_copy(copy_method, make_mesh):
    """Test shallow copying either using Mesh.copy(deep=False) or copy.copy().
    Test deep-copying via Mesh.copy(), copy.deepcopy() and pickle/unpickle."""
    mesh = make_mesh(20)
    copy = copy_method(mesh)
    assert_mesh_equal(copy, mesh)

    if copy_method in shallow_copy_methods:
        # This is a shallow copy. The contents should be the same objects.
        if mesh.is_ids_mesh:
            assert copy.__ids__ is mesh.__ids__
            assert copy.__vertices__ is mesh.__vertices__
        else:
            assert copy.__vectors__ is mesh.__vectors__
    else:

        # This is a deep copy. The contents should be copied too.
        if mesh.is_ids_mesh:
            assert not np.shares_memory(copy.__ids__, mesh.__ids__)
            assert not np.shares_memory(copy.__vertices__, mesh.__vertices__)
        else:
            assert not np.shares_memory(copy.__vectors__, mesh.__vectors__)


@pytest.mark.parametrize("make_mesh", [ids_mesh, vectors_mesh])
def test_centers(make_mesh):
    self = make_mesh(100)
    assert self.centers == pytest.approx(np.mean(self.vectors, axis=1))


@pytest.mark.parametrize("make_mesh", [ids_mesh, vectors_mesh, closed_mesh])
def test_polygon_map(make_mesh):
    self = make_mesh(100)
    # For each polygon, what vertices does it share with each neighbour.
    points_match = self.ids[self.polygon_map][:, :, np.newaxis] \
                   == self.ids[:, np.newaxis, :, np.newaxis]

    from motmot._polygon_map import make_polygon_map
    no_C = make_polygon_map(self.ids, len(self.vertices), use_C=False)
    assert np.all(self.polygon_map == no_C)
    no_neighbour_mask = self.polygon_map == -1

    assert self.polygon_map.shape == self.ids.shape
    if make_mesh is closed_mesh:
        assert not np.any(no_neighbour_mask)

    # Count how many each polygon has in common with each neighbour.
    match_counts = np.sum(points_match, axis=(2, 3))

    # The should (provided said polygon has a neighbour) all be 2 i.e. the
    # vertex on either end of the edge they share. In very rare cases though, a
    # polygon may share multiple edges with a neighbour.
    assert np.all(match_counts[~no_neighbour_mask] >= 2)
