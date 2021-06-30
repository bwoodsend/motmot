# -*- coding: utf-8 -*-
"""
"""
import re
import copy
import pickle

import numpy as np
import pytest

from motmot import Mesh, geometry
from tests import data, faces_mesh, vectors_mesh, assert_mesh_equal, closed_mesh

pytestmark = pytest.mark.order(2)


def test_basics():
    """Test the basics of a mesh, generating vertices/faces from a vectors mesh
    and vectors from a vertices/faces mesh."""
    # Read a mesh from an STL file. This is automatically a vectors mesh.
    mesh = Mesh(data.rabbit_path, name="cat")
    assert mesh.name == "cat"

    # Test generating vertices/faces from a vectors mesh.
    # The work is all handled by hirola and therefore doesn't need to be tested
    # extensively here.
    assert np.array_equal(mesh.vectors, mesh.vertices[mesh.faces])
    _assert_first_appearances_are_sorted(mesh.faces.flat, len(mesh.vertices))

    # A vertices/faces mesh.
    self = Mesh(mesh.vertices, mesh.faces, name="octopus")
    assert self.faces is mesh.faces
    assert np.array_equal(self.vertices, mesh.vertices)
    assert self.name == "octopus"

    # Length and per_polygon should be the same for both types of mesh.
    assert len(mesh) == len(self) == len(mesh.vectors)
    assert mesh.per_polygon == self.per_polygon == mesh.vectors.shape[1]

    # Test generating vectors from vertices/faces.
    assert np.array_equal(self.vectors, mesh.vectors)
    assert len(self.vertex_table) == len(self.vertices)

    # Test the duplicity warning for duplicate vertices in a vertices/faces
    # mesh.
    vertices_with_duplicate = mesh.vertices[np.arange(-1, len(mesh.vertices))]
    contains_duplicate = Mesh(vertices_with_duplicate, mesh.faces)
    with pytest.warns(UserWarning):
        contains_duplicate.vertex_table


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


def test_square_normals():
    self = faces_mesh(10, 4)
    magnitudes = geometry.magnitude_sqr(self.units)
    assert magnitudes[np.isfinite(magnitudes)] == pytest.approx(1)


def test_square_areas():
    """Test non-triangular area calculation on 5 similar squares."""
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    self = Mesh(np.array([
            square,  # The simplest square should have area 1.
            square + 1,  # Translating the square should have no effect.
            np.roll(square, 1, axis=1) / 2,  # 1/2 scale -> 1/4 area.
            square * 5,  # Scale up by 5 -> 25 * area.
            square * [1, 0, 1],  # Collapsing an axis gives it area 0.
        ]))  # yapf: disable
    assert self.areas == pytest.approx([1, 1, .25, 25, 0])


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
    self = faces_mesh(5)
    old = self.vertices.copy()
    self.translate([1, 2, 3])
    assert np.all(self.vertices == old + [1, 2, 3])

    self = vectors_mesh(5)
    old = self.vectors.copy()
    self.translate([1, 2, 3])
    assert np.all(self.vectors == old + [1, 2, 3])


def test_rotate():
    m = Mesh.rotation_matrix([0, 0, 1], np.pi / 2).round(3)

    self = faces_mesh(5)
    old = self.vertices.copy()
    self.rotate_using_matrix(m)
    assert np.all(self.vertices.T == [old[:, 1], -old[:, 0], old[:, 2]])
    vectors = self.vectors.copy()
    self.rotate_using_matrix(m, [1, 0, 0])
    assert np.all(self.vertices.T == [1 - old[:, 0], 1 - old[:, 1], old[:, 2]])

    self = Mesh(old[self.faces])
    self.rotate_using_matrix(m)
    assert np.all(self.vectors == vectors)


def test_reset():
    self = faces_mesh(10)
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
        r'<Faces Mesh at 0x[0-9a-fA-F]+ \| 10 vertices \| 6 4-sided polygons>',
        r,
    )


@pytest.mark.parametrize("make_mesh", [faces_mesh, vectors_mesh])
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
@pytest.mark.parametrize("make_mesh", [faces_mesh, vectors_mesh])
def test_copy(copy_method, make_mesh):
    """Test shallow copying either using Mesh.copy(deep=False) or copy.copy().
    Test deep-copying via Mesh.copy(), copy.deepcopy() and pickle/unpickle."""
    mesh = make_mesh(20)
    copy = copy_method(mesh)
    assert_mesh_equal(copy, mesh)

    if copy_method in shallow_copy_methods:
        # This is a shallow copy. The contents should be the same objects.
        if mesh.is_faces_mesh:
            assert copy.__faces__ is mesh.__faces__
            assert copy.__vertices__ is mesh.__vertices__
        else:
            assert copy.__vectors__ is mesh.__vectors__
    else:

        # This is a deep copy. The contents should be copied too.
        if mesh.is_faces_mesh:
            assert not np.shares_memory(copy.__faces__, mesh.__faces__)
            assert not np.shares_memory(copy.__vertices__, mesh.__vertices__)
        else:
            assert not np.shares_memory(copy.__vectors__, mesh.__vectors__)


@pytest.mark.parametrize("make_mesh", [faces_mesh, vectors_mesh])
def test_centers(make_mesh):
    self = make_mesh(100)
    assert self.centers == pytest.approx(np.mean(self.vectors, axis=1))


@pytest.mark.parametrize("make_mesh", [faces_mesh, vectors_mesh, closed_mesh])
def test_polygon_map(make_mesh):
    self = make_mesh(100)
    # For each polygon, what vertices does it share with each neighbour.
    points_match = self.faces[self.polygon_map][:, :, np.newaxis] \
                   == self.faces[:, np.newaxis, :, np.newaxis]

    from motmot._polygon_map import make_polygon_map
    no_C = make_polygon_map(self.faces, len(self.vertices), use_C=False)
    assert np.all(self.polygon_map == no_C)
    no_neighbour_mask = self.polygon_map == -1

    assert self.polygon_map.shape == self.faces.shape
    if make_mesh is closed_mesh:
        assert not np.any(no_neighbour_mask)

    # Count how many each polygon has in common with each neighbour.
    match_counts = np.sum(points_match, axis=(2, 3))

    # The should (provided said polygon has a neighbour) all be 2 i.e. the
    # vertex on either end of the edge they share. In very rare cases though, a
    # polygon may share multiple edges with a neighbour.
    assert np.all(match_counts[~no_neighbour_mask] >= 2)


def test_as_array():
    """numpy.asarray(mesh) must be blocked."""
    self = faces_mesh(10)
    with pytest.raises(TypeError, match="Meshes can't .*"):
        np.asarray(self)


def test_displacements():
    self = faces_mesh(5, 4)
    assert self.displacements.shape == (5, 4, 3)

    # Another non-test.
    # Just the same thing as the original but in a for loop.
    for i in range(len(self)):
        for j in range(self.per_polygon):
            neighbour = self.polygon_map[i, j]
            if neighbour == -1:
                assert np.isnan(self.displacements[i, j]).all()
            else:
                assert np.all(self.centers[neighbour] - self.centers[i] \
                              == self.displacements[i, j])


def test_closes_point():
    """Test Mesh.closed_point().

    There are an unfortunate number of permutations involved in this method.
    This test must cover all combinations of:

        * distance_upper_bound is given or omitted.
        * interpolate is either true or false.

    PyKDTree is hardcoded to only accept 2D input arrays. Also verify that our
    workaround for other array shapes works.

    """
    from tests import square_grid
    self = square_grid(5)
    self.translate([0, 0, 5])

    # Single point, no interpolate, no upper bound.
    assert self.closest_point([.012, .013, 1], interpolate=False) \
           == pytest.approx([.015, .015, 5])
    # The same but with interpolation.
    assert self.closest_point([.012, .013, 1]).tolist() == [.012, .013, 5]
    # Multiple points.
    a = self.closest_point([[.016, .09, 2], [.012, .083, 9]])
    assert np.allclose(a, [[.016, .09, 5], [.012, .083, 5]])

    # Multiple points, interpolation off, with upper bound.
    a, b = self.closest_point([[.012, .023, 4.8], [.032, .043, 1]],
                              distance_upper_bound=.5,
                              interpolate=False).tolist()
    assert a == pytest.approx([.015, .025, 5])
    assert np.isnan(b).all()

    # Multiple points, interpolation on, with upper bound.
    a, b = self.closest_point([[.12, .23, 4.8], [.032, .043, 1]],
                              distance_upper_bound=.5)
    assert a == pytest.approx([.12, .23, 5])
    assert np.isnan(b).all()


def test_writing_core_attributes_of_vectors_mesh():
    """Tests writing to the vectors, vertices and faces attributes of a faces
    mesh.

    - `vectors` should not be writeable.
    - `vertices` should be writeable, resizable, allow changes of dtypes but
      always silently enforce C contiguous arrays.
    - `faces` should be writeable and resizable but silently enforce C
      contiguity and numpy.intp dtype.

    After any modifications, all lazy attributes should be reset.

    """
    self = faces_mesh(5, 4)
    assert self.vertices.flags.writeable

    old_vertices = self.vertices
    old_normals = self.normals
    old_faces = self.faces

    self.vertices = np.asarray(np.append([[9, 9, 9]], self.vertices, axis=0),
                               dtype=np.float32, order="f")
    assert self.vertices is not old_vertices
    assert self.normals is not old_normals
    assert self.faces is old_faces
    assert self.vertices.flags.c_contiguous
    assert self.vertices.dtype == np.float32

    self.faces = np.asarray(self.faces + 1, dtype=np.uint8, order="f")
    assert self.faces.dtype == np.intp
    assert self.faces.dtype

    with pytest.raises(ValueError, match="faces mesh's vectors .* readonly"):
        self.vectors += 1


def test_writing_core_attributes_of_faces_mesh():
    """Tests writing to the vectors, vertices and faces attributes of a vectors
     mesh.

    - `vectors` should be writeable, resizable, allow changes of dtypes but
      always silently enforce C contiguous arrays.
    - `vertices` should not be writeable.
    - `faces` should not be writeable.

    After any modifications, all lazy attributes should be reset.

    """
    self = vectors_mesh(5, 4)
    assert self.vectors.flags.writeable
    assert not self.vertices.flags.writeable

    old_vertices = self.vertices
    self.vectors += [1, 2, 3]
    assert np.all(self.vertices != old_vertices)

    with pytest.raises(ValueError, match="vectors mesh's vertices .*"):
        self.vertices = 8

    with pytest.raises(ValueError, match="vectors mesh's faces .*"):
        self.faces += 1

    self.vectors = np.asarray(self.vectors[:-2], order="f", dtype=np.float32)
    assert self.dtype == np.float32
    assert self.vectors.flags.c_contiguous


def test_invalid_core_inputs():
    """Test setting vectors, vertices or faces to arrays with invalid shapes.
    This can be done either on construction or later by setting attributes.
    """

    # --- Vectors ---

    # Anything not 3D is invalid.
    with pytest.raises(ValueError,
                       match=r"Vectors .* a 3D .* Received .* \(1,\)"):
        Mesh([1])
    with pytest.raises(ValueError, match=r".* \(3,\)"):
        Mesh([1, 2, 3])
    with pytest.raises(ValueError, match=r".* \(1, 3\)"):
        Mesh([[1, 2, 3]])

    # Test setting the attribute.
    self = Mesh([[[1, 2, 3]]])
    with pytest.raises(ValueError):
        self.vectors = [[1, 2, 3]]

    # Non C contiguous arrays must become contiguous.
    self.vectors = np.zeros((3, 3, 3), order="f")
    assert self.vectors.flags.c_contiguous

    # --- Vertices ---

    # An empty vertices array should be normalised to shape (0, 3)
    # i.e. Zero 3D vertices.
    self = Mesh([], [[1, 2, 3]])
    assert self.vertices.shape == (0, 3)

    # A single vertex should become a length 1 array of vertices.
    self.vertices = [1, 2, 3]
    assert self.vertices.shape == (1, 3)
    # The data type should be preserved.
    assert self.dtype == int

    with pytest.raises(
            ValueError,
            match=r".* vertices .* 3\. Received .* shape \(1, 4\)\."):
        self.vertices = [[1, 2, 3, 4]]

    with pytest.raises(
            ValueError,
            match=r"'vertices' .* too many .* A 2D .* shape \(5, 4, 3\)\."):
        self.vertices = np.zeros((5, 4, 3))

    # Non C contiguous arrays must become contiguous.
    self.vertices = np.zeros((4, 3), order="f")
    assert self.vertices.flags.c_contiguous

    # --- Faces ---

    # Faces follows almost the same rules as Vertices.
    self = Mesh([], [])
    # Assume zero triangular polygons by default.
    assert self.faces.shape == (0, 3)

    # Again, a single polygon should be promoted to a 2D array with length 1.
    self.faces = np.array([1, 2, 3], np.int8)
    assert self.faces.shape == (1, 3)
    # But the integer type must be normalised to ptrdiff_t.
    assert self.faces.dtype == np.intp

    # C contiguity must be enforced.
    self.faces = np.zeros((10, 3), order="f")
    assert self.faces.flags.c_contiguous

    # Arbitrary shapes are allowed...
    self.faces = [[1, 2, 3, 4]]
    assert self.faces.shape == (1, 4)
    self.faces = np.empty((0, 12))
    assert self.faces.shape == (0, 12)
    assert (len(self), self.per_polygon) == (0, 12)

    # ... unless they are 3D.
    with pytest.raises(
            ValueError,
            match=r"'faces' .* too many .* A 2D .* shape \(5, 4, 3\)\."):
        self.faces = np.zeros((5, 4, 3))
