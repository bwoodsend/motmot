import numpy as np
import pytest

from motmot._polygon_walk import connected, slug
from motmot import Mesh, geometry
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

    faces, count = self.group_connected_polygons()
    assert count == 2
    assert faces[0] == 0

    assert np.all(faces[~upper_half] == upper_half[0])
    assert np.all(faces[upper_half] == 1 - upper_half[0])

    from rockhopper import RaggedArray
    grouped = RaggedArray.group_by(self.vectors, faces, count)
    halves = [Mesh(i) for i in grouped]
    assert (self[faces == 0].vectors == halves[0].vectors).all()
    assert (self[faces == 1].vectors == halves[1].vectors).all()


def test_closed_vertex_map():
    """Test Mesh.vertex_map on the simplest possible closed mesh - a
    tetrahedron."""
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    faces = np.array([[2, 1, 0], [1, 2, 3], [0, 3, 2], [3, 0, 1]])
    self = Mesh(vertices, faces)

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
        # pre-sorting. Sorting vertices is fiddly. Instead look up their
        # vertex IDs, sort those, then compare to ``sorted(neighbours)``.
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


def test_on_boundary():
    """Test Mesh.on_boundary()."""
    self = square_grid(3)
    # The boundaries of the square_grid() mesh are just its extremes.
    target = np.any(self.min[:2] == self.vertices[:, :2], axis=1) \
         | np.any(self.vertices[:, :2] == self.max[:2], axis=1)

    on_edge = [self.on_boundary(i) for i in self.vertices]
    assert on_edge == target.tolist()
    on_edge = [self.on_boundary(i) for i in range(len(self.vertices))]
    assert target.tolist() == on_edge

    with pytest.raises(ValueError, match="Only single"):
        self.on_boundary(self.vertices[:2])


@pytest.mark.parametrize("strict", [False, True])
def test_local_maxima(strict):
    # Create an egg-box shaped (hilly) surface.
    self = square_grid(30)
    self = Mesh(self.vertex_table.destroy(), self.faces)
    self.vertices[:, 2] = np.round(
        np.sin(10 * np.pi * (self.vertices[:, 0] + .05)) *
        np.sin(10 * np.pi * self.vertices[:, 1]), 5) * .1

    # Use geometric height for simplicity.
    heights = self.vertices[:, 2]

    # This egg-box should have:
    #   * strict local maxima at the top of each hill,
    #   * soft local maxima in the saddle between 4 hills,
    #   * all kinds of nonsense on the boundaries.

    # Find all locally high points, allowing ones on the boundaries.
    maxima = self.local_maxima(heights, strict=strict)

    # Explicitly check each vertex's height against its neighbours use a more
    # explicit, brute force approach.
    for vertex_id in range(len(self.vertices)):
        neighbours = self.vertex_map[vertex_id]
        if strict:
            is_maxima = (heights[neighbours] < heights[vertex_id]).all()
        else:
            is_maxima = (heights[neighbours] <= heights[vertex_id]).all()
        assert is_maxima == (vertex_id in maxima)

    # Find all locally high points, removing ones on the boundaries.
    maxima_ = self.local_maxima(heights, strict=strict, boundaries=False)
    for vertex_id in maxima_:
        assert vertex_id in maxima
        if strict:
            assert heights[vertex_id] == .1
        else:
            assert heights[vertex_id] in (0, .1)

    with pytest.raises(ValueError):
        self.local_maxima([1, 2, 3])


@pytest.mark.parametrize("strict", [False, True])
def test_local_maxima_with_colliding_indices(strict):
    """Test Mesh().local_maxima() where columns in mesh.faces contain duplicates
    and those duplicates disagree on whether the duplicated vertex id is a local
    maxima. Prevents a nastily subtle bug caused by NumPy's buffering."""

    # Create a mesh which is just 2 lines joining 3 vertices of ascending
    # height.
    vertices = geometry.zip(0, 0, [1, 2, 3])
    heights = vertices[:, 2]
    # The middle vertex id (1) appears in the same column of faces.
    faces = [[1, 0], [1, 2]]

    # The only local maxima is the 3rd point. But if done wrong, the middle
    # point may be included too.

    # Try where (1, 0) is seen before (1, 2).
    self = Mesh(vertices, faces)
    assert self.local_maxima(heights, boundaries=True).tolist() == [2]

    # Try where (1, 0) is seen after (1, 2).
    self = Mesh(vertices, faces[::-1])
    assert self.local_maxima(heights, boundaries=True).tolist() == [2]
