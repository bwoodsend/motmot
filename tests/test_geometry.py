import numpy as np
import pytest

from motmot import geometry as g


def _test_reduce(arr, operator: np.ufunc):
    assert np.array_equal(operator.reduce(arr, axis=-1),
                          g._reduce_last_axis(arr, operator))
    assert np.array_equal(operator.reduce(arr, axis=-1, keepdims=True),
                          g._reduce_last_axis(arr, operator, keepdims=True))


def test_reduce():
    _test_reduce([1, 2, 3], np.add)
    _test_reduce(np.arange(10).reshape((5, 2)), np.bitwise_xor)
    _test_reduce(np.empty((10, 0)), np.multiply)


def test_magnitude():
    assert g.magnitude([3, 4]) == 5


def test_inner():
    assert g.inner_product([1, 2, 3], [4, 5, 6]) == 1 * 4 + 2 * 5 + 3 * 6


def test_normalise():
    x = np.random.random((10, 3))
    assert pytest.approx(1) == g.magnitude_sqr(g.normalised(x))
    g.normalise(x)
    assert pytest.approx(1) == g.magnitude_sqr(x)


V0s = [np.eye(3), np.empty((0, 3)), np.array([.6, .8, 0])]


@pytest.mark.parametrize("v0_", V0s)
def test_orthagonal_basis(v0_):
    v0, v1, v2 = g.orthogonal_bases(v0_)
    assert g.inner_product(v0, v1) == pytest.approx(0)
    assert g.inner_product(v0, v2) == pytest.approx(0)
    assert g.inner_product(v1, v2) == pytest.approx(0)
    assert g.magnitude(v0) == pytest.approx(1)
    assert g.magnitude(v1) == pytest.approx(1)
    assert g.magnitude(v2) == pytest.approx(1)


def test_orthagonal_basis_error():
    with pytest.raises(ValueError, match="v0 must"):
        g.orthogonal_bases([0, 0, 0])


def test_unit_vector():
    self = g.UnitVector([3, 4, 0])
    assert self.vector.tolist() == [.6, .8, 0]
    assert np.all(self == [.6, .8, 0])
    assert repr(self) == "UnitVector([0.6, 0.8, 0.0])"
    assert self([5, 10, 3]).tolist() == 11
    assert self(self) == 1
    assert self(-self) == -1
    assert isinstance(self + 1, np.ndarray)
    assert isinstance(-self, g.UnitVector)

    points = np.random.random((10, 3))

    v0, v1, v2 = map(g.UnitVector, g.orthogonal_bases(self))
    _points = self.remove_component(points)
    assert self(_points) == pytest.approx(0)
    assert v1(_points) == pytest.approx(v1(points))
    assert v2(_points) == pytest.approx(v2(points))

    _points = self.with_(points, 4)
    assert self(_points) == pytest.approx(4)
    assert v1(_points) == pytest.approx(v1(points))
    assert v2(_points) == pytest.approx(v2(points))

    _points = self.match(points, points[0])
    assert self(_points) == pytest.approx(self(points[0]))
    assert v1(_points) == pytest.approx(v1(points))
    assert v2(_points) == pytest.approx(v2(points))

    _points = self.get_component(points)
    assert self(_points) == pytest.approx(self(points))
    assert v1(_points) == pytest.approx(0)
    assert v2(_points) == pytest.approx(0)

    for (p, v) in zip(g.get_components(points, v0, v1, v2), (v0, v1, v2)):
        assert np.all(p == v(points))

    transformed = g.get_components_zipped(points, v0, v1, v2)
    for (i, v) in enumerate((v0, v1, v2)):
        a = transformed[..., i]
        b = v(points)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        # These should be identical but, possibly due to one going via the GPU
        # and then other the CPU, or for some other reason, they are not.
        assert a == pytest.approx(b)


def test_matched_sign():
    self = g.UnitVector("z")

    assert (self.matched_sign(self) == self).all()
    assert (self.matched_sign(-self) == self).all()
    assert self.matched_sign([0, 1, -.5]).tolist() == [0, -1, .5]
    assert self.matched_sign([0, -1, 0]).tolist() == [0, -1, 0]


def test_furthest():
    self = g.UnitVector([1, 2, 3])
    points = np.random.random((100, 3))

    point, projection, arg = self.furthest(points, return_args=True,
                                           return_projection=True)
    assert projection == self(points).max()
    assert np.all(point == points[arg])

    assert np.all(self.furthest(points) == point)
    assert self.furthest(points, return_args=True)[1] == arg
    assert self.furthest(points, return_projection=True)[1] == projection

    reshaped = points.reshape((4, 25, 3))
    top_3 = points[self(points).argsort()[:-4:-1]]
    assert np.all(self.furthest(points, n=3) == top_3)
    assert np.all(self.furthest(reshaped, n=3) == top_3)
    assert np.all(self.furthest(reshaped) == top_3[0])
    assert self.furthest(reshaped, n=0).shape == (0, 3)

    with pytest.raises(ValueError):
        self.furthest(points[:0])


def test_named_unit_vectors():
    assert g.UnitVector("z").tolist() == [0, 0, 1]
    assert g.UnitVector("- J").tolist() == [0, -1, 0]

    with pytest.raises(ValueError, match="'Q' is not .*"):
        g.UnitVector("Q")


def test_center_of_mass():
    points = np.random.random((10, 10, 3))
    assert g.center_of_mass(points) == pytest.approx(points.mean(axis=(0, 1)))
    weights = points[..., 0] > .4
    assert g.center_of_mass(points, weights) == pytest.approx(
        points[weights].mean(axis=0))


def test_area():
    # Dots and lines should have area 0.
    assert g.area([[0, 0, 0]]) == 0
    assert g.area([[0, 0, 0], [0, 1, 0]]) == 0
    # As should triangles that are just lines.
    assert g.area([[0, 0, 0], [0, 1, 0], [0, 2, 0]]) == 0

    # A simple 1x1 right angle triangle.
    polygon = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    assert g.area(polygon) == .5

    # A simple 1x1 square.
    polygon.append([0, 1, 0])
    assert g.area(polygon) == 1

    # Remove a triangle from the square. This should reduce the area instead of
    # adding to it.
    polygon.append([.5, .5, 0])
    g.area(polygon) == .75

    polygon[-1][2] = 1


def test_zip():
    """Test geometry.zip()."""
    zipped = g.zip(0, 1, 2)
    assert zipped.tolist() == [0, 1, 2]
    assert zipped.dtype == int

    assert g.zip([1, 2, 3], 0).tolist() == [[1, 0], [2, 0], [3, 0]]

    x = np.random.random((4, 3))
    y = np.random.random((1, 3))
    zipped = g.zip(x, y)
    assert zipped.shape == (4, 3, 2)
    assert np.array_equal(zipped[..., 0], x)
    assert np.array_equiv(zipped[..., 1], y)


def test_unzip():
    """Test geometry.unzip()."""
    assert g.unzip([1, 2]) == (1, 2)
    data = np.arange(24)

    x, y, z = g.unzip(data.reshape((8, 3)))
    assert np.array_equal(x, np.arange(0, 24, 3))
    assert np.array_equal(y, np.arange(1, 24, 3))
    assert np.array_equal(z, np.arange(2, 24, 3))

    x_, y_, z_ = g.unzip(data.reshape((2, 4, 3)))
    assert x_.shape == y_.shape == z_.shape == (2, 4)
    assert np.array_equal(x_.ravel(), x)
    assert np.array_equal(y_.ravel(), y)
    assert np.array_equal(z_.ravel(), z)


def test_closest():
    """Test geometry.closest()."""
    points = g.zip(np.arange(10), np.arange(10)[:, np.newaxis])
    assert g.closest(points.reshape((-1, 2)), [4.4, 1.6]).tolist() == [4, 2]
    assert g.closest(points, [6.6, 9.1]).tolist() == [7, 9]


def test_snap_to_plane():
    points = [[1, 2, 3], [10, 11, 12], [0, 5, 3]]
    normal = [0, -1, 0]
    origins = [[0, 0, 0], [20, 3, 10], [13, 12, 91]]
    out = g.snap_to_plane(points, origins, normal).tolist()

    # `out` should be the same x and z values as `points` but same y values as
    # `origins`.
    assert out == [[1, 0, 3], [10, 3, 12], [0, 12, 3]]
