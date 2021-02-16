# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from motmot import geometry as g

pytestmark = pytest.mark.order(0)


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
        assert np.all(transformed[..., i] == v(points))


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