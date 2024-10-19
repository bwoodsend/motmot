import numpy as np
import pytest

from motmot import Mesh
from tests import cylinder, icosasphere_10, icosasphere_15, icosasphere_1

pytestmark = pytest.mark.order(4)


def test_directional():
    self = Mesh(*icosasphere_1)

    assert self.curvature.directional.shape == (len(self), self.per_polygon, 3)
    edges = np.roll(self.vectors, -1, axis=1) - self.vectors
    assert np.cross(edges, self.curvature.directional) \
           == pytest.approx(0, abs=1e-8)


def test_scaleless():
    # Create a low and a high resolution ball both with radius 1.
    low = Mesh(*icosasphere_10)
    high = Mesh(*icosasphere_15)
    assert low.curvature.scaleless.shape == (len(low), low.per_polygon)

    # Sanity check that the balls do have radius 1.
    assert low.min.tolist() == [-1, -1, -1]
    assert low.max.tolist() == [1, 1, 1]

    # Scaleless curvature should be independent of mesh resolution.
    # This ball has radius 1 so the curvatures should on average be 1 / 1 == 1.
    assert low.curvature.scaleless.mean() == pytest.approx(1, abs=.01)
    assert high.curvature.scaleless.mean() == pytest.approx(1, abs=.01)

    # Scaleless curvature should be the reciprocal or the ball radius.
    # Double the size:
    low.vertices[:] *= 2
    low.reset()
    # Gives half the curvature.
    assert low.curvature.scaleless.mean() == pytest.approx(.5, abs=.01)


def test_signed():
    self = Mesh(*icosasphere_10)
    assert self.curvature.signed.shape == (len(self), self.per_polygon)
    assert np.all(self.curvature.signed > 0)

    self.faces[:] = self.faces[:, ::-1]
    self.reset()
    assert np.all(self.curvature.signed < 0)


@pytest.mark.parametrize("resolution", [3, 5, 10])
@pytest.mark.parametrize("radius", [1, 10])
def test_cylinder(resolution, radius):
    self = cylinder(resolution)
    self.vertices[:] *= radius

    # This cylinder is only a pipe with its circular ends missing.
    mask: np.ndarray = self.polygon_map != -1
    assert mask.mean() == .5

    # Missing neighbours should have nan curvatures.
    assert np.isnan(self.curvature.directional[~mask]).all()
    assert np.isfinite(self.curvature.directional[mask]).all()

    # Because the cylinder is oriented parallel to the Z axis, all non-nan rows
    # in curvature.directional should either be:
    #   [0, 0, sin(360 / resolution)] or [0, 0, -sin(360 / resolution)].
    target = np.array([0, 0, np.sin(np.deg2rad(360 / resolution))])
    finite = self.curvature.directional[mask]
    assert np.all(np.isclose(finite, target) | np.isclose(finite, -target))
    assert self.curvature.magnitude[mask] == pytest.approx(target[2])

    # Both scaleless and signed should be 1 / radius everywhere that isn't nan.
    assert self.curvature.scaleless[mask] == pytest.approx(1 / radius)
    assert self.curvature.signed[mask] == pytest.approx(1 / radius)
