# -*- coding: utf-8 -*-
"""
"""

import numpy as np

from motmot._compat import cached_property
from motmot.geometry import inner_product, magnitude
from motmot._mesh import Mesh


class Curvature(object):
    """A bucket for everything curvature related.

    Unlike the more traditional definition of curvature, this is defined per
    polygon edge rather than per vertex. Edges are defined as the shared two
    vertices between a polygon and a neighbour from :attr:`Mesh.polygon_map`.

    Curvature can have several similar forms. In most cases you will want
    :attr:`scaleless` or :attr:`signed` for an orientation and resolution
    independent measurement. :attr:`directional` combined with
    :func:`geometry.inner_product` can be useful for searching for very specific
    shapes with a predefined orientation.

    """
    _mesh: Mesh

    def __init__(self, mesh):
        self._mesh = mesh

    @cached_property
    def directional(self) -> np.ndarray:
        """Curvature's rawest form - the cross product of each triangle's unit
        normal with each of its adjacent triangles' unit normals.

        Returns:
            A :py:`(len(mesh), mesh.per_polygon, 3)` numpy array.

        This form gives a vector rather than a scalar value for each edge.
        The magnitude of this vector is the ``sin()`` of the angle between the
        two polygons and the direction is tangential to the edge between them.

        """
        out = np.cross(self._mesh.units[self._mesh.polygon_map],
                       self._mesh.units[:, np.newaxis])

        return np.where(self._mesh.polygon_map[:, :, np.newaxis] != -1, out,
                        np.array([[[np.nan, np.nan, np.nan]]], dtype=out.dtype))

    @cached_property
    def magnitude(self) -> np.ndarray:
        """Curvature magnitudes about polygons' edges.

        Returns:
            A :py:`(len(mesh), mesh.per_polygon)` numpy array.

        Given by taking magnitudes of :attr:`directional`.

        """
        return magnitude(self.directional)

    @cached_property
    def scaleless(self) -> np.ndarray:
        r"""The reciprocal of the radius of curvature. Or in pig's English, a
        sphere with radius ``10`` will have scaleless curvature ``1 / 10 = 0.1``
        throughout.

        Returns:
            A :py:`(len(mesh), mesh.per_polygon)` numpy array.

        This property is independent of mesh resolution. Given by
        :attr:`magnitude` / :attr:`magnitude(displacements)<Mesh.displacements>`.

        """
        old = np.seterr(divide="ignore", invalid="ignore")
        out = self.magnitude / magnitude(self._mesh.displacements)
        np.seterr(**old)
        return out

    @cached_property
    def signed(self) -> np.ndarray:
        """Signed :attr:`scaleless` curvature magnitudes.

        Like :attr:`scaleless` but signed so that bumps or bulges have positive
        values and slots or grooves or creases have negative values.

        """
        signs = np.sign(
            inner_product(
                self._mesh.units[:, np.newaxis],
                self._mesh.displacements,
            )) * -1
        return self.scaleless * signs
