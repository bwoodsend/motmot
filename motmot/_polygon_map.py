# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from cslug import ptr
from rockhopper import RaggedArray

from motmot._slug import slug


def make_polygon_map(faces: np.ndarray, id_count: int, use_C=True):
    """Create an array mapping each polygon to its neighbours."""
    # This pivots around the fact that if polygon p0 shares an edge v0->v1 with
    # polygon p1, then p1 contains the edge v1->v0 (hard-coding the assumption
    # that a polygon's vertices are consistently listed in anticlockwise order).
    # So to find a polygon's neighbour on a given edge, we need to find that
    # edge reversed in another polygon.

    n, d = faces.shape

    # Normally, we'd tackle this with an {(v0, v1): polygon_id} dictionary but
    # that is far too slow. Instead, group by v0 into ragged arrays.

    _polygons, _sub_ids = np.divmod(np.arange(faces.size, dtype=np.intp),
                                    faces.shape[1])
    _dest_vertices = faces[_polygons, (1 + _sub_ids) % d]

    polygons, dest_vertices = \
        RaggedArray.groups_by(faces.ravel(), _polygons, _dest_vertices,
                              id_max=id_count)

    # Now ``polygons[v0]`` is a list of polygon ids containing v0.
    # And ``dest_vertices[v0]`` is a list of all the v1s that v0 links to.

    # From here it's just a case of searching ``dest_vertices[v1]`` for v0, then
    # checking in ``polygons[v1]`` for which polygon the v0 corresponds to. This
    # step has been translated into C for speed.

    polygon_map = np.empty(faces.shape, dtype=np.intp, order="C")
    if use_C:
        populate_polygon_map_c(polygon_map, faces, n, d, dest_vertices,
                               polygons)
    else:
        populate_polygon_map_py(polygon_map, faces, n, d, dest_vertices,
                                polygons)

    return polygon_map


def populate_polygon_map_c(polygon_map, faces, n, d, dest_vertices: RaggedArray,
                           polygons: RaggedArray):
    assert faces.dtype == np.intp
    assert faces.flags.c_contiguous
    slug.dll.populate_polygon_map(ptr(polygon_map), ptr(faces), n, d,
                                  dest_vertices._c_struct._ptr,
                                  polygons._c_struct._ptr)


def populate_polygon_map_py(polygon_map, faces, n, d, dest_vertices, polygons):
    """A pure-Python proof-of-concept/test for :meth:`populate_polygon_map_c`.
    """
    for i in range(n):
        for j in range(d):
            v0 = faces[i, j]
            v1 = faces[i, (j + 1) % d]
            for k in range(dest_vertices.starts[v1], dest_vertices.ends[v1]):
                if dest_vertices.flat[k] == v0:
                    # Note that ``polygons.starts`` is ``dest_vertices.starts``
                    # and similarly for ``xxx.ends``, which is why it is ok
                    # to use indices ``k`` for ``dest_vertices`` with
                    # ``polygons`` without any kind of conversion or remapping.
                    polygon_map[i, j] = polygons.flat[k]
                    break
            else:
                polygon_map[i, j] = -1


if __name__ == "__main__":
    faces = np.array([[0, 1, 2, 5], [3, 4, 0, 5], [2, 1, 0, 6]], dtype=np.intp)
    id_count = faces.max() + 1
    target = np.array([[2, 2, -1, 1], [-1, -1, 0, -1], [0, 0, -1, -1]])
    assert np.array_equal(make_polygon_map(faces, id_count, use_C=True), target)
