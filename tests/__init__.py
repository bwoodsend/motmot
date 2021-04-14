# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import hoatzin

from motmot import Mesh


def assert_mesh_equal(a: Mesh, b: Mesh, ignore_path=False):
    """Check two meshes are equal.

     This ignores all lazy attributes including :attr:`Mesh.normals` as they are
     tested elsewhere.

    """
    import numpy as np
    assert np.all(a.vectors == b.vectors)
    assert np.all(a.ids == b.ids)
    assert np.all(a.vertices == b.vertices)
    assert a.name == b.name
    if not ignore_path:
        assert a.path == b.path
    assert a.is_ids_mesh == b.is_ids_mesh


def unique_vertices(n):
    n = int(n)
    table = hoatzin.HashTable(n * 3 // 2, np.dtype(None) * 3)
    while len(table) < n:
        table.add(np.random.random((n - len(table), 3)))
    return table.destroy()


def ids_mesh(n, d=3):
    vertices = unique_vertices(n * 3 // 2)
    ids = np.append(np.arange(len(vertices)),
                    np.random.randint(0, len(vertices), n * d - len(vertices)))
    return Mesh(vertices, ids.reshape((n, d)))


def vectors_mesh(n, d=3):
    _mesh = ids_mesh(n, d)
    return Mesh(_mesh.vectors)


def closed_mesh(*spam):
    from tests.data import rabbit_path
    return Mesh(rabbit_path)


def cylinder(n):
    t = np.arange(n) * (2 * np.pi / n)
    vertices = np.empty((n * 2, 3))
    vertices[:n, 2] = 0
    vertices[n:, 2] = 10

    vertices[:n, 0] = np.cos(t, out=vertices[n:, 0])
    vertices[:n, 1] = np.sin(t, out=vertices[n:, 1])

    ids = np.empty((n, 4), dtype=int)
    ids[:, 0] = np.arange(0, n)
    ids[:, 1] = np.arange(n, 2 * n)
    ids[:, 2] = np.roll(ids[:, 1], 1)
    ids[:, 3] = np.roll(ids[:, 0], 1)

    return Mesh(vertices, ids)


def square_grid(n):
    """Create a flat plane of n x n squares."""
    from motmot.geometry import zip
    r = np.mgrid
    top_left = zip(r[:n], r[:n][:, np.newaxis], 0).reshape((-1, 3))
    vectors = top_left[:, np.newaxis] + \
              np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    return Mesh(vectors * .01)
