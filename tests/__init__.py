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
