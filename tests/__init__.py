from pathlib import Path
import json

import numpy as np
import hirola

from motmot import Mesh


def assert_mesh_equal(a: Mesh, b: Mesh, ignore_path=False):
    """Check two meshes are equal.

     This ignores all lazy attributes including :attr:`Mesh.normals` as they are
     tested elsewhere.

    """
    import numpy as np
    assert np.all(a.vectors == b.vectors)
    assert np.all(a.faces == b.faces)
    assert np.all(a.vertices == b.vertices)
    assert a.name == b.name
    if not ignore_path:
        assert a.path == b.path
    assert a.is_faces_mesh == b.is_faces_mesh


def unique_vertices(n):
    n = int(n)
    table = hirola.HashTable(n * 3 // 2, np.dtype(None) * 3)
    while len(table) < n:
        table.add(np.random.random((n - len(table), 3)))
    return table.destroy()


def faces_mesh(n, d=3):
    np.random.seed(0)
    vertices = unique_vertices(n * 3 // 2)
    faces = np.append(
        np.arange(len(vertices)),
        np.random.randint(0, len(vertices), n * d - len(vertices)))
    return Mesh(vertices, faces.reshape((n, d)))


def vectors_mesh(n, d=3):
    _mesh = faces_mesh(n, d)
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

    faces = np.empty((n, 4), dtype=int)
    faces[:, 0] = np.arange(0, n)
    faces[:, 1] = np.arange(n, 2 * n)
    faces[:, 2] = np.roll(faces[:, 1], 1)
    faces[:, 3] = np.roll(faces[:, 0], 1)

    return Mesh(vertices, faces)


def square_grid(n):
    """Create a flat plane of n x n squares."""
    from motmot.geometry import zip
    r = np.mgrid
    top_left = zip(r[:n], r[:n][:, np.newaxis], 0).reshape((-1, 3))
    vectors = top_left[:, np.newaxis] + \
              np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    return Mesh(vectors * .01)


def _icosasphere(n):
    b = Path(__file__).with_name(f"icosasphere-{n}.json").read_bytes()
    return [np.array(i) for i in json.loads(b)]


icosasphere_1 = _icosasphere(1)
icosasphere_10 = _icosasphere(10)
icosasphere_15 = _icosasphere(15)
