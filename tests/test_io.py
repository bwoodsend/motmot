import io
from pathlib import Path
import os

import numpy as np
import pytest
from stl.mesh import Mesh as numpy_Mesh

from motmot import Mesh
from tests import data

numpy_rabbit = numpy_Mesh.from_file(str(data.rabbit_path))

INPUTS = [
    data.rabbit_path,
    str(data.rabbit_path),
    numpy_rabbit.vectors,
    data.rabbit_xz,
]


@pytest.mark.parametrize("path", INPUTS, ids=repr)
def test_read(path):

    self = Mesh(path)
    assert self.is_faces_mesh is False
    assert np.array_equal(self.vectors, numpy_rabbit.vectors)

    if isinstance(path, (io.BytesIO, np.ndarray)):
        assert self.path is None
    else:
        assert isinstance(self.path, Path)
        del self.path
        assert self.path is None

    assert self.vectors.flags.contiguous


def test_read_bytes_io():
    test_read(io.BytesIO(data.rabbit_path.read_bytes()))


@pytest.mark.parametrize(
    "name", ["foo.stl", "foo.stl.gz", "foo.stl.bz2", "foo.stl.xz"], ids=repr)
def test_write(name):
    vectors = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
    self = Mesh(vectors, name="bob")
    assert self.name == "bob"

    path = data.DUMP_DIR / name
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    self.save(path)

    if name == "foo.stl":
        assert b"bob     " == path.read_bytes()[:8]

    mesh = Mesh(path)
    assert np.all(mesh.vectors == vectors)
    assert mesh.name == b"bob"

    os.remove(path)
    self.name = None
    self.save(path)
    assert Mesh(path).name == b""


@pytest.mark.filterwarnings("error")
def test_write_invalid():
    """Verify that numpy-stl's mesh cleanups are all off and that no numpy
    warnings are emitted."""
    vectors = np.array([
        [[np.nan] * 3] * 3,
        [[0] * 3] * 3,
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    ], dtype=np.float32)
    self = Mesh(vectors)
    file = io.BytesIO()
    self.save(file)
    out = Mesh(io.BytesIO(file.getvalue()))
    assert out.vectors.tobytes() == self.vectors.tobytes()
