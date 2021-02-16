# -*- coding: utf-8 -*-
"""
"""

import io
from pathlib import Path

import numpy as np
import pytest
from stl.mesh import Mesh as numpy_Mesh

from motmot import Mesh
from tests import data

pytestmark = pytest.mark.order(1)

numpy_rabbit = numpy_Mesh.from_file(str(data.rabbit_path))

INPUTS = [
    data.rabbit_path,
    str(data.rabbit_path),
    io.BytesIO(data.rabbit_path.read_bytes()), numpy_rabbit.vectors
]


@pytest.mark.parametrize("path", INPUTS)
def test_read(path):

    self = Mesh(path)
    assert self.is_ids_mesh is False
    assert np.array_equal(self.vectors, numpy_rabbit.vectors)

    if isinstance(path, (io.BytesIO, np.ndarray)):
        assert self.path is None
    else:
        assert isinstance(self.path, Path)
        del self.path
        assert self.path is None