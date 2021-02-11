# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from motmot import Mesh


def ids_mesh(n, d=3):
    vertices = np.arange((n * 5 // 3) * 3).reshape((-1, 3))
    ids = np.append(np.arange(len(vertices)),
                    np.random.randint(0, len(vertices), n * d - len(vertices)))
    return Mesh(vertices, ids.reshape((n, d)))


def vectors_mesh(n, d=3):
    _mesh = ids_mesh(n, d)
    return Mesh(_mesh.vectors)
