# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import hoatzin

from motmot import Mesh


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
