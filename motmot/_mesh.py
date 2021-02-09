# -*- coding: utf-8 -*-
"""
"""

import os
from pathlib import Path
import io
import warnings
from typing import Optional

import numpy as np
from stl.mesh import Mesh as _Mesh
from hoatzin import HashTable

from motmot._compat import cached_property
from motmot._misc import idx


def _subsample(name, indices, doc=""):
    """Create an alias property for slices of mesh data."""

    def get(self):
        return self.vectors[indices]

    def set(self, value):
        self.vectors[indices] = value

    get.__name__ = name
    return property(get, set, None, doc)


class Mesh(object):
    __vectors__: np.ndarray
    __ids__: np.ndarray
    __vertices__: np.ndarray
    is_ids_mesh: bool
    """If true, this mesh internally uses :attr:`vertices` and :attr:`ids`.
    Otherwise, it uses :attr:`vectors`."""

    def __init__(self, vertices, ids=None, name=""):
        """A :class:`Mesh` can be constructed in three different ways:

        1. From a filename or file-like binary stream::

            mesh = Mesh("my-mesh.stl")

        2. From a 3D :attr:`vectors` array. It should have shape :py:`(n, d, 3)`
           where ``n`` is the number of polygons and ``d`` is the number of
           vertices per polygon.

        3. From :attr:`vertices` and polygon :attr:`ids`.

        Currently only STL files can be read directly. For other 3D files, read
        using meshio_ then convert to :class:`motmot.Mesh`::

            import meshio
            _mesh = meshio.Mesh.read("my-mesh.ply")
            mesh = Mesh(_mesh.points, _mesh.cells[0].data)

        Beware however that this assumes that the mesh uses a fixed number of
        vertices per polygon. :mod:`motmot` doesn't support mixed polygon types.

        .. _meshio: https://github.com/nschloe/meshio

        """
        if ids is None:
            if isinstance(vertices, (str, os.PathLike)):
                mesh = _Mesh.from_file(vertices, calculate_normals=False)
                self.__vectors__ = mesh.vectors
                self.path = vertices
            elif isinstance(vertices, io.IOBase):
                mesh = _Mesh.from_file(None, fh=vertices,
                                       calculate_normals=False)
                self.__vectors__ = np.ascontiguousarray(mesh.vectors)
            else:
                self.__vectors__ = np.ascontiguousarray(vertices)
            self.is_ids_mesh = False
            dtype = self.__vectors__.dtype
        else:
            self.__vertices__ = np.ascontiguousarray(vertices)
            self.__ids__ = np.asarray(ids, dtype=np.intp, order="C")
            self.is_ids_mesh = True
            dtype = self.__vertices__.dtype
        self.name = name

    _vertex_table: HashTable

    @property
    def path(self) -> Optional[Path]:
        """The filename used to open this mesh. This is set to none if this mesh
        was not read directly from a file.

        Returns:
            Either a :class:`pathlib.Path` filename or :py:`None`.

        """
        return getattr(self, "_path", None)

    @path.setter
    def path(self, path):
        self._path = Path(path) if path else None

    @path.deleter
    def path(self):
        self._path = None

    @cached_property
    def _vertex_table(self) -> HashTable:
        """The brain behind vertex uniquifying and fast vertex lookup."""
        if not self.is_ids_mesh:
            points = self.__vectors__
            length = points.shape[0] * points.shape[1]
        else:
            points = self.__vertices__
            length = len(points)

        # Create a hash table with enough space to fit all points (assuming the
        # worst case where all points are unique) plus 25% padding for speed,
        # and a key dtype of 3 numbers (probably floats).
        table = HashTable(int(1.25 * length), points.dtype * 3)
        # Add all points to the table logging the vertex ID for each point.
        ids = table.add(points)

        if not self.is_ids_mesh:
            # For vectors meshes, set ids attribute.
            self.__ids__ = ids
        elif len(table) < len(self.__vertices__):
            # For ids meshes, `self.ids` should already be set and be identical
            # to the `ids` found above unless there were duplicates in
            # `self.vertices`. These can cause some algorithms to break.
            warnings.warn("Duplicate vertices in mesh.vertices.")

        return table

    @property
    def vertices(self) -> np.ndarray:
        """All points in the mesh with duplicity removed.

        If this mesh is not originally an IDs mesh, i.e. :attr:`vertices` had to
        be calculated from :attr:`vectors`, then this array is read-only.

        """
        if self.is_ids_mesh:
            return self.__vertices__
        return self._vertex_table.unique

    @property
    def ids(self) -> np.ndarray:
        """Indices of vertices used to construct each polygon.

        Returns:
            Integer array with shape :py:`(len(mesh), vertices per polygon)`.

        """
        if self.is_ids_mesh:
            return self.__ids__
        return self._ids_from_vectors

    @cached_property
    def _ids_from_vectors(self):
        self._vertex_table
        return self.__ids__

    @cached_property
    def _vectors_from_ids(self):
        assert self.is_ids_mesh
        return self.__vertices__[self.__ids__]

    @property
    def vectors(self) -> np.ndarray:
        """The :py:`(x, y, z)` coordinates of each corner of each polygon.

        Returns:
            An :py:`(number of polygons, vertices per polygon, 3)` shaped array.

        """
        if self.is_ids_mesh:
            return self._vectors_from_ids
        return self.__vectors__

    x = _subsample(
        "x", idx[:, :, 0], "The x coordinate of each vertex of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, :, 0]`.")
    y = _subsample(
        "y", idx[:, :, 1], "The y coordinate of each vertex of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, :, 1]`.")
    z = _subsample(
        "z", idx[:, :, 2], "The z coordinate of each vertex of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, :, 2]`.")
    v0 = _subsample(
        "v0", idx[:, 0], "The 1\\ :superscript:`st` corner of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, 0]`.")
    v1 = _subsample(
        "v1", idx[:, 1], "The 2\\ :superscript:`nd` corner of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, 1]`.")
    v2 = _subsample(
        "v2", idx[:, 2], "The 3\\ :superscript:`rd` corner of each polygon. "
        "Equivalent to :py:`mesh.vectors[:, 2]`.")
