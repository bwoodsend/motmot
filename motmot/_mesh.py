# -*- coding: utf-8 -*-
"""
"""

import os
from pathlib import Path
import io
import warnings
from typing import Optional
import collections

import numpy as np
import numpy
from stl.mesh import Mesh as _Mesh
from hoatzin import HashTable

from motmot._compat import cached_property
from motmot._misc import idx
from motmot import geometry


def _subsample(name, indices, doc=""):
    """Create an alias property for slices of mesh data."""

    def get(self):
        return self.vectors[indices]

    def set(self, value):
        self.vectors[indices] = value

    get.__name__ = name
    return property(get, set, None, doc)


def _independent(*of_s):

    def wrapped(cached: cached_property):
        for of in of_s:
            assert cached.func is not None
            independencies[of].add(cached.func.__name__)
        return cached

    return wrapped


independencies = collections.defaultdict(set)
independent_of_rotate = _independent("rotate")
independent_of_translate = _independent("translate")
independent_of_transform = _independent("rotate", "translate")


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

        self._bounds = np.empty((2, 3), dtype)

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
            Integer array with shape :py:`(len(mesh), mesh.per_polygon)`.

        """
        if self.is_ids_mesh:
            return self.__ids__
        return self._ids_from_vectors

    @independent_of_transform
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
            An :py:`(number of polygons, mesh.per_polygon, 3)` shaped array.

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

    def __len__(self):
        """Length is defined as the number of polygons."""
        return len(self.__ids__ if self.is_ids_mesh else self.__vectors__)

    @property
    def per_polygon(self) -> int:
        """The number of corners each polygon has."""
        return (self.__ids__ if self.is_ids_mesh else self.__vectors__).shape[1]

    @independent_of_transform
    @cached_property
    def vertex_counts(self) -> np.ndarray:
        """The number of times each vertex id appears in :attr:`ids`.

        Returns:
            1D integer array with the same length as :attr:`vertices`.

        """
        counts = np.zeros(len(self.vertices), np.intc)
        np.add.at(counts, self.ids, 1)
        return counts

    @cached_property
    def max(self) -> np.ndarray:
        """The maximum ``x``, ``y`` and ``z`` value. Shares memory with
        :attr:`bounds`."""
        self._bounds[1] = [i.max() for i in (self.x, self.y, self.z)]
        return self._bounds[1]

    @cached_property
    def min(self) -> np.ndarray:
        """The minimum ``x``, ``y`` and ``z`` value. Shares memory with
        :attr:`bounds`."""
        self._bounds[0] = [i.min() for i in (self.x, self.y, self.z)]
        return self._bounds[0]

    @independent_of_translate
    @cached_property
    def dims(self) -> np.ndarray:
        """The overall length, width and height of the mesh. Or the difference
        between :attr:`max` and :attr:`min`."""
        return self.max - self.min

    @property
    def bounds(self) -> np.ndarray:
        """The minimum and maximum ``(x, y, z)`` values. Equivalent to
        :py:`array([mesh.min, mesh.max])`.

        Returns:
            2D array with shape :py:`(2, 3)`.

        """
        self.min
        self.max
        return self._bounds

    @independent_of_translate
    @cached_property
    def normals(self) -> np.ndarray:
        """Normals to each polygon.

        Returns:
            2D array with shape :py:`(len(mesh), 3)`.

        Normals point outwards provided that the polygons corners are listed in
        counter-clockwise order (which is the usual convention).

        """
        return np.cross(self.v0 - self.v1, self.v1 - self.v2)

    @independent_of_transform
    @cached_property
    def areas(self) -> np.ndarray:
        """Surface area of each polygon.

        Returns:
            1D array with length :py:`len(mesh)`.

        """
        return geometry.magnitude(self.normals) / 2

    @independent_of_transform
    @cached_property
    def area(self) -> float:
        """The total surface area. This is simply the sum of :attr:`areas`."""
        return np.sum(self.areas)

    @independent_of_translate
    @cached_property
    def units(self):
        """Normalised outward :attr:`normals` for each polygon.

        Returns:
            2D array with shape :py:`(len(mesh), 3)`.

        """
        return self.normals / (self.areas[:, np.newaxis] * 2)

    @independent_of_translate
    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """Weighted outward normal for each vertex in :attr:`vertices`.

        Returns:
            2D array with shape :py:`(len(mesh.vertices), 3)`.

        Computed as the normalised average of the surface :attr:`normals` of the
        faces that contain that vertex. Averages are weighted by :attr:`areas`.

        """
        normals = np.zeros_like(self.vertices)
        for ids in self.ids.T:
            np.add.at(normals, ids, self.normals)

        old = np.seterr(invalid="ignore", divide="ignore")
        normals = geometry.normalised(normals)
        np.seterr(**old)

        return normals

    def translate(self, translation):
        """Move this mesh without rotating."""
        # Avoid inplace array modification because it either breaks or loses
        # precision if the dtypes don't match.
        if self.is_ids_mesh:
            self.__vertices__ = self.__vertices__ + translation
        else:
            self.__vectors__ = self.__vectors__ + translation
        reset(self, reset_on_translate)

    def rotate_using_matrix(self, rotation_matrix, point=None):
        """Rotate inplace, the mesh using a **rotation_matrix**.

        Internally this is just a matrix multiplication where the mesh's
        vertices are *post-multiplied* by **rotation_matrix**.

        """
        if point is not None:
            point = np.asarray(point)
            self.translate(-point)
            self.rotate_using_matrix(rotation_matrix)
            self.translate(point)
            return

        # Inplace matrix multiplication (i.e. ``@=``) is not allowed.
        if self.is_ids_mesh:
            self.__vertices__ = self.__vertices__ @ rotation_matrix
        else:
            self.__vectors__ = self.__vectors__ @ rotation_matrix
        reset(self, reset_on_rotate)

    # Shamelessly nick this from numpy-stl.
    rotation_matrix = staticmethod(_Mesh.rotation_matrix)
    rotate = _Mesh.rotate

    def reset(self):
        """Invalidate all cached properties.

        Use after directly writing or setting one of this mesh's array
        attributes.
        """
        reset(self, reset_all)


cached_properties = {
    i.attrname for i in vars(Mesh).values() if isinstance(i, cached_property)
}
reset_on_rotate = cached_properties - independencies["rotate"]
reset_on_translate = cached_properties - independencies["translate"]
reset_all = cached_properties


def reset(self, attrs):
    for i in attrs:
        self.__dict__.pop(i, None)
