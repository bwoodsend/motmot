# -*- coding: utf-8 -*-
"""
"""

import os
from pathlib import Path
import io
import warnings
from typing import Optional
import collections
import copy

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

    def __repr__(self):
        if not self.is_ids_mesh:
            return f"<Vectors {type(self).__name__} at {hex(id(self))} | " \
                   f"{len(self)} {self.vectors.shape[1]}-sided polygons>"
        return f"<IDs {type(self).__name__} at {hex(id(self))} | " \
               f"{len(self.vertices)} vertices | " \
               f"{len(self)} {self.ids.shape[1]}-sided polygons>"

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

        Note that for non co-planer (not flat) polygons, this definition becomes
        progressively more arbitrary as the polygons become more complicated.
        Different areas can be obtained for identical polygons simply by
        *rolling* a polygon's corners so that a different corner is listed
        first.

        """
        if self.per_polygon == 3:
            # Triangles are much simpler than other shapes and can therefore
            # take the simpler/faster approach.
            return geometry.magnitude(self.normals) / 2
        return geometry.area(self.vectors)

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
        old = np.seterr(invalid="ignore", divide="ignore")
        if self.per_polygon == 3:
            # Again, this is marginally faster but only works for triangular
            # meshes.
            out = self.normals / (self.areas[:, np.newaxis] * 2)
        else:
            out = geometry.normalised(self.normals)
        np.seterr(**old)
        return out

    @independent_of_translate
    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """Weighted outward normal for each vertex in :attr:`vertices`.

        Returns:
            2D array with shape :py:`(len(mesh.vertices), 3)`.

        Computed as the normalised average of the surface :attr:`normals` of the
        faces that contain that vertex. Averages are weighted by :attr:`areas`.

        """
        old = np.seterr(invalid="ignore", divide="ignore")
        normals = geometry.normalised(self._vertex_normals)
        np.seterr(**old)
        return normals

    @property
    def _vertex_normals(self):
        """Raw un-normalised vertex normals."""
        normals = np.zeros_like(self.vertices)
        for ids in self.ids.T:
            np.add.at(normals, ids, self.normals)
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

    def __getitem__(self, item):
        """

        :rtype: Mesh
        """
        if self.is_ids_mesh:
            return type(self)(self.vertices, self.ids[item], name=self.name)
        return type(self)(self.vectors[item], name=self.name)

    @staticmethod
    def __array__():
        raise TypeError("Meshes can't be converted directly to arrays. Use one "
                        "of `vertices`, `vectors` or `ids` attributes instead.")

    def crop(self, mask, in_place=False):
        """Return a subsample of the original mesh. Inclusion is defined by
        **mask**.

        Args:
            mask (numpy.ndarray or slice):
                Polygons to include.
            in_place (bool):
                Modify this mesh instead of making a modified copy, defaults to
                False.

        Returns:
            Mesh:
                This mesh if **in_place** or a new cropped one.

        A minimal usage example:

        .. code-block:: python

            # Get only polygons with non-negative average Z-values.
            cropped = mesh.crop(mesh.centers[:, 2] >= 0)

        For an IDs based mesh this samples :attr:`ids` and leaves
        :attr:`vertices` untouched without copying and is equivalent to::

            cropped = Mesh(mesh.vertices, mesh.ids[mask], name=mesh.name)

        For a vectors based mesh this function simply samples :attr:`vectors`::

            cropped = Mesh(mesh.vectors[mask], name=mesh.name)

        Please ensure you are aware of `when indexing copies in numpy`_ if you
        intend to modify either the cropped or the original mesh afterwards.

        .. _`when indexing copies in numpy`: https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing

        When **inplace** is false (default), cropping can also be achieved by
        indexing the mesh directly::

            cropped = mesh[mask]

        """
        if not in_place:
            return self[mask]
        if self.is_ids_mesh:
            self.__ids__ = self.__ids__[mask]
        else:
            self.__vectors__ = self.__vectors__[mask]
        self.reset()
        return self

    def __copy__(self):
        cls = type(self)
        out = cls.__new__(cls)
        out.__setstate__(self.__getstate__())
        return out

    def __deepcopy__(self, memo):
        cls = type(self)
        out = cls.__new__(cls)
        state = self.__getstate__()
        state = {key: copy.deepcopy(val, memo) for (key, val) in state.items()}
        out.__setstate__(state)
        return out

    def copy(self, deep=True):
        """Make a shallow or deep copy of the mesh.

        Args:
            deep (bool):
                If true, copy the underlying :attr:`vectors` or :attr:`vertices`
                and :attr:`ids` arrays. Otherwise output will share these arrays
                with this mesh.

        Returns:
            Mesh: Another mesh.

        Caches of cached properties are never copied.

        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __getstate__(self):
        if self.is_ids_mesh:
            return {
                "vertices": self.__vertices__,
                "ids": self.__ids__,
                "name": self.name,
                "path": self.path,
            }
        return {"vectors": self.vectors, "name": self.name, "path": self.path}

    def __setstate__(self, dic):
        self.__init__(dic.get("vectors", dic.get("vertices")), dic.get("ids"),
                      dic["name"])
        self.path = dic["path"]

    @cached_property
    def centers(self) -> np.ndarray:
        """The center of each polygon, defined as the mean of each polygon's
        corners.

        Returns:
            2D array with shape :py:`(len(mesh), 3)`.

        """
        # This could just be `self.vectors.mean(axis=1)` but
        # numpy.ufunc.reduce() with axis != None is much slower than pure
        # Python.
        out = self.vectors[:, 0]
        for i in range(1, self.vectors.shape[1]):
            out = out + self.vectors[:, i]
        return out / self.vectors.shape[1]

    @independent_of_transform
    @cached_property
    def polygon_map(self) -> np.ndarray:
        """Maps each polygon to its adjacent (shares a common edge) polygons.

        The format is an numpy int array with the same shape as
        :attr:`Mesh.ids`. A polygon is referenced by its position in
        :attr:`Mesh.ids` (or :attr:`Mesh.vectors`).

        For example, assume a triangular mesh is called  ``mesh``. And suppose
        :py:`mesh.polygon_map[n]` is :py:`[i, j, k]`, then:

        - The edge going from vertex *0* to vertex *1* of the triangle
          :py:`mesh.vectors[n]` would be shared with the triangle
          :py:`mesh.vectors[i]`,
        - The edge going from vertex *1* to vertex *2* of the triangle
          :py:`mesh.vectors[n]` would be shared with the triangle
          :py:`mesh.vectors[j]`,
        - And the edge going from vertex *2* to vertex **0** of the triangle
          :py:`mesh.vectors[n]` would be shared with the triangle
          :py:`mesh.vectors[k]`,

        Any polygons which are missing a neighbour on a particular edge (i.e.
        on the boundary of a non-closed mesh) use :py:`-1` as a placeholder.

        """
        from motmot._polygon_map import make_polygon_map
        return make_polygon_map(self.ids, len(self.vertices))

    def connected_polygons(self, initial, mask=None, polygon_map=None):
        """Recursively walk connected polygons.

        Finds the whole connected region containing the triangle `arg` where
        `whole connected region` here means that all triangles within that region
        are joined indirectly by at least one triangle.

        Args:
            initial (int or numpy.ndarray):
                The polygon id(s) to start at.
            mask (numpy.ndarray):
                Only include regions covered by `mask` if specified.
            polygon_map (numpy.ndarray):
                An alternative polygon map to use. Defaults to
                :attr:`polygon_map`.
        Returns:
            numpy.ndarray:
                A connected region as a 1D bool array.

        This is implemented by navigating the :attr:`polygon_map`.
        Restrictions on connectivity can be given either with the **mask**
        parameter, which mimics removing polygons, or by overriding
        **polygon_map** and replacing values with ``-1`` to block edges between
        polygons. Note that the **mask** only prevents the algorithm from
        traversing **onto** unmasked polygons. If a polygon is in **initial**
        then it will still be included in the output.

        """
        from motmot._polygon_walk import connected
        polygon_map = self.polygon_map if polygon_map is None else polygon_map
        return connected(polygon_map, initial, mask)

    def group_connected_polygons(self, mask=None, polygon_map=None):
        """Group and enumerate all polygons which are indirectly connected.

        Returns:
            (numpy.ndarray, int):
                A :py:`(group_ids, group_count)` pair.

        Functionally, this is equivalent to calling :meth:`connected_polygons`
        repeatedly until every polygon is assigned a group.

        To convert the output to a list of meshes use::

            from rockhopper import RaggedArray
            ragged = RaggedArray.group_by(mesh.vectors, *mesh.group_connected_polygons())
            sub_meshes = [Mesh(i) for i in ragged]

        Or if your using vertices/ids meshes::

            from rockhopper import RaggedArray
            ragged = RaggedArray.group_by(mesh.ids, *mesh.group_connected_polygons())
            sub_meshes = [Mesh(mesh.vertices, ids) for ids in ragged]

        Note that both will work irregardless of :attr:`is_ids_mesh`, however
        the mismatched implementation will be slower.

        """
        from motmot._polygon_walk import group_connected
        polygon_map = self.polygon_map if polygon_map is None else polygon_map
        return group_connected(polygon_map, mask)


cached_properties = {
    i.attrname for i in vars(Mesh).values() if isinstance(i, cached_property)
}
reset_on_rotate = cached_properties - independencies["rotate"]
reset_on_translate = cached_properties - independencies["translate"]
reset_all = cached_properties


def reset(self, attrs):
    for i in attrs:
        self.__dict__.pop(i, None)
