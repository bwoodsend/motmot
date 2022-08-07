# -*- coding: utf-8 -*-
"""
"""

import os
from pathlib import Path
import io
import warnings
from typing import Optional, Union
import copy
from numbers import Integral
import operator

import numpy as np
import numpy
from stl.mesh import Mesh as _Mesh
from hirola import HashTable
from rockhopper import RaggedArray

from motmot._compat import cached_property
from motmot._misc import idx, Independency, as_nD, open_
from motmot import geometry


def _subsample(name, indices, doc=""):
    """Create an alias property for slices of mesh data."""

    def get(self):
        return self.vectors[indices]

    def set(self, value):
        self.vectors[indices] = value

    get.__name__ = name
    return property(get, set, None, doc)


independent = Independency()


class Mesh(object):
    __vectors__: np.ndarray
    __faces__: np.ndarray
    __vertices__: np.ndarray
    is_faces_mesh: bool = False
    """If true, this mesh internally uses :attr:`vertices` and :attr:`faces`.
    Otherwise, it uses :attr:`vectors`."""

    def __init__(self, vertices, faces=None, name=""):
        """A :class:`Mesh` can be constructed in three different ways:

        1. From a filename or file-like binary stream::

            mesh = Mesh("my-mesh.stl")

        2. From a 3D :attr:`vectors` array. It should have shape :py:`(n, d, 3)`
           where ``n`` is the number of polygons and ``d`` is the number of
           vertices per polygon.

        3. From :attr:`vertices` and polygon :attr:`faces`.

        Currently only STL files can be read directly. For other 3D files, read
        using meshio_ then convert to :class:`motmot.Mesh`::

            import meshio
            _mesh = meshio.Mesh.read("my-mesh.ply")
            mesh = Mesh(_mesh.points, _mesh.cells[0].data)

        Beware however that this assumes that the mesh uses a fixed number of
        vertices per polygon. :mod:`motmot` doesn't support mixed polygon types.

        .. _meshio: https://github.com/nschloe/meshio

        """
        self.name = ""

        if faces is None:
            if not isinstance(vertices, (list, np.ndarray)):
                with open_(vertices, "rb") as f:
                    # numpy-stl's detection for streams has some holes in it.
                    # For some reason, just passing the open compressed file to
                    # numpy-stl causes it to only read some of it. Create a
                    # redundant intermediate io.BytesIO(). Note that even ASCII
                    # STLs must be read in binary mode.
                    fh = io.BytesIO(f.read())

                mesh = _Mesh.from_file(None, fh=fh, calculate_normals=False)
                self.vectors = mesh.vectors
                if isinstance(vertices, (str, os.PathLike)):
                    self.path = vertices
                self.name = mesh.name
            else:
                self.vectors = vertices
        else:
            self.is_faces_mesh = True
            self.vertices = vertices
            self.faces = faces
        self.name = name or self.name

        self._bounds = np.empty((2, 3), self.dtype)

    _vertex_table: HashTable

    def __repr__(self):
        if not self.is_faces_mesh:
            return f"<Vectors {type(self).__name__} at {hex(id(self))} | " \
                   f"{len(self)} {self.vectors.shape[1]}-sided polygons>"
        return f"<Faces {type(self).__name__} at {hex(id(self))} | " \
               f"{len(self.vertices)} vertices | " \
               f"{len(self)} {self.faces.shape[1]}-sided polygons>"

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
    def vertex_table(self) -> HashTable:
        """The lookup table behind vertex uniquifying and fast vertex lookup.

        This object, a :class:`hirola.HashTable`, is similar to a :class:`dict`
        with this mesh's unique vertices as its keys and an enumeration as its
        values.

        To get a vertex ID (or IDs) for a given point(s) use::

            ids = mesh.vertex_table[points]

        This is the reciprocal of::

            points = mesh.vertices[ids]

        To quickly test if a vertex or vertices is in :attr:`vertices` use
        :meth:`mesh.vertex_table.contains() <hirola.HashTable.contains>`.

        """
        if not self.is_faces_mesh:
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
        faces = table.add(points)

        if not self.is_faces_mesh:
            # For vectors meshes, set faces attribute.
            self.__faces__ = faces
        elif len(table) < len(self.__vertices__):
            # For faces meshes, `self.faces` should already be set and be
            # identical to the `faces` found above unless there were duplicates
            # in `self.vertices`. These can cause some algorithms to break.
            warnings.warn("Duplicate vertices in mesh.vertices.")

        return table

    @property
    def dtype(self):
        """The :class:`numpy.dtype` of :attr:`vertices` and :attr:`vectors`.
        """
        if self.is_faces_mesh:
            return self.__vertices__.dtype
        return self.__vectors__.dtype

    @property
    def vertices(self) -> np.ndarray:
        """All points in the mesh with duplicity removed.

        If this mesh is not originally an faces mesh, i.e. :attr:`vertices` had
        to be calculated from :attr:`vectors`, then this array is read-only.

        """
        if self.is_faces_mesh:
            return self.__vertices__
        return self.vertex_table.keys

    @vertices.setter
    def vertices(self, x):
        if not self.is_faces_mesh:
            raise ValueError("A vectors mesh's vertices are readonly. "
                             "Write to mesh.vectors instead.")
        # Vertices must be C contiguous for hirola.HashTable.
        x = np.ascontiguousarray(x)

        # If given an empty array:
        if x.size == 0:
            # Make it valid by default.
            x = np.empty((0, 3), x.dtype)

        elif x.shape[-1] != 3:
            raise ValueError("The last axis of vertices must be of length 3. "
                             f"Received an array with shape {x.shape}.")

        # Implicitly promote a single vertex [x, y, z] to [[x, y, z]]
        self.__vertices__ = as_nD(x, 2, "vertices")
        self.reset()

    @property
    def faces(self) -> np.ndarray:
        """Indices of vertices used to construct each polygon.

        Returns:
            Integer array with shape :py:`(len(mesh), mesh.per_polygon)`.

        """
        if self.is_faces_mesh:
            return self.__faces__
        return self._faces_from_vectors

    @faces.setter
    def faces(self, x):
        if not self.is_faces_mesh:
            raise ValueError("A vectors mesh's faces are readonly. "
                             "Write to mesh.vectors instead.")

        # faces must be C contiguous and of a fixed dtype for the various C
        # functions to work.
        x = np.asarray(x, dtype=np.intp, order="C")

        # Set empty input `[]` to something sane. Namely 0 triangles.
        if x.size == 0 and x.ndim == 1:
            x = np.empty((0, 3), np.intp)

        self.__faces__ = as_nD(x, 2, "faces")
        self.reset()

    @independent.of("translate", "rotate")
    @cached_property
    def _faces_from_vectors(self):
        self.vertex_table
        return self.__faces__

    @cached_property
    def _vectors_from_faces(self):
        assert self.is_faces_mesh
        return self.__vertices__[self.__faces__]

    @property
    def vectors(self) -> np.ndarray:
        """The :py:`(x, y, z)` coordinates of each corner of each polygon.

        Returns:
            An :py:`(number of polygons, mesh.per_polygon, 3)` shaped array.

        """
        if self.is_faces_mesh:
            return self._vectors_from_faces
        return self.__vectors__

    @vectors.setter
    def vectors(self, x):
        if self.is_faces_mesh:
            raise ValueError("A faces mesh's vectors are readonly. "
                             "Write to mesh.vertices instead.")
        x = np.ascontiguousarray(x)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(
                "Vectors must be a 3D array with last axis of length 3. "
                f"Received an array with shape {x.shape}.")
        self.__vectors__ = x
        self.reset()
        self._bounds = np.empty((2, 3), self.__vectors__.dtype)

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
        return len(self.__faces__ if self.is_faces_mesh else self.__vectors__)

    @property
    def per_polygon(self) -> int:
        """The number of corners each polygon has."""
        return (self.__faces__
                if self.is_faces_mesh else self.__vectors__).shape[1]

    @independent.of("translate", "rotate")
    @cached_property
    def vertex_counts(self) -> np.ndarray:
        """The number of times each vertex id appears in :attr:`faces`.

        Returns:
            1D integer array with the same length as :attr:`vertices`.

        """
        counts = np.zeros(len(self.vertices), np.intc)
        np.add.at(counts, self.faces, 1)
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

    @independent.of("translate")
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

    @independent.of("translate")
    @cached_property
    def normals(self) -> np.ndarray:
        """Normals to each polygon.

        Returns:
            2D array with shape :py:`(len(mesh), 3)`.

        Normals point outwards provided that the polygons corners are listed in
        counter-clockwise order (which is the usual convention).

        """
        return np.cross(self.v0 - self.v1, self.v1 - self.v2)

    @independent.of("transform")
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

    @independent.of("translate", "rotate")
    @cached_property
    def area(self) -> float:
        """The total surface area. This is simply the sum of :attr:`areas`."""
        return np.sum(self.areas)

    @independent.of("translate")
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

    @independent.of("translate")
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
        for faces in self.faces.T:
            np.add.at(normals, faces, self.normals)
        return normals

    def translate(self, translation):
        """Move this mesh without rotating."""
        # Avoid inplace array modification because it either breaks or loses
        # precision if the dtypes don't match.
        if self.is_faces_mesh:
            self.__vertices__ = self.__vertices__ + translation
        else:
            self.__vectors__ = self.__vectors__ + translation

        # noinspection PyUnresolvedReferences
        self._reset_on_translate()

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
        if self.is_faces_mesh:
            self.__vertices__ = self.__vertices__ @ rotation_matrix
        else:
            self.__vectors__ = self.__vectors__ @ rotation_matrix

        # noinspection PyUnresolvedReferences
        self._reset_on_rotate()

    # Shamelessly nick this from numpy-stl.
    rotation_matrix = staticmethod(_Mesh.rotation_matrix)
    rotate = _Mesh.rotate

    def reset(self):
        """Invalidate all cached properties.

        Use after directly writing or setting one of this mesh's array
        attributes.
        """
        # noinspection PyUnresolvedReferences
        self._reset_all(self)

    def __getitem__(self, item) -> 'Mesh':
        if self.is_faces_mesh:
            return type(self)(self.vertices, self.faces[item], name=self.name)
        return type(self)(self.vectors[item], name=self.name)

    @staticmethod
    def __array__():
        raise TypeError(
            "Meshes can't be converted directly to arrays. Use one "
            "of `vertices`, `vectors` or `faces` attributes instead.")

    def crop(self, mask: Union[np.ndarray, slice],
             in_place: bool = False) -> 'Mesh':
        """Return a subsample of the original mesh. Inclusion is defined by
        **mask**.

        Args:
            mask:
                Polygons to include.
            in_place:
                Modify this mesh instead of making a modified copy, defaults to
                False.
        Returns:
            This mesh if **in_place** or a new cropped one.

        A minimal usage example:

        .. code-block:: python

            # Get only polygons with non-negative average Z-values.
            cropped = mesh.crop(mesh.centers[:, 2] >= 0)

        For a faces mesh this samples :attr:`faces` and leaves
        :attr:`vertices` untouched without copying and is equivalent to::

            cropped = Mesh(mesh.vertices, mesh.faces[mask], name=mesh.name)

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
        if self.is_faces_mesh:
            self.__faces__ = self.__faces__[mask]
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

    def copy(self, deep=True) -> 'Mesh':
        """Make a shallow or deep copy of the mesh.

        Args:
            deep:
                If true, copy the underlying :attr:`vectors` or :attr:`vertices`
                and :attr:`faces` arrays. Otherwise output will share these
                arrays with this mesh.
        Returns:
            Another mesh.

        Caches of cached properties are never copied.

        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __getstate__(self):
        if self.is_faces_mesh:
            return {
                "vertices": self.__vertices__,
                "faces": self.__faces__,
                "name": self.name,
                "path": self.path,
            }
        return {"vectors": self.vectors, "name": self.name, "path": self.path}

    def __setstate__(self, dic):
        self.__init__(dic.get("vectors", dic.get("vertices")), dic.get("faces"),
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

    @independent.of("translate", "rotate")
    @cached_property
    def polygon_map(self) -> np.ndarray:
        """Maps each polygon to its adjacent (shares a common edge) polygons.

        The format is an numpy int array with the same shape as
        :attr:`Mesh.faces`. A polygon is referenced by its position in
        :attr:`Mesh.faces` (or :attr:`Mesh.vectors`).

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
        return make_polygon_map(self.faces, len(self.vertices))

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

        Or if your using vertices/faces meshes::

            from rockhopper import RaggedArray
            ragged = RaggedArray.group_by(mesh.faces, *mesh.group_connected_polygons())
            sub_meshes = [Mesh(mesh.vertices, faces) for faces in ragged]

        Note that both will work irregardless of :attr:`is_faces_mesh`, however
        the mismatched implementation will be slower.

        """
        from motmot._polygon_walk import group_connected
        polygon_map = self.polygon_map if polygon_map is None else polygon_map
        return group_connected(polygon_map, mask)

    @cached_property
    def displacements(self) -> np.ndarray:
        """The displacement from each polygon's center to each of its
        neighbours' centers.

        Returns:
            A :py:`(len(mesh), mesh.per_polygon, 3)` numpy array.

        Defaults to ``nan`` when a neighbour is missing.

        """
        return np.where(
            (self.polygon_map != -1)[:, :, np.newaxis],
            self.centers[self.polygon_map] - self.centers[:, np.newaxis],
            np.array([[[np.nan, np.nan, np.nan]]], self.centers.dtype),
        )

    @cached_property
    def curvature(self):
        """Everything curvature related. See :class:`motmot.Curvature`.
        Different *flavours* of curvature are accessible via different
        sub-attributes of this property such as
        :attr:`mesh.curvature.scaleless <Curvature.scaleless>`.
        """
        from motmot._curvatures import Curvature
        return Curvature(self)

    @independent.of("rotate", "translate")
    @cached_property
    def vertex_map(self) -> RaggedArray:
        """A mapping from each vertex id to every other vertex that it is
        directly connected to.

        Each row in this :class:`~rockhopper.RaggedArray` lists all the
        neighbours of one vertex from :attr:`vertices`.
        e.g. If :py:`mesh.vertex_map[10]` is :py:`[13, 17, 19, 22]` then that
        would imply that :py:`mesh.vertices[10]` is connected to each of
        :py:`mesh.vertices[[13, 17, 19, 22]]` by a single polygon edge.

        This mapping is guaranteed to:

        * Be symmetric. If vertex **A** is connected to vertex **B** then **B**
          is connected to **A**.
        * Contain no self references. **A** will never be listed as connected to
          **A**.
        * Contains no duplicates. **A** will never be listed as connected to
          **B** twice.

        The order in which neighbours appear is arbitrary and not guaranteed to
        be consistent across :mod:`motmot` versions.

        .. seealso::

            :meth:`connected_vertices` if you prefer to work directly with
            vertices rather than vertex faces.

        """
        # In a closed mesh each edge is written once going from A to B and once
        # going from B to A.
        # In a non closed mesh, edges on boundaries will only be listed one way.
        # This function must make sure to mirror the *singular* edges on
        # boundaries (obviously without accidentally duplicating any others).

        # Work out which (if any) edges will need to be copied backwards.
        singular = np.argwhere(self.polygon_map == -1)

        # To make the map requires an array listing the start of each edge
        # and an array listing the end of each edge.
        starts = np.empty(self.faces.size + len(singular), np.intc)
        ends = np.empty(self.faces.size + len(singular), np.intc)

        for i in range(self.per_polygon):
            # Add only the counter-clockwise edges.
            # There is usually no need to add clockwise edges because the
            # polygon on the opposite side of the edge will add it.
            s = slice(i * len(self), (1 + i) * len(self))
            starts[s] = self.faces[:, i]
            ends[s] = self.faces[:, (i + 1) % self.per_polygon]

        if len(singular):
            # Add the edges going clockwise for the *singular* edges. i.e
            # Those which have no polygon on the other side of the edge.
            s = slice(-len(singular), None)
            starts[s] = self.faces[singular[:, 0],
                                   (singular[:, 1] + 1) % self.per_polygon]
            ends[s] = self.faces[singular[:, 0], singular[:, 1]]

        return RaggedArray.group_by(ends, starts, len(self.vertices))

    def connected_vertices(self, vertex: np.ndarray) -> np.ndarray:
        """List vertices which are directly connected to **vertex** by one
        polygon edge.

        Args:
            vertex:
                A single point from :attr:`vertices`.
                A NumPy array with shape ``(3,)``.
        Returns:
            An :py:`(n, 3)` array where ``n`` is the number of connected
            vertices.
        Raises:
            KeyError:
                If **vertex** is not in :attr:`vertices`.

        .. seealso::

            This method uses :attr:`vertex_map` under the hood.
            Use :attr:`vertex_map` if you're using vertex ids instead of raw
            vertices.

        """
        if vertex.shape != (3,):
            raise NotImplementedError("Must take a single vertex.")
        return self.vertices[self.vertex_map[self.vertex_table[vertex]]]

    @cached_property
    def reverse_faces(self):
        """A mapping of which polygons each vertex is in.

        This mapping uses flat indices. i.e. To find all instances of vertex
        123 use::

            polygon_ids, corners = \\
                np.divmod(self.reverse_ids[123], self.per_polygon)

        Then :py:`self.faces[polygon_ids, corners]` will all equal 123.

        """
        return RaggedArray.group_by(np.arange(self.faces.size),
                                    self.faces.ravel())

    def on_boundary(self, vertex: Union[np.ndarray, Integral]) -> bool:
        """Test if a **vertex** touches the edge of this mesh.

        Args:
            vertex:
                A 3D point in :attr:`vertices`. Or a single vertex ID.
        Returns:
            True if it touches, false otherwise.

        .. note::

            To check if a polygon touches a mesh edge, simply use::

                any(mesh.polygon_map[polygon_id] == -1)

        """
        if isinstance(vertex, Integral):
            id = vertex
        else:
            id = self.vertex_table[vertex]
        if not np.isscalar(id):
            raise ValueError("Only single vertices are supported.")
        polygons, sub_faces = np.divmod(self.reverse_faces[id],
                                        self.per_polygon)
        return np.any(self.polygon_map[polygons, sub_faces] == -1) \
               or np.any(self.polygon_map[polygons, sub_faces - 1] == -1)

    @cached_property
    def kdtree(self):
        """A KDTree_ with :attr:`centers` as its input data.

        This object powers all non-exact point lookup operations such as
        :meth:`closest_point`. Use its ``query_xxx()`` methods for more flexible
        lookup.

        .. _kdtree: https://github.com/storpipfugl/pykdtree

        """
        assert np.isfinite(self.centers).all()
        from pykdtree.kdtree import KDTree
        return KDTree(self.centers)

    def closest_point(self, target,
                      distance_upper_bound: Optional[float] = None,
                      interpolate=True) -> np.ndarray:
        """Find the nearest point on the surface of the mesh to **target**.

        Args:
            target:
                The point(s) to query. A :class:`numpy.ndarray` with
                :py:`shape[-1] == 3`.
            distance_upper_bound:
                A optional maximum allowed distance from **target** before
                giving up. In this case :py:`nan` is returned.
            interpolate:
                If true, the output will be the nearest point on the surface of
                the nearest polygon. Otherwise, it will only be the center of
                the nearest polygon.
        Returns:
            The nearest point(s) on the surface of the mesh. A
            :class:`numpy.ndarray` with the same shape as **target**.

        .. note::

            Under extreme circumstances, namely if **point** lies between two
            very close parallel-ish surfaces with enormous polygons, then the
            output is not guaranteed to be optimal. It is, however, guaranteed
            to be better than querying the nearest vertex.

        """
        target = np.asarray(target, dtype=self.kdtree.data.dtype)

        # PyKDTree only accepts 2D input arrays.
        # For single points (1D array) or arrays of arrays of points (>2D) we
        # need to convert inputs to 2D arrays then convert the outputs back
        # again. Ideally this would be fixed in pykdtree but the project is
        # borderline abandoned.
        shape = target.shape
        target = target.reshape((-1, 3))
        out = np.empty(target.shape, dtype=self.kdtree.data.dtype)

        # Find the nearest polygon.
        if distance_upper_bound is None:
            distances, args = self.kdtree.query(target)
            mask = slice(None)
        else:
            distances, args = self.kdtree.query(
                target, distance_upper_bound=distance_upper_bound)
            mask = np.isfinite(distances)
            out[~mask] = np.nan
            args = args[mask]

        if interpolate:
            out[mask] = geometry.snap_to_plane(target[mask], self.centers[args],
                                               self.units[args])
        else:
            out[mask] = self.centers[args]

        return out.reshape(shape)

    def local_maxima(self, heights: np.ndarray, boundaries: bool = True,
                     strict: bool = True) -> np.ndarray:
        """Find all :attr:`vertices` whose corresponding value in **heights** is
        greater than that of all its :meth:`connected_vertices`.

        Args:
            heights:
                A per-vertex scalar to rank by.
            boundaries:
                If false, ignore any vertices which :meth:`touch the mesh
                boundary <on_boundary>`.
            strict:
                If true, a vertex's value from **heights** must be strictly
                greater than its neighbours. Otherwise, it may be greater or
                equal.
        Returns:
            The vertex ids of the vertices which are local maxima.

        """
        mask = np.ones(len(self.vertices), bool)

        heights = np.asarray(heights)
        if heights.shape != (len(self.vertices),):
            raise ValueError("`heights` must be a 1D array with the same "
                             "length as `vertices`.")

        less_than = operator.lt if strict else operator.le

        for i in range(self.per_polygon):
            this = self.faces[:, i]
            next = self.faces[:, i - 1]
            # If the next vertex round is higher, then this is not a local
            # maxima. Read as:
            #   mask[this] &= less_than(heights[next], heights[this])
            np.logical_and.at(mask, this, less_than(heights[next],
                                                    heights[this]))
            # Strictly speaking, we could skip this line for closed meshes.
            # If this vertex is higher than the next vertex round, then the next
            # vertex is not a local maxima. Read as:
            #   mask[next] &= less_than(heights[this], heights[next])
            np.logical_and.at(mask, next, less_than(heights[this],
                                                    heights[next]))

        args = np.nonzero(mask)[0]
        if not boundaries:
            args = np.array([i for i in args if not self.on_boundary(i)])

        return args

    def save(self, file):
        """Write the mesh to a file or pseudo file. Currently only STL format
        and compressed variants of STL (``.stl.xz``) are supported."""
        data = np.empty(len(self), _Mesh.dtype)
        mesh = _Mesh(data, name=self.name)
        mesh.vectors = self.vectors
        mesh.normals = self.units
        mesh.name = self.name
        # Monkeypatch numpy-stl's header making method to use just the name we
        # gave it.
        mesh.get_header = lambda *args: (mesh.name or "")[:80].ljust(80, " ")
        with open_(file, "wb") as f:
            mesh.save("unused", fh=f, update_normals=False)


independent.init(Mesh)
Mesh._reset_on_rotate = independent.reset_on("rotate")
Mesh._reset_on_translate = independent.reset_on("translate")
Mesh._reset_all = independent.reset_all
