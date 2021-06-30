# Motmot

<!---
    from urllib.parse import quote
    quote("python- {}-#4691C2.svg".format(
                " | ".join(["3.6", "3.7", "3.8", "3.9", "PyInstaller"])))
--->

![](https://img.shields.io/badge/python-%203.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%20PyInstaller-%234691C2.svg)


A sophisticated mesh class for analysing colourless [Polygon meshes] such as
an [STL file]
providing a seamless abstraction between raw *vectors* meshes or the more
efficient *vertices + faces* (a.k.a *vertices + polygons*) form.

* Free software: MIT license
* Source code: https://github.com/bwoodsend/motmot/
* Releases: https://pypi.org/project/motmot/
* Documentation: https://bwoodsend.github.io/motmot/index.html


### Related projects

This library focuses almost exclusively on analysing meshes.
It it highly likely that you will need to supplement it with other libraries
to read/write to a certain format or to simplify an existing mesh.

* Mesh read/write:

  * [numpy-stl]:
    Reads and writes STL files. This is a dependency of `motmot`.
  * [meshio](https://github.com/nschloe/meshio):
    Reads and writes a multitude of mesh formats.
  * [pymesh](https://github.com/taxpon/pymesh):
    Reads and writes STL and OBJ files.

* Mesh analysis:

  * [PyMesh](https://github.com/PyMesh/PyMesh):
    A highly sophisticated mesh library which unfortunately depends on some
    rather
    hairy C++ libraries, making it not generally installable.
    It's not even on PyPI.
  * [trimesh](https://github.com/mikedh/trimesh):
    Another general purpose mesh library. This one is pure Python and focuses
    strictly on triangular and preferably closed meshes.
    It also brings a few readers and writers with it.
    This library is very powerful.
    It's quite likely that you'd be better off using it instead of `motmot`.

* Mesh generation:

  * [meshzoo](https://github.com/nschloe/meshzoo):
    Creates finite mesh approximations of standard geometric shapes like cubes
    or spheres.

* Mesh cleaning:

  * [quad_mesh_simplify](https://github.com/jannessm/quadric-mesh-simplification):
    Decimate (collapse redundant or near redundant vertices in) meshes to make
    the filesize much smaller with negligible impact on quality.
  * [Py_Fast-Quadric-Mesh-Simplification](https://github.com/Kramer84/Py_Fast-Quadric-Mesh-Simplification):
    Another mesh decimator. This one is much faster but not on PyPI (yet).


### Usage

The basic API for ``motmot`` is modelled off that of [numpy-stl] with a few
alterations.


#### Initialisation

Meshes can be :

1.  Constructed from scratch using a single *vectors* array.
    This array should be 3D with shape ``(n, k, 3)`` where:

    * ``n`` is the number of polygons in the mesh,
    * ``k`` is the number of corners each polygon has,
    * ``3`` corresponds to having 3 axes. i.e. ``x``, ``y`` and ``z``.

    ```python
    # vectors is a (n, 3, 3) numpy array.
    triangle_mesh = Mesh(vectors)

    # vectors is a (n, 4, 3) numpy array.
    square_mesh = Mesh(vectors)
    ```

2.  Or using the more memory efficient *vertices + faces* form.

    ```python
    # `vertices` is an array of points. It should contain no duplicates.
    # `faces` is an integer array indicating which vertices are used to construct
    # each polygon.
    mesh = Mesh(vertices, faces)
    ```

3.  Read from an STL file. This uses [numpy-stl] under the hood.
    Currently, STL is the only file format that `motmot` will read implicitly:

    ```python
    from motmot import Mesh
    mesh = Mesh("some-file.stl")
    ```

4.  Read from an lzma, gzip or bzip2 compressed STL file:

    ```python
    from motmot import Mesh

    # An lzma compressed STL file. Create using `xz some-file.stl` in bash.
    mesh = Mesh("some-file.stl.xz")
    # A gzip compressed STL file. Create using `gzip some-file.stl` in bash.
    mesh = Mesh("some-file.stl.gz")
    # A bzip2 compressed STL file. Create using `bzip2 some-file.stl` in bash.
    mesh = Mesh("some-file.stl.bz2")
    ```


5.  Stream from any subclass of ``io.RawIOBase``.
    From here you can read from arbitrary sources such as
    embedded files, streams, archives or other pseudo files.
    For example, the following reads an STL directly from a web request:

    ```python
    from urllib import request
    from motmot import Mesh

    # Pull an STL file from the internet and load it without an intermediate
    # temporary file.
    url = "https://raw.githubusercontent.com/bwoodsend/vtkplotlib/master/" \
          "vtkplotlib/data/models/rabbit/rabbit.stl"

    with request.urlopen(url) as req:
        mesh = Mesh(req)
    ```


#### Vertices + Faces meshes vs Vectors meshes

There are two forms of mesh.

1.  A *vectors* mesh is essentially a list of polygons where
    each polygon is a list of points (its corners) and
    each point is an ``(x, y, z)`` triplet.
    This form is simple but wasteful because points which appear in multiple
    polygons are written multiple times which wastes memory and rendering time.

2.  A *vertices+faces* mesh takes all the unique points from a *vectors* mesh,
    calling them the *vertices*, then replaces each point in *vectors* with its
    index from *vertices*, calling this *faces*.
    Note that *faces* is often also known as *facets* or *polygons*.

Motmot makes the two forms interchangeable.
Each of *vectors*, *vertices* and *faces* are all available as attributes on all
meshes but,
depending on how a mesh is constructed,
*vectors* may be internally derived from *vertices* and *faces* or vice-versa.

```python
import numpy as np
from motmot import Mesh

# Define the 8 vertices in a cuboid.
vertices = np.array([
    [0., 0., 0.],
    [3., 0., 0.],
    [0., 5., 0.],
    [3., 5., 0.],
    [0., 0., 9.],
    [3., 0., 9.],
    [0., 5., 9.],
    [3., 5., 9.],
])

# Define the 6 sides of a cube or cuboid.
faces = np.array([
    # Draw a square using vertices[6], vertices[2], vertices[0] and vertices[4]
    [6, 2, 0, 4],
    # Draw a square using vertices[0], vertices[1], vertices[5] and vertices[4]
    [0, 1, 5, 4],
    # And so on...
    [0, 2, 3, 1],
    [5, 1, 3, 7],
    [3, 2, 6, 7],
    [4, 5, 7, 6],
])

# Construct a vertices+faces mesh.
mesh = Mesh(vertices, faces)
# This attribute is set to True to signify that this was originally a faces mesh.
mesh.is_faces_mesh
# Although `vectors` can still be derived automatically.
mesh.vectors

# Construct a vectors mesh.
mesh = Mesh(vertices[faces])
# This attribute is set to False to signify that this was originally a vectors
# mesh.
mesh.is_faces_mesh
# But `vertices` and `faces` can still be derived automatically.
mesh.vertices, mesh.faces
```


#### Mesh properties

This is just a brief summary of what is available.
See the corresponding entry in the
[the API reference]
for more information on each property.

```python
# Outward normal to each polygon:
>>> mesh.normals
array([[-45.,   0.,   0.],
       [ -0., -27.,  -0.],
       [ -0.,  -0., -15.],
       [ 45.,   0.,  -0.],
       [ -0.,  27.,   0.],
       [  0.,   0.,  15.]])

# Normalised (magnitude of 1.0) outward normal to each polygon:
>>> mesh.units
array([[-1.,  0.,  0.],
       [ 0., -1.,  0.],
       [ 0.,  0., -1.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])

# Area of each polygon.
>>> mesh.areas
array([45., 27., 15., 45., 27., 15.])

# Total surface area (just a sum of mesh.areas).
>>> mesh.area
174.0

# The number of times each vertex is used (which admittedly
# isn't particularly interesting for a cuboid):
>>> mesh.vertex_counts
array([3, 3, 3, 3, 3, 3, 3, 3], dtype=int32)

# A mapping of which other vertices each vertex is directly connect to.
>>> mesh.vertex_map
RaggedArray.from_nested([
    [1, 7, 3],  # vertices[0] connects to vertices[[1, 7, 3]].
    [2, 6, 0],  # vertices[1] connects to vertices[[2, 6, 0]].
    [4, 1, 3],  # and so on...
    [5, 0, 2],
    [5, 6, 2],
    [4, 7, 3],
    [1, 4, 7],
    [0, 5, 6],
])

# Because this mesh's vertices appear the same number of times,
# this example slightly trivialises the problem. Consider instead
# a mesh with only the first three faces. Not all vertices have
# the same number of neighbours.
>>> mesh[:3].vertex_map
RaggedArray.from_nested([
    [1, 3],
    [2, 6, 0],
    [4, 1, 3],
    [0, 2, 5],
    [5, 2, 6],
    [3, 4],
    [4, 1],
])

# If you prefer to use raw vertices rather than vertex IDs then
# use the connected_vertices() method.
>>> mesh.connected_vertices(mesh.vertices[0])
array([[0., 5., 0.],
       [3., 5., 9.],
       [0., 0., 9.]])

# Similarly, `polygon_map` maps every polygon to each of its neighbours.
# Read the first line of the following as *polygon 0 shares an edge each with
# polygons 4, 2, 1 and 5*.
>>> mesh.polygon_map
array([[4, 2, 1, 5],
       [2, 3, 5, 0],
       [0, 4, 3, 1],
       [1, 2, 4, 5],
       [2, 0, 5, 3],
       [1, 3, 4, 0]])
```


#### Vertex Lookup

`motmot` leverages two libraries for looking up vertices.

* [hirola.HashTable](https://hirola.readthedocs.io/en/latest/) for [exact lookup](#exact-lookup)
* [pykdtree.kdtree.KDTree](https://github.com/storpipfugl/pykdtree) for [fuzzy lookup](#approximate-lookup)


##### Exact lookup

It is easy to convert vertex IDs to real vertices.
Simply pass them as indices to `mesh.vertices`.

```python
>>> ids = [0, 4, 5, 2]
>>> points = mesh.vertices[ids]
>>> points
array([[0., 0., 0.],
       [0., 0., 9.],
       [3., 0., 9.],
       [0., 5., 0.]])
```

Go the other way by indexing the `vertex_table` attribute.

```python
>>> mesh.vertex_table[points]
array([0, 4, 5, 2], dtype=int64)
```

Some things to be aware of:

*   The `dtype` of the points queried must match `mesh.dtype`.

*   As with regular floats in a regular Python `dict`,
    even the smallest deviation will cause lookup to fail.

    ```python
    >>> mesh.vertex_table[[3., 0., 9.]]
    5
    >>> mesh.vertex_table[[3., 0, 9.00000000001]]
    KeyError: 'key = array([3., 0., 9.]) is not in this table.'
    ```


##### Approximate lookup

To find *nearest points*, `motmot` uses a [KDTree].
The API here is very shallow and it is quite likely that you may wish to
create and use `KDTree`s directly rather than use `motmot`'s methods.

A KDTree, fitted to `mesh.centers` (the middle of each polygon),
is found at the `mesh.kdtree` attribute.

Given a set of points defined as:
```python
points = np.array([[2., 3.5, 4.2], [2.3, 4.2, 1.1]], mesh.dtype)
```

Find the nearest point on the mesh surface to each point:

```python
>>> mesh.closest_point(points)
array([[3. , 3.5, 4.2],
       [2.3, 4.2, 0. ]])
```
Or to restrict the output to only `mesh.centers` without interpolating between
them:

```python
>>> mesh.closest_point(points, interpolate=False)
array([[3. , 2.5, 4.5],
       [1.5, 2.5, 0. ]])
```

For anything else, use `mesh.kdtree` directly.


#### Laziness

A `motmot.Mesh` *lazy loads* its properties using a backport of
[@functools.cached_property].
This allows them to be calculated when only you need them so that no time is
ever wasted calculating something which you do not use.
Take for example, [mesh.normals].
Nothing is calculated on
`mesh = Mesh(vertices, faces)` so that if the normals are never used then they are
never calculated.
Accessing the attribute `mesh.normals` initialises and returns
them making `mesh.normals` look like a regular attribute on the outside.
The value is cached so that the calculation never runs more than once.
i.e. `mesh.normals is mesh.normals`.

Caches should be reset after a mesh is modified.
Most of this is done automatically.
Mesh modifier methods such as `rotate()`, `translate()` or `crop(in_place=True)`
will all invalidate affected caches themselves.
Similarly, setting any of the `vertices`, `faces` or `vectors` attributes will
reset all caches.
Writing in place to those arrays (e.g. `mesh.vectors[:] = x`) however
is undetectable to `motmot`.
Call `mesh.reset()` after doing an in place modification.


[numpy-stl]: https://github.com/wolph/numpy-stl
[@functools.cached_property]: https://docs.python.org/3/library/functools.html#functools.cached_property
[the API reference]: https://motmot.readthedocs.io/en/latest/stubs/mesh.html
[mesh.normals]: https://motmot.readthedocs.io/en/latest/stubs/mesh.html#motmot.Mesh.normals
[KDTree]: https://github.com/storpipfugl/pykdtree
[Polygon meshes]: https://en.wikipedia.org/wiki/Polygon_mesh
[Triangle mesh]: https://en.wikipedia.org/wiki/Triangle_mesh
[STL file]: https://en.wikipedia.org/wiki/STL_(file_format)
