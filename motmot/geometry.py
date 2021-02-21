# -*- coding: utf-8 -*-
"""
"""

import inspect
import re
import numpy as np
import numpy  # Needed for PyCharm's type checking.


def inner_product(a, b, keepdims=False):
    """Calculates the scalar/inner/interior/dot/"whatever you want to call it"
    product of vectors **a** and **b**, returning a scalar.

    Arguments **a** and **b** must be numpy-broadcastable.

    .. seealso::

        The :class:`UnitVector` class for a more convenient way to perform
        multiple :meth:`inner_product` calls.
    """
    return _sum_last_axis(np.asarray(a) * b, keepdims=keepdims)


def magnitude(vector, keepdims=False):
    """Calculate the hypotenuse/magnitude/length  of a **vector**."""
    return np.sqrt(magnitude_sqr(vector, keepdims))


def magnitude_sqr(vector, keepdims=False):
    """Calculate the square of the hypotenuse of a **vector**.

    This is faster than :func:`magnitude` because it skips taking the square
    root and can be used to compare or sort distances.

    """
    vector = np.asarray(vector)
    return _sum_last_axis(vector * vector, keepdims)


def _sum_last_axis(x, keepdims=False):
    """Faster ``np.sum(x, axis=-1, keepdims=keepdims)``."""
    return _reduce_last_axis(x, np.add, keepdims)


def _reduce_last_axis(x, operator, keepdims=False):
    """Faster :meth:`numpy.ufunc.reduce()`."""
    if keepdims:
        return _reduce_last_axis(x, operator)[..., np.newaxis]

    x = np.asarray(x)

    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], operator.identity, x.dtype)

    itr = iter(x.T)
    out = next(itr)
    for i in itr:
        out = operator(out, i)
    out = out.T

    return out


def furthest(points, direction, n=None, return_projection=False,
             return_args=False):
    """Select the point the furthest in **direction**.

    Args:
        points (numpy.ndarray):
            Some points.
        direction (numpy.ndarray or UnitVector):
            A direction.
        n (int or None):
             Specify the furthest **n** points instead of just one.
        return_projection (bool):
             If true, also return the projection of the furthest point(s).
        return_args (bool):
            If true, also return the index(es) similar to :func:`numpy.argmax`.

    """
    points = np.asarray(points)

    if points.size == 0:
        raise ValueError(
            "Can't select furthest from an empty array of `points`.")

    heights = inner_product(points, direction)

    if n is None:
        args = np.argmax(heights, axis=None)
    else:
        if not n:
            args = np.empty(0, int)
        else:
            args = np.argpartition(heights, -n, axis=None)[-n:]
            args = args[np.argsort(heights[np.unravel_index(
                args, heights.shape)])][::-1]

    args = np.unravel_index(args, heights.shape)

    out = points[args]
    if not (return_args or return_projection):
        return out
    out = out,
    if return_projection:
        out += (heights[args],)
    if return_args:
        out += (args,)
    return out


def normalise(vector):
    """Modify in-place **vector** so that it has magnitude ``1.0``.

    Args:
        vector (numpy.ndarray):
            Vector(s) to normalise.

    Returns:
        numpy.ndarray:
            The original magnitudes of **vector**.

    .. seealso:: :func:`normalised` which leaves the original as-is.

    """
    magnitudes = magnitude(vector, keepdims=True)
    vector /= magnitudes
    return magnitudes


def normalised(vector):
    """Return a normalised copy of **vector**.

    .. seealso:: :func:`normalise` to modify in-place.

    """
    return vector / magnitude(vector, keepdims=True)


def get_components(points, *unit_vectors):
    """Get the inner product for each unit vector as separate arrays.

    Args:
        points (numpy.ndarray):
        *unit_vectors (numpy.ndarray or UnitVector):

    Returns:
        tuple[numpy.ndarray]:
            Projections for each unit vector.

    """
    return tuple(inner_product(points, uv) for uv in unit_vectors)


def get_components_zipped(points, *unit_vectors):
    """Get the inner product for each unit vector.

    Args:
        points:
        *unit_vectors (numpy.ndarray or UnitVector):

    Returns:
        numpy.ndarray:
            Projections as one array.

    The unit-vector iteration corresponds to the last axis of the output. i.e ::

        out[..., i] == inner_product(unit_vectors[i], points)

    The more linear-algebra savvy developer will know that this is just a matrix
    multiplication. This function is purely to reduce confusion for those (like
    me) who can never remember if you pre or post multiply and when to
    transpose.

    """
    return points @ np.array(unit_vectors).T


def center_of_mass(points, weights=None):
    """The (weighted) mean of **points**."""
    if weights is None:
        return np.array([i.mean() for i in points.T])
    else:
        weights = weights[(...,) + (np.newaxis,) * (points.ndim - weights.ndim)]
        return np.array([i.sum() for i in (points * weights).T]) / weights.sum()


def orthogonal_bases(v0):
    """Create a set of perpendicular axes.

    Args:
        v0 (numpy.ndarray):
            The first axis.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            Three perpendicular unit vectors.

    The first returned axis is just the input :func:`normalised`. Although there
    are many possible valid outputs for any given input, this function is fully
    deterministic. Reproducibility is guaranteed.

    """
    old = np.seterr(divide="ignore", invalid="ignore")

    v0 = np.asarray(v0)
    shape = v0.shape
    v0 = v0.reshape((-1, v0.shape[-1]))
    v0 = normalised(v0)
    if not np.isfinite(v0).all():
        raise ValueError("v0 must have non zero magnitude")

    to_do = None
    v1 = np.empty_like(v0)
    v2 = np.empty_like(v0)

    for seed in np.eye(3):  # pragma: no branch
        if to_do is not None and len(to_do) == 0:
            break
        v0[to_do], v1[to_do], v2[to_do], _to_do = \
            _orthogonal_bases(v0[to_do], seed)
        if to_do is None:
            to_do = _to_do
        else:
            to_do = to_do[_to_do]

    np.seterr(**old)
    return v0.reshape(shape), v1.reshape(shape), v2.reshape(shape)


def _orthogonal_bases(v0, seed):
    v1 = np.cross(v0, seed)
    v1_mag = normalise(v1)
    v2 = np.cross(v0, v1)
    return v0, v1, v2, (v1_mag.ravel() == 0).nonzero()[0]


class UnitVector(object):
    """Unit vectors symbolise directions.

    A :class:`UnitVector` wraps around an numpy array, stored in the
    :attr:`vector` attribute, which contains the actual data. It behaves
    exactly like an array but with extra methods.

    It is also callable, which applies the inner-product.

    **Usage Example**:

    Suppose we have a point cloud of ``100`` points called ``points``.

    .. code-block:: python

        import numpy as np
        from motmot import geometry

        points = np.random.uniform(-30, 30, (100, 3))

    In a wonky coordinate system that has been so that `up` is actually the
    diagonal unit-vector below.

    .. math::

        \\begin{bmatrix}
        \\frac{3}{5}\\quad
        \\frac{4}{5}\\quad
        0
        \\end{bmatrix}

    Which is passed to Python using the following. Note that the vector is
    normalised automatically.

    .. code-block:: python

        >>> up = geometry.UnitVector([3, 4, 0])
        >>> up
        UnitVector([0.6 0.8 0. ])

    This can do everything a numpy array can do. Any outputs produced are of
    type :class:`numpy.ndarray`.

    .. code-block:: python

        >>> up * 2
        array([1.2, 1.6, 0. ])
        >>> up + 1
        array([1.6, 1.8, 1. ])
        >>> up + up
        array([1.2, 1.6, 0. ])

    There is one exception to the above: Negating it
    returns another :class:`UnitVector`. ::

        >>> -up
        UnitVector([-0.6 -0.8 -0. ])

    To get the heights of **points** use the inner product. ::

        heights = up.inner_product(points)

    A typical 3D analysis will involve lots of inner-product calls so the above
    gets combersome pretty quickly. For convenience, you may instead call the
    vector directly. ::

        heights = up(points)

    Once you have your heights, the following numpy functions are your friends::

        heights.min()   # minimum height
        heights.max()   # maximum height
        heights.mean()  # average height
        heights.ptp()   # `peak to peak` equivalent to max - min

    Get the highest point using::

        highest = up.furthest(points)

    Or::

        highest, max_height = up.furthest(points, return_projection=True)

    If you actually want the lowest then invert ``up`` to get a ``down`` vector.
    ::

        lowest = (-up).furthest(points)

    Methods :meth:`get_component` and :meth:`remove_component` split a point
    into parts parallel and perpendicular to ``up``.

    Methods :meth:`with_` and :meth:`match` return points
    modified to have specified heights. They both have the effect of mapping
    the input onto a plane with ``up`` as the plane normal.

    .. code-block:: python

        # Get `points` but with a height of 10.
        up.with_projection(points, 10)
        # Get `points` but with the same height as ``points[0]``.
        up.match_projection(points, points[0])

    """

    def __init__(self, vector):
        """
        Args:
            vector(numpy.ndarray or UnitVector or str or list or tuple):

                The direction as a vector, or an axis name as a string.

        **vector** is normalised automatically.

        The str axis name format is an optional sign (+-) followed by any of
        ``'ijkxyz'``. Whitespaces and case are ignored. ::

            >>> UnitVector("Z")
            UnitVector([0. 0. 1.])
            >>> UnitVector("+x")
            UnitVector([1. 0. 0.])
            >>> UnitVector("-y")
            UnitVector([ 0. -1.  0.])
            >>> UnitVector("-j")
            UnitVector([ 0. -1.  0.])

        """
        if isinstance(vector, str):
            vector = _str_to_vector(vector)
        self.vector = normalised(np.asarray(vector))

    vector: np.ndarray
    """The raw numpy vector."""

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.vector.tolist())

    def __call__(self, vector, keepdims=False):
        return inner_product(self.vector, vector, keepdims=keepdims)

    inner_product = __call__

    def matched_sign(self, vector):
        """Return a reversed copy of **vector** if ``self(vector) < 0``. i.e if
        the angle between *this* vector and **vector** is more than 180°.
        """
        return vector * np.where(self(vector) < 0, -1, 1)

    def __neg__(self):
        return type(self)(-self.vector)

    def remove_component(self, vector):
        """Get the components of **vector** perpendicular to *this* direction.
        """
        return vector - self.get_component(vector)

    def get_component(self, vector):
        """Get the component of **vector** parallel to *this* direction."""
        return self(vector, keepdims=True) * self.vector

    def furthest(self, points, n=None, return_projection=False,
                 return_args=False):
        """Select, from **points**, the point with the highest projection in
        *this* direction.
        """
        return furthest(points, self.vector, n, return_projection, return_args)

    def with_(self, point, projection):
        """Translate **point** along *this* direction so that
        ``self(point) == projection``.

        The output is a modified copy.
        """
        return (np.asarray(projection)[..., np.newaxis] -
                self(point, keepdims=True)) * self + point

    def match(self, point, target_point):
        """Translate **point** so that ``self(point) == self(target_point)``

        Or the returned point is inline with **target_point**.
        """
        return self.with_(point, self(target_point))


_EXCLUDES = {
    "__array_finalize__",
    "__array_function__",
    "__array_interface__",
    "__array_prepare__",
    "__array_priority__",
    "__array_struct__",
    "__array_ufunc__",
    "__array_wrap__",
    "__neg__",
    "__reduce__",
    "__getattr__",
    "__getattribute__",
    "__class__",
    "__copy__",
    "__deepcopy__",
    "__reduce_ex__",
    "__prepare__",
    "__setstate__",
    "__new__",
    "__setattr__",
    "__setattribute__",
    "__dir__",
}  # | set(dir(UnitVector)) - {"__eq__"}

_METHOD_SOURCE = {}
# _METHOD_SOURCE.update(dict.fromkeys(dir(object), object))
_METHOD_SOURCE.update(
    dict.fromkeys(set(dir(np.ndarray)) - _EXCLUDES, np.ndarray))
_METHOD_SOURCE.update(
    dict.fromkeys(
        set(dir(UnitVector)) -
        set(i for i in dir(UnitVector)
            if getattr(object, i, None) is getattr(UnitVector, i)),
        UnitVector,
    ))


def _unitvector_proxy_wrap(attr):

    def f(self, *args):
        #print(attr)
        return getattr(self.vector, attr)(*args)

    return f


for (_attr, _value) in inspect.getmembers(np.ndarray, callable):
    if _METHOD_SOURCE.get(_attr) is np.ndarray:
        setattr(UnitVector, _attr, _unitvector_proxy_wrap(_attr))

_AXES = dict(zip("ijk", np.eye(3)))
_AXES.update(zip("xyz", np.eye(3)))
_SIGNS = {"+": 1, None: 1, "-": -1}
_str_vector_re = re.compile(r"\s*([+-])?\s*(?:([i-k])|([x-z]))")


def _str_to_vector(x):
    """Convert an axis name, such as 'Z' to a unit vector [0, 0, 1]."""
    match = _str_vector_re.match(x.lower())
    if match is None:
        raise ValueError("{!r} is not a valid unit-vector.".format(x))
    sign, ijk, xyz = match.groups()
    if ijk is None:
        axis = ord(xyz) - ord("x")
    else:
        axis = ord(ijk) - ord("i")
    out = np.zeros(3)
    out[axis] = _SIGNS[sign]
    return out


def area(polygon) -> np.ndarray:
    """Calculate the area of arbitrary polygons."""
    polygon: np.ndarray = np.asarray(polygon)
    assert polygon.ndim >= 2
    assert polygon.shape[-1] == 3

    if polygon.shape[-2] < 3:
        # Area of a line or a dot is 0.
        return np.zeros(polygon.shape[:-2], polygon.dtype)

    # Triangulate using the first vertex as a focus. i.e. A hexagon:
    #   (p0, p1, p2, p3, p4, p5)
    # becomes 4 triangles which all start at p0:
    #   (p0, p1, p2)
    #   (p0, p2, p3)
    #   (p0, p3, p4)
    #   (p0, p4, p5)

    from_v0 = polygon[..., 1:, :] - polygon[..., 0, np.newaxis, :]
    crosses = [(np.cross(from_v0[..., i, :], -from_v0[..., i + 1, :]))
               for i in range(polygon.shape[-2] - 2)]

    # Area calculation can be simple but isn't if either all vertices are not
    # co-planer (in which case `area` is technically arbitrary as it differs
    # depending on which vertex is used as a triangulation focus) or if the
    # polygon contains inside corners > 180° (think of a pie with a wedge
    # removed).

    # In the simple case, just magnitude() each item in `crosses`, add them up,
    # then divide by 2.
    # Non-co-planer polygons are doomed almost by definition - this complication
    # is more or less ignored.
    # To handle the 2nd problem, some sign magic must be done to treat the
    # missing slice of the pie as a negative area. This is done by looking at
    # normals to each triangle: The normal to a missing wedge will point in
    # the opposite direction to all the other triangles' normals.

    # Generate a consensus normal to this polygon which can be used to
    # sign-match individual triangle normals and thereby detect missing wedges.
    normal = sum(crosses[1:], crosses[0])

    # Calculate a signed area for each triangle. Note that these areas are
    # doubled what they should be.
    areas_x2 = (
        np.copysign(magnitude(i), inner_product(normal, i)) for i in crosses)

    # Add all double-areas together and un-double them.
    return sum(areas_x2, next(areas_x2)) / 2
