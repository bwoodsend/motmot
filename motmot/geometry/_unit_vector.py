# -*- coding: utf-8 -*-
"""
"""

import inspect
import re

import numpy as np

from motmot.geometry import inner_product, furthest, normalised


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
