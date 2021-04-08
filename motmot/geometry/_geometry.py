# -*- coding: utf-8 -*-
"""
"""

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
