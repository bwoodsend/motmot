import operator

import numpy as np
from rockhopper import RaggedArray
import pytest

from motmot import Mesh, geometry, _compat, Curvature
from tests import faces_mesh, vectors_mesh


def cache_ables(cls):
    return [
        i for i in dir(cls)
        if isinstance(getattr(cls, i), (property, _compat.cached_property)) \
           and not i.startswith("_")
    ]


attrs = cache_ables(Mesh)
attrs.append("vertex_table")
attrs.remove("curvature")
[attrs.append("curvature." + i) for i in cache_ables(Curvature)]
attrs.remove("dtype")


def directly_modify(mesh):
    mesh.x += mesh.y * 2 + mesh.z
    mesh.reset()


def test_simple():
    self = vectors_mesh(10)
    normals = self.normals

    self.translate([1, 2, 3])
    assert self.normals is normals

    self.rotate([0, 0, 1], 3)
    assert self.normals is not normals


modifiers = [
    lambda mesh: None,
    lambda mesh: mesh.translate([5, 6, 7]),
    lambda mesh: mesh.rotate_using_matrix(Mesh.rotation_matrix([-.5, 4, 8], 2)),
    lambda mesh: mesh.rotate([-.5, 4, .8], -.6),
    lambda mesh: mesh.rotate([-.5, 4, .8], -.6, point=[5, 7, 2]),
    lambda mesh: mesh.crop((np.arange(len(mesh)) % 5) > 1, in_place=True),
    directly_modify,
] # yapf: disable


@pytest.mark.parametrize("modifier", modifiers)
@pytest.mark.parametrize("attr", attrs)
@pytest.mark.parametrize("use_id_mesh", [False, True])
def test_lazy_updates(modifier, attr, use_id_mesh):
    """
    Assert that our lazy attributes' caches are being cleared when needed.

    This works by:

        * duplicating a mesh,
        * loading a lazy attribute on one,
        * modifying (rotate/translate) both,
        * compare values of the lazy attribute.

    If the cache was correctly invalidated on the modification then the values
    should match at the end.

    """
    # attrgetter() can handle nested getattrs e.g. mesh.curvature.scaleless
    get = operator.attrgetter(attr)
    old = np.seterr(all="ignore")

    # Get two copies of the same mesh.
    if use_id_mesh:
        trial = faces_mesh(10)
        placebo = Mesh(trial.vertices.copy(), trial.faces.copy())
    else:
        trial = vectors_mesh(10)
        placebo = Mesh(trial.vectors.copy())

    trial.path = placebo.path = None

    # Initialise a lazy attribute.
    get(trial)

    # Modify both meshes.
    modifier(trial)
    modifier(placebo)

    # Check the outputs match.
    trial_, placebo_ = get(trial), get(placebo)
    if attr == "path":
        assert trial_ == placebo_
        return

    elif attr == "vertex_table":
        trial_, placebo_ = trial_.keys, placebo_.keys

    elif attr == "vertex_normals":
        # Occasionally, a vertex normal is [0, 0, 0] but with rounding errors.
        # These (normally tiny) rounding errors get geometry.normalised() up
        # inconsistently so [1e-15, 0, 0] -> [1, 0, 0] but [0, 0, 1e-15] ->
        # [0, 0, 1] causing the test to fail. Blot out these values by checking
        # if the non-normalised normals are close to 0.
        placebo_[geometry.magnitude(placebo._vertex_normals) < 1e-12] = np.nan
        trial_[geometry.magnitude(trial._vertex_normals) < 1e-12] = np.nan

    elif isinstance(trial_, RaggedArray):
        # This is a RaggedArray which doesn't (yet) support ==.
        assert np.array_equal(trial_.flat, placebo_.flat)
        assert np.array_equal(trial_.starts, placebo_.starts)
        assert np.array_equal(trial_.ends, placebo_.ends)
        np.seterr(**old)
        return

    elif attr == "kdtree":
        trial_ = trial_.data
        placebo_ = placebo_.data

    if isinstance(placebo_, np.ndarray):
        # ``np.nan == np.nan`` gives False which can cause this test to fail
        # incorrectly. Remove all nans.
        finite_mask = np.isfinite(placebo_)
        assert np.all(finite_mask == np.isfinite(trial_))
        trial_ = trial_[finite_mask]
        placebo_ = placebo_[finite_mask]

    # Some lazy attributes know that they are independent of certain
    # modifications. e.g. ``Mesh.normals`` isn't reset after a translation
    # as theoretically they should be independent. However, rounding errors
    # can get in the way very slightly.
    assert trial_ == pytest.approx(placebo_, rel=1e-3)

    np.seterr(**old)
