# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from motmot import Mesh, _compat
from tests import data, ids_mesh, vectors_mesh

pytestmark = pytest.mark.order(-1)

attrs = [
    i for i in dir(Mesh)
    if isinstance(getattr(Mesh, i), (property, _compat.cached_property)) \
    and not i.startswith("_")
]
attrs.append("_vertex_table")


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
    old = np.seterr(all="ignore")

    # Get two copies of the same mesh.
    if use_id_mesh:
        trial = ids_mesh(10)
        placebo = Mesh(trial.vertices.copy(), trial.ids.copy())
    else:
        trial = vectors_mesh(10)
        placebo = Mesh(trial.vectors.copy())

    trial.path = placebo.path = None

    # Initialise a lazy attribute.
    getattr(trial, attr)

    # Modify both meshes.
    modifier(trial)
    modifier(placebo)

    # Check the outputs match.
    trial_, placebo_ = getattr(trial, attr), getattr(placebo, attr)
    if attr == "path":
        assert trial_ == placebo_
        return

    elif attr == "_vertex_table":
        trial_, placebo_ = trial_.unique, placebo_.unique

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
