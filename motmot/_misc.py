# -*- coding: utf-8 -*-
"""Dumping ground for random bits and bobs."""

from collections import defaultdict as _default_dict
import os
import io
import numpy as np

from motmot._compat import cached_property as _cached_property

idx = type("IDx", (object,), dict(__getitem__=lambda self, x: x))()


class Independency(_default_dict):
    """Manage invalidation of :class:`cached_property`.

    Most cached properties will need refreshing after calling a method which
    modifies the object. There are exceptions. A mesh's area for example doesn't
    change if it is rotated. Provide a means to bulk invalidate cached
    properties and to exclude properties from invalidation.
    """

    def __init__(self):
        super().__init__(set)
        self.cls = None

    def init(self, cls: type):

        self.cls = cls
        self.cached_properties = {
            i.attrname for i in vars(cls).values()
            if isinstance(i, _cached_property) for cls in self.cls.mro()
        }

    def of(self, *operations):

        def wrap(cached: _cached_property):
            for operation in operations:
                self[operation].add(cached.func.__name__)
            return cached

        return wrap

    def reset_on(self, operation):
        return self._resetter(self.cached_properties - self[operation])

    def _resetter(self, to_invalidate):

        def reset(self):
            for i in to_invalidate:
                self.__dict__.pop(i, None)

        return reset

    def reset_all(self, obj):
        for i in self.cached_properties:
            obj.__dict__.pop(i, None)


def read_archived(file) -> io.BytesIO:
    """Read a file which may be ``.xz``, ``.bz2`` or ``.gz`` compressed.

    Args:
        file:
            The filename.
    Returns:
        The decompressed bytes wrapped in an :class:`io.BytesIO`.

    """
    assert isinstance(file, (str, os.PathLike))
    suffix = os.path.splitext(file)[1]

    if suffix == ".xz":
        from lzma import open
    elif suffix == ".gz":
        from gzip import open
    elif suffix == ".bz2":
        from bz2 import open
    else:
        from builtins import open

    with open(file, "rb") as f:
        # For some reason, just passing the open compressed file to
        # numpy-stl causes it to only read some of it.
        # Create a redundant intermediate io.BytesIO().
        file = io.BytesIO(f.read())

    return file


def as_nD(x: np.ndarray, n: int, name: str) -> np.ndarray:
    """Prepend new axes until **x** is **n** dimensional. Raise an error if
    **x** has too many dimensions."""
    if x.ndim > n:
        raise ValueError(
            f"'{name}' has too many dimensions. "
            f"A {n}D array was expected but got an array with shape {x.shape}.")
    while x.ndim < n:
        x = x[np.newaxis]
    return x
