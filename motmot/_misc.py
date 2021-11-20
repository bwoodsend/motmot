# -*- coding: utf-8 -*-
"""Dumping ground for random bits and bobs."""

import contextlib
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


@contextlib.contextmanager
def _buffer_open(file: io.IOBase, *args, **kwargs):
    """A dummy context manager to be used in place of open() if the file is
    already a either an open file object or a pseudo file."""
    yield file
    return


def _opener(file):
    """Select an appropriate open() like function for a given file type."""
    if isinstance(file, io.IOBase):
        return _buffer_open

    if not isinstance(file, (str, os.PathLike)):
        from builtins import open
        return open

    suffix = os.path.splitext(file)[1]

    if suffix == ".xz":
        from lzma import open
    elif suffix == ".gz":
        from gzip import open
    elif suffix == ".bz2":
        from bz2 import open
    else:
        from builtins import open

    return open


def open_(file, *args, **kwargs):
    """A drop-in replacement to the builtin open() function.

    * If given an open file handle or pseudo file (e.g. io.BytesIO()) the file
      is returned wrapped in a NO-OP context manager.
    * If given a filename ending in a standard compression format suffix (.gz,
      .xz, .bz2) then a appropriate file wrapper is returned which handles the
      compression or decompression (e.g. lzma.open()),

    Otherwise, it defers to the builtin open() behaviour.

    """
    return _opener(file)(file, *args, **kwargs)


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
