# -*- coding: utf-8 -*-
"""Dumping ground for random bits and bobs."""

from collections import defaultdict as _default_dict
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
