# -*- coding: utf-8 -*-
"""
"""

# This is a backport of functools.cached_property().

_UNINITIALISED = object()


class cached_property:  # pragma: no cover

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        from threading import RLock
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r}).")

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        cache = instance.__dict__
        val = cache.get(self.attrname, _UNINITIALISED)
        if val is _UNINITIALISED:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _UNINITIALISED)
                if val is _UNINITIALISED:
                    val = self.func(instance)
                    cache[self.attrname] = val
        return val
