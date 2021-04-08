# -*- coding: utf-8 -*-
"""
"""

import pytest

from motmot._compat import cached_property
from motmot._misc import Independency

pytestmark = pytest.mark.order(0)


def test():
    independent = Independency()

    class Foo(object):
        x = 0
        y = 0

        def set_x(self, new):
            self.x = new
            reset_on_set_x(self)

        def set_y(self, new):
            self.y = new
            reset_on_set_y(self)

        @independent.of("set_y")
        @cached_property
        def x_mirror(self):
            return self.x

        @independent.of("set_x")
        @cached_property
        def y_mirror(self):
            return self.y

        @cached_property
        def total(self):
            return self.x + self.y

    independent.init(Foo)
    reset_on_set_x = independent.reset_on("set_x")
    reset_on_set_y = independent.reset_on("set_y")

    self = Foo()
    self.x, self.y = 1, 2
    # All cached properties should initialise to the correct values.
    assert self.x_mirror == 1
    assert self.y_mirror == 2
    assert self.total == 3

    self.x, self.y = (3, 4)
    # The cached properties should not have been reset and still hold out of
    # date values.
    assert self.x_mirror == 1
    assert self.y_mirror == 2
    assert self.total == 3

    self.set_x(5)
    # set_x() invalidates the cache on x_mirror and total but not y_mirror.
    assert self.x_mirror == 5
    assert self.y_mirror == 2
    assert self.total == 9

    self.x = 6
    self.set_y(7)
    # set_y() invalidates the cache on y_mirror and total but not x_mirror.
    assert self.x_mirror == 5
    assert self.y_mirror == 7
    assert self.total == 13

    self.x, self.y = 8, 9
    # reset_all() should reset everything.
    independent.reset_all(self)
    assert self.x_mirror == 8
    assert self.y_mirror == 9
    assert self.total == 17