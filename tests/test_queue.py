# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from motmot._queue import Queue

pytestmark = pytest.mark.order(3)


def test():
    self = Queue(10)
    assert not self

    for i in range(7):
        self.append(i)
    assert len(self) == 7
    assert self

    assert np.all(self.queue[:7] == np.arange(7))
    assert np.all(self.in_waiting == np.arange(7))

    assert self.consume() == 0
    assert self.consume() == 1
    assert self.consume() == 2
    assert self.consume_index == 3

    assert np.array_equal(self.in_waiting, np.arange(3, 7))
    assert repr(self) == "Queue [ - - - 3 4 5 6 - - - ]"
    assert len(self) == 4

    self.appends(np.arange(7, 12) % self.max_size)

    assert self.in_waiting.tolist() == [3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert repr(self) == "Queue [ 0 1 - 3 4 5 6 7 8 9 ]"
    assert len(self) == 9


def test_add():
    self = Queue(6)

    self.add(4)
    assert self.in_waiting.tolist() == [4]
    self.add(4)
    assert self.in_waiting.tolist() == [4]
    self.add(5)
    assert self.in_waiting.tolist() == [4, 5]
    self.add(4)
    assert self.in_waiting.tolist() == [4, 5]
    self.consume()
    assert self.in_waiting.tolist() == [5]
    self.add(4)
    assert self.in_waiting.tolist() == [5, 4]
