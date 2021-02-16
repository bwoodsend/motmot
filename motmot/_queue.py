# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from cslug import ptr

from motmot._slug import slug


class Queue(object):
    """Wrapper class around the C Queue structure.

    For problems that intuitively would be solved by recursive functions but,
    due to the high recursion depths that may involve ``O(len(mesh))`` or
    ``O(len(mesh) ** 2)`` etc., will cause ``StackOverflowErrors`` (the C
    equivalent of ``RecursionError``).

    Instead of invoking itself functions should treat a :meth:`Queue` as a to-do
    list by pop-reading from the front, appending to the end, and stopping when
    the queue is empty.

    This Queue is implemented in C to be used in C. Whilst its methods have been
    ported into Python, they are for debugging only and will likely be slower
    than plain Python ``list``.

    Supported methods in C are:

    .. code-block:: C

        // Add x to the end of the queue if x is not already in it.
        Q_add(queue, x);

        // Append x to the end of the queue - note dangerous. See below.
        Q_append(queue, x);

        // Read and pop the element at the front of the queue.
        int x = Q_consume(queue);

        // Get number of unprocessed elements left in the queue.
        int count = Q_len(queue);

        // Check if queue is empty.
        bool empty = Q_is_empty(queue);

        // Just `append` wrapped in a for loop.
        Q_appends(queue, array, len_array);

    In Python, functions are available as queue methods
    e.g. ``queue.append(x)``.


    This class also makes some tight assumptions and restrictions on usage.

    - Queue elements must be of C ``int`` type and  satisfy
      ``0 <= x < queue.max_size``.

    - It is a strict FIFO (first in first out). Elements can only be added to
      end and removed from the start.

    - Elements in the queue **must be unique**. Use ``add(queue, x)`` instead of
      ``append(queue, x)`` to enforce this.

    - ``queue.max_size`` must be 1 greater than the maximum number of unique
      elements that will go in the queue.

    It is recommended to create this object in Python then pass
    ``queue.c_queue_ptr`` to any C function that needs a queue, rather than
    allocate and (likely forget to) deallocate memory in C.

    """

    def __init__(self, max_size):
        self.queue = np.empty(max_size, dtype=np.intp, order="C")
        self.reverse_queue = np.full_like(self.queue, -1)
        self.max_size = max_size

        self._queue = ptr(self.queue)
        self._reverse_queue = ptr(self.reverse_queue)

        self._raw = slug.dll.Queue(self._queue, self._reverse_queue, 0, 0,
                                   self.max_size)

        self.needs_clearing = None

    @property
    def consume_index(self):
        return self._raw.consume_index

    @property
    def append_index(self):
        return self._raw.append_index

    def append(self, value):
        slug.dll.Q_append(self._raw._ptr, value)

    def appends(self, value):
        value = np.asarray(value, dtype=np.intp, order="C")
        slug.dll.Q_appends(self._raw._ptr, ptr(value), value.size)

    def consume(self):
        return slug.dll.Q_consume(self._raw._ptr)

    def add(self, value):
        slug.dll.Q_add(self._raw._ptr, value)

    def __len__(self):
        return slug.dll.Q_len(self._raw._ptr)

    def __del__(self):
        try:
            slug.dll.Q_destroy_queue(self._raw._ptr)
        except AttributeError:
            pass

    def get_between(self, start, end):
        start %= self.max_size
        end %= self.max_size

        parts = []
        if start > end:
            parts.append(self.queue[start:])
        parts.append(self.queue[start:end])
        if start > end:
            parts.append(self.queue[:end])

        return np.concatenate(parts)

    @property
    def in_waiting(self):
        return self.get_between(self.consume_index, self.append_index)

    def __repr__(self):
        out = [type(self).__name__, "["]
        toggle = self.append_index < self.consume_index
        for (i, x) in enumerate(self.queue):
            if (self.consume_index <= i) == (i < self.append_index) ^ toggle:
                out.append(str(x))
            else:
                out.append("-")
        out += ["]"]
        return " ".join(out)

    def __bool__(self):
        return not slug.dll.Q_is_empty(self._raw._ptr)
