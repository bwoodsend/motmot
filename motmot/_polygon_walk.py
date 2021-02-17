# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from cslug import ptr

from motmot._queue import Queue, slug


def connected(polygon_map, initial, mask=None):
    """Find all nodes connected to the **initial** node(s)."""
    # Just a wrapper for :c:`connected()` from ``polygon_map.c``.

    initial = np.asarray(initial, dtype=np.intp, order="C").ravel()
    assert (0 <= initial).all() and (initial < len(polygon_map)).all()

    if mask is not None:
        polygon_map = np.where(mask[polygon_map], polygon_map, -1)

    queue = Queue(len(polygon_map) + 1)

    assert polygon_map.dtype == np.intp
    slug.dll.connected(ptr(initial), len(initial), polygon_map.shape[1],
                       ptr(polygon_map), queue._raw._ptr)

    return queue.get_between(0, queue.append_index)
