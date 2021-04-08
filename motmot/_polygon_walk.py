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

    polygon_map = mask_polygon_map(polygon_map, mask)

    queue = Queue(len(polygon_map) + 1)

    assert polygon_map.dtype == np.intp
    slug.dll.connected(ptr(initial), len(initial), polygon_map.shape[1],
                       ptr(polygon_map), queue._raw._ptr)

    return queue.get_between(0, queue.append_index)


def mask_polygon_map(polygon_map, mask):
    if mask is not None:
        if mask.ndim == 1:
            mask = mask[polygon_map]
        polygon_map = np.where(mask, polygon_map, -1)
    return polygon_map


def group_connected(polygon_map, mask=None):
    """Group all connected nodes."""
    # Wrap :c:`group_connected()` from ``polygon_map.c``.

    polygon_map = mask_polygon_map(polygon_map, mask)
    queue = Queue(len(polygon_map) + 1)

    group_ids = np.full(len(polygon_map), -1, np.intp, order="C")

    groups_count: int
    groups_count = slug.dll.group_connected(ptr(polygon_map),
                                            ptr(polygon_map.ctypes.shape),
                                            ptr(group_ids), queue._raw._ptr)

    return group_ids, groups_count
