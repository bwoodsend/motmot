# -*- coding: utf-8 -*-
"""
"""

from cslug import CSlug, anchor, Header

queue_h = Header(*anchor("queue.h", "queue.c"), includes="_queue.h")
slug = CSlug(anchor("motmot", "queue.c", "_queue.h", "polygon_map.c"),
             headers=queue_h)
