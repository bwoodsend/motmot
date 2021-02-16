// -*- coding: utf-8 -*-

#ifndef _QUEUE_H
#define _QUEUE_H

#include <stdio.h>
#include <stddef.h>

typedef struct Queue
{
  ptrdiff_t * queue;
  ptrdiff_t * reverse_queue;
  ptrdiff_t append_index;
  ptrdiff_t consume_index;
  ptrdiff_t max_size;
} Queue;

#endif

