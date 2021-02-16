// -*- coding: utf-8 -*-

#include "queue.h"

void Q_append(Queue *queue, ptrdiff_t arg) {
  queue -> queue[queue -> append_index] = arg;
  queue -> reverse_queue[arg] = queue -> append_index;
  queue -> append_index += 1;
  queue -> append_index %= queue -> max_size;
}

ptrdiff_t Q_consume(Queue *queue) {
  ptrdiff_t arg = Q_consume_later(queue);
  queue -> reverse_queue[arg] = -1;
  return arg;
}

ptrdiff_t Q_consume_later(Queue *queue) {
  ptrdiff_t arg = queue -> queue[queue -> consume_index];
  queue -> consume_index += 1;
  queue -> consume_index %= queue -> max_size;
  return arg;
}

ptrdiff_t Q_contains(Queue *queue, ptrdiff_t arg) {
  return queue -> reverse_queue[arg] != -1;
}

ptrdiff_t Q_is_empty(Queue *queue) {
  return queue -> append_index == queue -> consume_index;
}

void Q_appends(Queue * queue, ptrdiff_t * args, ptrdiff_t len_args) {
  for (ptrdiff_t i = 0; i < len_args; i++) {
    Q_append(queue, args[i]);
  }
}

void Q_add(Queue * queue, ptrdiff_t arg) {
  if (! Q_contains(queue, arg)) Q_append(queue, arg);
}

ptrdiff_t Q_len(Queue * queue) {
  ptrdiff_t out = (queue -> append_index - queue -> consume_index);
  if (out < 0) out += queue -> max_size;
  return out;
}
