// -*- coding: utf-8 -*-

#include <stdio.h>
#include <stddef.h>
#include "queue.h"


typedef struct RaggedArray
{
  void * flat;
  int itemsize;
  int length;
  int * starts;
  int * ends;
} RaggedArray;


void populate_polygon_map(ptrdiff_t * polygon_map, ptrdiff_t * faces,
                          int n, int d,
                          RaggedArray * dest_vertices, RaggedArray * polygons) {

  // The algorithm this belongs to is 2-part with the first half being written
  // in Python in ``_polygon_map.py``. See there.

  // Pull some variable getters out of the loop.
  ptrdiff_t * dest_vertices_flat = (ptrdiff_t *) dest_vertices -> flat;
  int * rag_starts = dest_vertices -> starts;
  int * rag_ends = dest_vertices -> ends;
  ptrdiff_t * polygons_flat = (ptrdiff_t *) polygons -> flat;

  // For each cell in the ``polygon_map``.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {

      // Get the edge to search for.
      // We have to do some pointer arithmetic because ``faces`` is a 2D
      // array in Python but 1D in C.
      ptrdiff_t * faces_row = faces + (d * i);
      ptrdiff_t v0 = faces_row[j];
      ptrdiff_t v1 = faces_row[(j + 1) % d];

      // Find, and default to -1, where we'll write the answer to.
      ptrdiff_t * neighbour = polygon_map + (d * i) + j;
      *neighbour = -1;

      // Search for an edge which starts at v1,
      for (int k = rag_starts[v1]; k < rag_ends[v1]; k++) {
        // and ends at v0.
        if (dest_vertices_flat[k] == v0) {
          *neighbour = polygons_flat[k];
          break;
        }
      }
    }
  }
}


ptrdiff_t ravel(ptrdiff_t i, ptrdiff_t j, ptrdiff_t d) {
  /* Convert 2D (n, d) array index to ravelled 1D index. */
  return i * d + j;
}


void connected(ptrdiff_t * initial, ptrdiff_t len_initial, ptrdiff_t d,
               ptrdiff_t * polygon_map, Queue * queue) {
/*
    Navigate `polygon_map` to create a connected region.

    If this function was recursive then the recursion depths required to get
    around a typical sized mesh would cause StackOverflowErrors.
    To avoid this, recursion is mimicked using a queue.
    This `queue` is just an integer array listing `arg`s in the order they are
    encountered, with a `reverse_queue` that lists an `arg`'s position in
    `queue` to facilitate `arg in queue` checking.

    In order to avoid having to inc-ref arrays, `queue` and `reverse_queue` are
    intended to be initialised in Python using numpy.

    An arg is only put into `queue` if it is to be included in the resulting
    region so an output array is not needed as Python can just look at the queue
    to see what was used.
 */

  // Add each `arg` from `args` to the queue.
  Q_appends(queue, initial, len_initial);

  // While there are unprocessed elements in the queue.
  while (!Q_is_empty(queue)) {

    // Read the oldest unprocessed value in the queue.
    ptrdiff_t arg = Q_consume_later(queue);

    // For each adjacent polygon:
    for (ptrdiff_t i = 0; i < d; i++) {

      // `polygon_map` in Python has shape `(n, d)` but is flattened by C.
      // The index must be ravelled to match.
      ptrdiff_t index = ravel(arg, i, d);

      // Lookup the arg of an adjacent polygon.
      ptrdiff_t arg_ = polygon_map[index];

      // Not allowed or missing adjacent polygons are masked by
      // -1s in `polygon_map`. These should be skipped.
      if (arg_ == -1) continue;

      // Finally Q_add `arg_` to the queue.
      // This is ignored if `arg_` is already in it.
      Q_add(queue, arg_);
    }
  }
}


ptrdiff_t group_connected(ptrdiff_t * polygon_map, ptrdiff_t * shape,
                          ptrdiff_t * group_ids, Queue * queue) {
  /* Split a disjointed mesh into connected sub-meshes. */

  // This works simply by calling connected() repeatedly until every polygon
  // has been allocated a group number.

  ptrdiff_t group_id = 0;
  for (ptrdiff_t polygon_id = 0; polygon_id < shape[0]; polygon_id++) {

    // Skip if this polygon is already in a group.
    if (group_ids[polygon_id] != -1) continue;

    // Find all polygons connected to `polygon_id`.
    ptrdiff_t start = queue -> append_index;
    connected(&polygon_id, 1, shape[1], polygon_map, queue);
    ptrdiff_t end = queue -> append_index;

    // Unpack the unusual output of connected().
    for (ptrdiff_t i = start; i != end; i = (i + 1) % queue -> max_size) {

      ptrdiff_t _arg = queue -> queue[i];
      group_ids[_arg] = group_id;
      // Reset this queue space.
      queue -> reverse_queue[_arg] = -1;
    }

    group_id ++;
  }
  return group_id;
}
