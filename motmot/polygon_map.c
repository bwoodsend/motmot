// -*- coding: utf-8 -*-

#include <stdio.h>
#include <stddef.h>


typedef struct RaggedArray
{
  void * flat;
  int itemsize;
  int length;
  int * starts;
  int * ends;
} RaggedArray;


void populate_polygon_map(ptrdiff_t * polygon_map, ptrdiff_t * ids,
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
      // We have to do some pointer arithmetic because ``ids`` is a 2D
      // array in Python but 1D in C.
      ptrdiff_t * ids_row = ids + (d * i);
      ptrdiff_t v0 = ids_row[j];
      ptrdiff_t v1 = ids_row[(j + 1) % d];

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
