#include "mpf.h"

/* ------------------ functions for mapping colors to rhs ------------------- */

void mpf_color_to_node_map_alloc
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  mpf_bucket_array_init(&solver->color_to_node_map, probe->n_nodes,
    probe->n_colors);
  mpf_bucket_array_alloc(&solver->color_to_node_map);
  mpf_bucket_array_values_init(&solver->color_to_node_map);
}

void mpf_color_to_node_map_set
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  for (MPF_Int i = 0; i < probe->P.m; ++i)
  {
    mpf_bucket_array_insert(&solver->color_to_node_map,
      probe->colorings_array[i], i);
  }
  mpf_bucket_array_find_max_bin_size(&solver->color_to_node_map);
}
