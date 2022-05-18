#include "mpf.h"

void mpf_generate_sampling_endpoints
(
  MPF_Probe *probe
)
{
  MPF_Int lower = 0;
  MPF_Int upper = 1;

  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    upper *= 3;
  }

  MPF_Int new_endpoint = 0;
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int n_selected = 0;
  MPF_Int conflict = 0;

  while ((n_selected < probe->n_endpoints) && (i < upper))
  {
    new_endpoint = (rand() % (upper-lower+1));

    j = 0;
    conflict = 0;
    while ((j < n_selected) && (!conflict))
    {
      if (new_endpoint == probe->endpoints_array[j])
      {
        conflict = 1;
      }
      ++j;
    }

    if (!conflict)
    {
      probe->endpoints_array[i] = new_endpoint;
      ++n_selected;
    }
    ++i;
  }
}
