#include "mpf.h"

MPF_Int mp_blk_max_fA_get
(
  MPF_Context *context
)
{
  if (context->probe.type == MPF_PROBE_BLOCKING)
  {
    return (MPF_Int) (pow((double)(context->probe.stride),
      (double)(context->probe.n_levels))+0.5);
  }
  else if (context->probe.type == MPF_PROBE_SAMPLING)
  {
    return (MPF_Int) (pow((double)(context->probe.stride),
      (double) (context->probe.n_levels))+0.5);
  }
  else if (context->probe.type == MPF_PROBE_PATH_SAMPLING)
  {
    return (MPF_Int) (pow((double)(context->probe.stride),
      (double) (context->probe.n_levels))+0.5);
  }
  else if (context->probe.type == MPF_PROBE_AVG_PATH_SAMPLING)
  {
    return (MPF_Int) (pow((double) (context->probe.stride),
      (double)(context->probe.n_levels))+0.5);
  }
  else if (context->probe.type == MPF_PROBE_BATCH_BLOCKING)
  {
    return (MPF_Int) (pow((double)(context->probe.stride),
      (double)(context->probe.n_levels))+0.5);
  }
  else if (context->probe.type == MPF_PROBE_BATCH_COMPACT_BLOCKING)
  {
    return (MPF_Int) (pow((double)(context->probe.stride),
      (double)(context->probe.n_levels))+0.5);
  }
  return 0;
}
