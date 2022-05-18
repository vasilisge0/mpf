#include "mpf.h"

void mpf_blocking_hybrid_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
)
{
  context->probe.max_blk = (MPF_Int)pow((double)stride, (double)n_levels);
  context->probe.type = MPF_PROBE_BLOCKING;
  context->probe.stride = stride;
  context->probe.n_levels = n_levels;
  context->probe.iterations = 1;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.find_pattern_function = mpf_blocking_hybrid;
  context->probe.color_function = mpf_color;
  context->probe.alloc_function = mpf_probe_alloc;
}
