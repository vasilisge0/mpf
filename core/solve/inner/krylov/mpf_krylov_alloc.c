#include "mpf.h"

void mpf_krylov_alloc
(
  MPF_Solver *solver
)
{
  solver->inner_mem = mpf_malloc(solver->inner_bytes);
}
