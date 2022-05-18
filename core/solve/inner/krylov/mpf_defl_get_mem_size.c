#include "mpf.h"

void mp_defl_memory_get
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int n_blocks = solver->n_batches;

  solver->inner_bytes = n*solver->batch*solver->iterations*n_blocks /* Vdefl */
                + (solver->iterations+1)*solver->iterations*solver->batch*solver->batch  /* H */
                + solver->iterations /* reflectors_array */
                + n*solver->batch  /* T (temp) */
                + n*solver->batch*solver->iterations  /* M (temp) */
                + n*solver->iterations /* eigenvectors */
                + solver->iterations   /* eigenvalues */
                + 2*solver->iterations*2*solver->iterations  /* G */
                + 2*solver->iterations*2*solver->iterations  /* F */
                + 2*n*solver->iterations;  /* Z */

  printf("%d\n", n*solver->batch*solver->iterations*n_blocks);
  printf("%d\n", n*solver->batch);
  printf("%d\n", n*solver->batch);
  printf("%d\n", (solver->iterations+1)*solver->iterations*solver->batch*solver->batch);
}
