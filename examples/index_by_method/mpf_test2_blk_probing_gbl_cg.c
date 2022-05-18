#include "mpf.h"

int main
(
  int argc, 
  char *argv[]
)
{
  /* create MPF_ContextHandle */
  MPF_Int blk_fA = atoi(argv[1]);
  MPF_ContextHandle context;
  mpf_context_create(&context, MPF_DIAG_FA, blk_fA);

  /* reads input matrix A */
  char* filename_A = argv[2];
  std::cout << "filename_A: " << filename_A << std::endl;
  mpf_read_A(context, filename_A);
  printf("A->m: %d, A->n: %d, A->nz: %d\n", context->A.m, context->A.n, context->A.nz);

  /* intitialize probing method */
  MPF_Int stride = atoi(argv[6]);
  MPF_Int nlevels = atoi(argv[7]);
  mpf_blocking_init(context, stride, nlevels);

  /* initialize outer solver */
  MPF_Int batch_size = atoi(argv[8]);
  MPF_Int nthreads_outer = atoi(argv[9]);
  MPF_Int nthreads_inner = atoi(argv[10]);
  mpf_batch_init(context, batch_size, nthreads_outer, nthreads_inner);

  /* intitialize solver */
  MPF_Int tolerance = atof(argv[11]);
  MPF_Int niters = atoi(argv[12]);
  mpf_gbl_cg_init(context, tolerance, niters);

  /* executes probing and solving stages */
  mpf_run(context);

  /* displays meta data from the execution step */
  mpf_printout(context);

  /* writes approximated fA in file (output) */
  char* filename_fA = argv[3];
  mpf_write_fA(context, filename_fA);

  /* saves metadata from the execution of mpf */
  char* filename_log = argv[4];
  char* filename_caller = argv[5];
  mpf_write_log(context, filename_log, filename_caller);

  /* destroy handle */
  mpf_context_destroy(context);

  return 0;
}
