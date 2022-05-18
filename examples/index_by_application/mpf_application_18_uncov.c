#include "mpf.h"

int main
(
  int argc, 
  char *argv[]
)
{
  /* create a handle for MPF_Context */
  MPF_Int blk_fA = atoi(argv[1]);
  MPF_ContextHandle context;
  mpf_context_create(&context, MPF_SP_FA, blk_fA);

  /* reads input matrix A */
  char* filename_A = argv[2];
  mpf_read_A(context, filename_A);
  std::cout << "filename_A: " << filename_A << std::endl;
  printf("A->m: %d, A->n: %d, A->nz: %d\n", context->A.m, context->A.n,
    context->A.nz);

  /* intitializes probing method */
  MPF_Int stride = atoi(argv[6]);
  MPF_Int nlevels = atoi(argv[7]);
  mpf_blocking_init(context, stride, nlevels);

  /* initializes outer solver */
  MPF_Int batch_size = atoi(argv[8]);
  MPF_Int nthreads_outer = atoi(argv[9]);
  MPF_Int nthreads_inner = atoi(argv[10]);
  mpf_batch_init(context, batch_size, nthreads_outer, nthreads_inner);

  /* intitializes inner solver */
  double tolerance = atof(argv[11]);
  MPF_Int niters = atoi(argv[12]);
  printf("nthreads_outer: %d, nthreads_inner: %d\n", nthreads_outer,
    nthreads_inner);
  printf("tolerance: %1.2E, niter: %d\n", tolerance, niters);
  mpf_cg_init(context, tolerance, niters);

  /* initializes jacobi preconditioner */
  mpf_jacobi_precond_init(context);

  /* runs the probing and solving stages */
  mpf_run(context);

  /* displays log from mpf_run() */
  mpf_printout(context);

  /* writes fA */
  char* filename_fA = argv[3];
  mpf_write_fA(context, filename_fA);

  /* writes execution log for mpf_run() */
  char* filename_log = argv[4];
  char* filename_caller = argv[5];
  mpf_write_log(context, filename_log, filename_caller);

  /* destroys handle */
  mpf_context_destroy(context);

  return 0;
}

