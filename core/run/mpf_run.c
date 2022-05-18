#include "mpf.h"

/*--------------------------- multilevel probing -----------------------------*/

/*============================================================================*/
/* mp: methods for approximating the dfagonal of the matrix inverse using     */
/* multilevel probing methods                                                 */
/*============================================================================*/
void mpf_run
(
  MPF_ContextHandle context
)
{
  printf("\n\n");
  printf("=========================== MPF ==============================\n");

  for (context->probe.current_iteration = 0; context->probe.current_iteration < context->probe.iterations; ++context->probe.current_iteration)
  {
    /* generates probing vectors */

    printf(">> initializing probing function\n");
    context->probe.alloc_function(&context->probe);

    if (context->probe.iterations > 1)
    {
      mpf_generate_sampling_endpoints(&context->probe);
    }

    printf(">> approximating significant structure of f(A)\n");
    context->probe.find_pattern_function(&context->probe, &context->A);

    if ((context->probe.type != MPF_PROBE_BATCH_BLOCKING) &&
       (context->probe.type != MPF_PROBE_BATCH_COMPACT_BLOCKING))
    {
      printf(">> coloring graph with S(T(f(A))) as adj. matrix\n");
      context->probe.color_function(&context->probe);
    }

    mpf_color_to_node_map_alloc(&context->probe, &context->solver);
    mpf_color_to_node_map_set(&context->probe, &context->solver);

    /* Solves linear system AW = V */

    if (context->probe.n_colors > 0)
    {
      printf(">> allocating memory for fA\n");
      context->fA_alloc_function(context);
      mpf_probe_free(&context->probe);

      printf(">> solving AW = V\n");
      if (context->solver.framework == MPF_SOLVER_FRAME_MPF)
      {
        context->solver.outer_function(&context->probe, &context->solver,
         &context->A, context->fA_out);
      }
      else if (context->solver.framework == MPF_SOLVER_FRAME_GKO)
      {
        mpf_batch_gko_solve(&context->probe, &context->solver,
         &context->A, &context->args, context->fA_out);
      }
    }
    else
    {
      printf("[mp] Error: probing did not produce any right-hand sides. \
        Exiting\n");
    }
  }

  if (context->probe.iterations > 1)
  {
    mpf_diag_fA_average(context);
  }

  /* == refinement == */

  //context->solver.rufine_function();

  ////mkl_sparse_destroy(context->P_handle);
  ////mpf_free(context->P);
  //
  //  /* solve AX = I with sparse_spmm operations using preconditioner produced */
  //  /* from previous step                                                     */
  //  //if ()
  //  //{
  //
  //  //temporary
  //  printf("[refinement]>>...\n");
  //  printf("X[0]: %1.8E\n", ((double*)context->diag_fA)[0]);
  //  mpf_d_sparse_solve(context->meta_solver.krylov, context->m_B,
  //    &context->A.csr, context->A_handle, context->mem_solver, &mpf_dsy_sparse_cg);
  //  printf("...<<[refinement]\n");
  //  //}
  //
  //context->runtime.total = context->runtime.probe
  //  + context->solver.runtime.total. + context->runtime.alloc;
}
