#include "mpf.h"

void mpf_batch_solve
(
  MPF_Probe* probe,
  MPF_Solver* solver,
  MPF_Sparse* A,
  void* fA_out
)
{
  MPF_Dense *B = &solver->B;
  MPF_Dense *X = &solver->X;
  solver->fA = fA_out;

  struct timespec start;
  struct timespec finish;

  #if MPF_MEASURE
    solver->runtime_pre_process = 0.0;
    solver->runtime_alloc = 0.0;
    solver->runtime_generate_rhs = 0.0;
    solver->runtime_inner = 0.0;
    solver->runtime_reconstruct = 0.0;
    solver->runtime_post_process = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  solver->inner_get_mem_size_function(solver);
  solver->outer_alloc_function(solver);

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    solver->runtime_alloc += mpf_time(start, finish);
  #endif

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  solver->pre_process_function(probe, solver);

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    solver->runtime_pre_process += mpf_time(start, finish);
  #endif

  #if MPF_PRINTOUT
    if (solver->recon_target == MPF_SP_FA)
    {
      printf("       nz_fA: %d\n", ((MPF_Sparse*)fA_out)->nz);
    }
    printf("     framework: mpf\n");
    printf("           m_A: %d\n", (int)A->m);
    printf("       n_max_B: %d\n", (int)solver->n_max_B);
    printf("     n_batches: %d\n", (int)solver->n_batches);
    printf("    blk_solver: %d\n", (int)solver->batch);
    printf("      n_colors: %d\n", (int)probe->n_colors);
    printf("    max_blk_fA: %d\n", (int)solver->max_blk_fA);
    printf("   bytes_inner: %d\n", (int)solver->inner_bytes);
    printf("nthreads_inner: %d\n", (int)solver->inner_nthreads);
    printf("nthreads_outer: %d\n", (int)solver->outer_nthreads);
  #endif

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  solver->inner_alloc_function(solver);

  if (solver->precond_type != MPF_PRECOND_NONE)
  {
    solver->precond_alloc_function(solver);
    solver->precond_generate_function(solver, A);
  }

  if (solver->defl_type != MPF_DEFL_NONE)
  {
    solver->defl_alloc_function(solver);
  }

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    solver->runtime_alloc += mpf_time(start, finish);
  #endif

  for (MPF_Int i = 0; i < solver->n_batches; ++i)
  {
    /* updates current rhs and block_size */

    solver->current_rhs = solver->batch*i;
    solver->current_batch = (1-(i+1)/(solver->n_batches))*solver->batch+((i+1)
      /(solver->n_batches))*(solver->n_max_B-solver->current_rhs);
    B->n = solver->current_batch;

    /* generates B and X */

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    solver->generate_rhs_function(probe, solver, B);
    solver->generate_initial_solution_function(X);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      solver->runtime_generate_rhs = mpf_time(start, finish);
    #endif

    /* solves AX = B */

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    solver->inner_function(solver, A, B, X);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      solver->runtime_inner += mpf_time(start, finish);
    #endif

    /* reconstructs entries of diag_fA */

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    solver->reconstruct_function(probe, solver);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      solver->runtime_reconstruct += mpf_time(start, finish);
    #endif

    /* updates preconditioner */

    if ((solver->precond_type != MPF_PRECOND_NONE) &&
        (solver->precond_update == MPF_PRECOND_ADAPTIVE))
    {
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif

      solver->precond_update_function(probe, solver);

      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        solver->runtime_precond += mpf_time(start, finish);
      #endif
    }

    /* updates basis used for deflation */

    if (solver->defl_type != MPF_DEFL_NONE)
    {
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif

      solver->defl_update_function(solver);

      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        solver->runtime_defl += mpf_time(start, finish);
      #endif
    }
  }

  /* converts matrix from symmetric to general format */

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  solver->post_process_function(probe, solver, fA_out);

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    solver->runtime_post_process += mpf_time(start, finish);
  #endif

  /* free preconditioner */

  if (solver->precond_type != MPF_PRECOND_NONE)
  {
    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    solver->precond_free_function(solver);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      solver->runtime_alloc += mpf_time(start, finish);
    #endif
  }

  /* free deflation basis */

  if (solver->defl_type != MPF_DEFL_NONE)
  {
    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    solver->defl_free_function(solver);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      solver->runtime_alloc += mpf_time(start, finish);
    #endif
  }

  /* compute total runtime */

  #if MPF_MEASURE
    solver->runtime_total =
        solver->runtime_total
      + solver->runtime_pre_process
      + solver->runtime_generate_rhs
      + solver->runtime_inner
      + solver->runtime_select
      + solver->runtime_reconstruct
      + solver->runtime_post_process;
  #endif

  solver->fA = NULL;
}

//commented to compile without cuda

//#include "cusparse_v2.h"
//#include "cublas_v2.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
void mpf_batch_gko_solve
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Args *args,
  void *fA_out
)
{
//  MPF_Dense *B = &solver->B;
//  MPF_Dense *X = &solver->X;
//  solver->fA = fA_out;
//
//  using vecblk = gko::matrix::Dense<double>;
//    // for omp executor
//    //solver->exec = gko::OmpExecutor::create();
//    //auto B_gko = vecblk::create(solver->exec, gko::dim<2>(solver->ld, solver->batch));
//    //auto X_gko = vecblk::create(solver->exec, gko::dim<2>(solver->ld, solver->batch));
//    //solver->B.data = B_gko->get_values();
//    //solver->X.data = X_gko->get_values();
//
//  // for cuda executor
//  auto omp_exec = gko::OmpExecutor::create();
//  solver->exec = gko::CudaExecutor::create(0, omp_exec);
//  auto B_gko = vecblk::create(omp_exec, gko::dim<2>(solver->ld, solver->batch));
//  auto X_gko = vecblk::create(omp_exec, gko::dim<2>(solver->ld, solver->batch));
//  solver->B.data = B_gko->get_values();
//  solver->X.data = X_gko->get_values();
//
//  auto d_B_gko = vecblk::create(solver->exec, gko::dim<2>(solver->ld, solver->batch));
//  auto d_X_gko = vecblk::create(solver->exec, gko::dim<2>(solver->ld, solver->batch));
//
//  using mtx = gko::matrix::Csr<double>;
//
//  struct timespec start;
//  struct timespec finish;
//
//  #if MPF_MEASURE
//    solver->runtime_pre_process = 0.0;
//    solver->runtime_alloc = 0.0;
//    solver->runtime_generate_rhs = 0.0;
//    solver->runtime_inner = 0.0;
//    solver->runtime_reconstruct = 0.0;
//    solver->runtime_post_process = 0.0;
//
//    clock_gettime(CLOCK_MONOTONIC, &start);
//  #endif
//
//  /* this should be replaced */
//  printf("reading A from file...\n");
//  auto A_gko = share(gko::read<mtx>(std::ifstream(args->filename_A), solver->exec));
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    solver->runtime_other = mpf_time(start, finish);
//    solver->runtime_total += solver->runtime_other;
//    printf("...finished in %1.1E seconds\n", solver->runtime_other);
//  #endif
//
//  /* solver factory */
//  using cg = gko::solver::Cg<>;
//
//  auto solver_factory = cg::build()
//   .with_criteria(
//      gko::stop::Iteration::build()
//          .with_max_iters(solver->iterations)
//          .on(solver->exec),
//      gko::stop::ResidualNormReduction<>::build()
//          .with_reduction_factor(1e-8)
//          .on(solver->exec))
//   .on(solver->exec);
//
//  /* generate solver */
//  auto solver_gko = solver_factory->generate(A_gko);
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &start);
//  #endif
//
//  solver->outer_alloc_function(solver);
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    solver->runtime_alloc += mpf_time(start, finish);
//  #endif
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &start);
//  #endif
//
//  solver->pre_process_function(probe, solver);
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    solver->runtime_pre_process += mpf_time(start, finish);
//  #endif
//
//  #if MPF_PRINTOUT
//    if (solver->recon_target == MPF_SP_FA)
//    {
//      printf("       nz_fA: %d\n", ((MPF_Sparse*)fA_out)->nz);
//    }
//    printf("   framework: gko\n");
//    printf("         m_A: %d\n", (int)A->m);
//    printf("     n_max_B: %d\n", (int)solver->n_max_B);
//    printf("    n_blocks: %d\n", (int)solver->n_batches);
//    printf("  blk_solver: %d\n", (int)solver->batch);
//    printf("    n_colors: %d\n", (int)probe->n_colors);
//    printf("  max_blk_fA: %d\n", (int)solver->max_blk_fA);
//    printf(" bytes_inner: %d\n", (int)solver->inner_bytes);
//  #endif
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &start);
//  #endif
//
//  //solver->inner_alloc_function(solver);
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    solver->runtime_alloc += mpf_time(start, finish);
//  #endif
//
//  #if MPF_PRINTOUT
//    printf("\nouter iterations (serial): \n");
//  #endif
//
//  for (MPF_Int i = 0; i < solver->n_batches; ++i)
//  {
//    #if DEBUG
//      printf("  %d/%d\n", i, solver->n_batches);
//    #endif
//
//    /* updates current rhs and block_size */
//
//    solver->current_rhs = solver->batch*i;
//    solver->current_batch = (1-(i+1)/(solver->n_batches))*solver->batch+((i+1)
//      /(solver->n_batches))*(solver->n_max_B-solver->current_rhs);
//    B->n = solver->current_batch;
//
//    /* generates B and X */
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &start);
//    #endif
//
//      // for ompexecutor
//      //solver->generate_rhs_function(probe, solver, B);
//      //solver->generate_initial_solution_function(X);
//
//    solver->exec->copy_from(lend(omp_exec), B->m * B->n, (double*)B->data, d_B_gko->get_values());
//    solver->exec->copy_from(lend(omp_exec), B->m * B->n, (double*)X->data, d_X_gko->get_values());
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &finish);
//      solver->runtime_generate_rhs = mpf_time(start, finish);
//    #endif
//
//    /* solves AX = B */
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &start);
//    #endif
//
//    /* solve system */
//    //solver_gko->apply(gko::lend(d_B_gko), gko::lend(d_X_gko));  /* for cuda executor */
//    solver_gko->apply(gko::lend(B_gko), gko::lend(X_gko));  /* for omp executor */
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &finish);
//      solver->runtime_inner += mpf_time(start, finish);
//    #endif
//
//    /* reconstructs entries of diag_fA */
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &start);
//    #endif
//
//    omp_exec->copy_from<double>(lend(solver->exec), (int)(B->m * B->n), (double*)d_B_gko->get_values(), (double*)B->data);
//    omp_exec->copy_from<double>(lend(solver->exec), (int)(B->m * B->n), (double*)d_X_gko->get_values(), (double*)X->data);
//    solver->exec->synchronize();
//
//      //for ompexecutor
//      //solver->reconstruct_function(probe, solver);
//
//    #if MPF_MEASURE
//      clock_gettime(CLOCK_MONOTONIC, &finish);
//      solver->runtime_reconstruct += mpf_time(start, finish);
//    #endif
//  }
//
//  /* converts diagonal blocks from symmetric to general format */
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &start);
//  #endif
//
//  solver->post_process_function(probe, solver, fA_out);
//
//  #if MPF_MEASURE
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    solver->runtime_post_process += mpf_time(start, finish);
//
//    solver->runtime_total =
//        solver->runtime_total
//      + solver->runtime_pre_process
//      + solver->runtime_generate_rhs
//      + solver->runtime_inner
//      + solver->runtime_select
//      + solver->runtime_reconstruct
//      + solver->runtime_post_process;
//  #endif
//
//  #if DEBUG
//    if (fA_out->data_type == MPF_COMPLEX)
//    {
//      mpf_matrix_d_announce(((MPF_Dense*)fA_out)->data, 2, 2,
//        ((MPF_Dense*)fA_out)->m, "fA->data");
//    }
//    else if (fA_out->data_type == MPF_COMPLEX)
//    {
//      mpf_matrix_z_announce(((MPF_Dense*)fA_out)->data, 2, 2,
//        ((MPF_Dense*)fA_out)->m, "fA->data");
//    }
//  #endif
//
//  solver->fA = NULL;
}
