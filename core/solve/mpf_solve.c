#include "mpf.h"

void mpf_batch_solve_alloc
(
  MPF_Solver *solver
)
{
  printf("in mpf_batch_solve_alloc --- \n");
  if (solver->framework == MPF_SOLVER_FRAME_MPF)
  {
    mpf_dense_alloc(&solver->B);
    mpf_dense_alloc(&solver->X);
  }

  if (solver->recon_target == MPF_DIAG_FA)
  {
    solver->buffer = NULL;
  }
  else if (solver->recon_target == MPF_SP_FA)
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->buffer = mpf_malloc(sizeof(double)*solver->B.m*2); 
    }
    else if (solver->data_type == MPF_COMPLEX)
    {
      printf("alloc MPF_COMPLEX\n");
      solver->buffer = mpf_malloc(sizeof(MPF_ComplexDouble)*solver->B.m*2); 
    }
  }
}

void mpf_fA_alloc
(
  MPF_Context *context
)
{
  if (context->solver.recon_target == MPF_DIAG_FA)
  {
    mpf_dense_alloc(&context->diag_fA);
  }
  else if (context->solver.recon_target == MPF_SP_FA)
  {
    /* get nz */
    ((MPF_Sparse*)context->fA_out)->nz = mpf_spai_get_nz(&context->A, context->solver.blk_fA,
      (MPF_Int*)context->probe.buffer, &((MPF_Int*)context->probe.buffer)[context->A.m]);
    mpf_sparse_csr_alloc(&context->fA);
  }
}

void mpf_batch_solve_free
(
  MPF_Solver *solver
)
{
  //if (solver->recon_target == MPF_DIAG_FA)
  //{
  //  mpf_dense_free(&solver->diag_fA);
  //}
  //else if (solver->recon_target == MPF_SP_FA)
  //{
  //  mpf_sparse_csr_free(&solver->fA);
  //}

  if (solver->framework == MPF_SOLVER_FRAME_MPF)
  {
    mpf_dense_free(&solver->B);
    mpf_dense_free(&solver->X);
    solver->buffer = NULL;
  }
}

void mpf_solver_alloc
(
  MPF_Solver *solver
)
{
  /* outer solver allocation */
  solver->outer_alloc_function(solver);

  /* inner solver allocation */
  solver->inner_alloc_function(solver);

  /* preconditioner */
  if (solver->use_precond)
  {
    //solver->precond_alloc_function(solver);
  }

  /* deflation */
  if (solver->use_defl)
  {
    //solver->defl_alloc_function(solver);
  }
}

void mpf_solver_free
(
  MPF_Solver *solver
)
{
  /* outer solver */
  if (solver->outer_free_function != NULL)
  {
    solver->outer_free_function(solver);
  }

  /* inner solver */
  if (solver->inner_free_function != NULL)
  {
    solver->inner_free_function(solver);
  }

  /* preconditioner */
  if (solver->use_precond)
  {
    //solver->precond_free_function(solver);
  }

  /* deflation */
  if (solver->use_defl)
  {
    //solver->defl_free_function(solver);
  }
}

void mpf_batch_init
(
  MPF_ContextHandle context,
  MPF_Int blk,
  MPF_Int nthreads_outer,
  MPF_Int nthreads_inner
)
{
  MPF_Solver* solver = &context->solver;
  MPF_Int ld = context->A.m;
  MPF_DataType data_type = context->A.data_type;

  solver->outer_type = MPF_SOLVER_BATCH;
  solver->outer_function = &mpf_batch_solve;
  solver->batch = blk;
  solver->outer_nthreads = nthreads_outer;
  solver->inner_nthreads = nthreads_inner;
  solver->use_inner = 1;
  solver->data_type = data_type;
  solver->ld = ld;

  solver->B.m = solver->ld;
  solver->B.n = solver->batch;
  solver->X.m = solver->ld;
  solver->X.n = solver->batch;

  solver->precond_type = MPF_PRECOND_NONE;
  solver->precond_update = MPF_PRECOND_STATIC;
  solver->defl_type = MPF_DEFL_NONE;

  if (solver->data_type == MPF_REAL)
  {
    solver->B.data_type = MPF_REAL;
    solver->X.data_type = MPF_REAL;

    if (solver->recon_target == MPF_DIAG_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_preprocess_diag;
      solver->generate_rhs_function = &mpf_d_generate_B;
      solver->generate_initial_solution_function = &mpf_d_zeros;
      solver->reconstruct_function = &mpf_d_reconstruct;
      solver->post_process_function = &mpf_diag_d_sym2gen;
      solver->outer_free_function = &mpf_batch_solve_free;
    }
    else if (solver->recon_target == MPF_SP_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_spai_init;
      solver->generate_rhs_function = &mpf_d_generate_B;
      solver->generate_initial_solution_function = &mpf_d_zeros;
      solver->reconstruct_function = &mpf_spai_d_reconstruct;
      solver->post_process_function = &mpf_sparse_csr_d_order;
      solver->outer_free_function = &mpf_batch_solve_free;
    }
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->B.data_type = MPF_COMPLEX;
    solver->X.data_type = MPF_COMPLEX;

    if (solver->recon_target == MPF_DIAG_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_preprocess_diag;
      solver->generate_rhs_function = &mpf_z_generate_B;
      solver->generate_initial_solution_function = &mpf_z_zeros;
      solver->reconstruct_function = &mpf_z_reconstruct;
      solver->post_process_function = &mpf_diag_zhe_sym2gen;
      solver->outer_free_function = &mpf_batch_solve_free;
    }
    else if (solver->recon_target == MPF_SP_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_spai_init;
      solver->generate_rhs_function = &mpf_z_generate_B;
      solver->generate_initial_solution_function = &mpf_z_zeros;
      solver->reconstruct_function = &mpf_spai_zhe_reconstruct;
      solver->post_process_function = &mpf_sparse_csr_z_order;
      solver->outer_free_function = &mpf_batch_solve_free;
    }

    /* assign preconditining functions */

    switch (solver->precond_type)
    {
      case MPF_PRECOND_BLK_DIAG:
        //solver->precond_alloc_function = mpf_precond_dsy_blkdiag_alloc;
        solver->precond_apply_function = mpf_sparse_d_mm_wrapper;
        //solver->precond_init_function = mpf_precond_blkdiag_init;
        //solver->precond_update_function = mpf_diagblk_dsy_reconstruct;
        //solver->precond_free_function = mpf_dsy_blkdiag_free;
        break;

      case MPF_PRECOND_SPAI:
        //solver->precond_alloc_function = ;
        //solver->precond_apply_function = mpf_sparse_d_mm_wrapper;
        //solver->precond_init_function = ;
        //solver->precond_update_function = ;
        //solver->precond_free_function = ;
        break;

      case MPF_PRECOND_NONE:
        break; 
    }
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->B.data_type = MPF_COMPLEX;
    solver->X.data_type = MPF_COMPLEX;

    if (solver->recon_target == MPF_DIAG_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_preprocess_diag;
      solver->generate_rhs_function = &mpf_z_generate_B;
      solver->generate_initial_solution_function = &mpf_z_zeros;
      solver->reconstruct_function = &mpf_z_reconstruct;
      solver->post_process_function = &mpf_diag_zsy_sym2gen;
      solver->outer_free_function = &mpf_batch_solve_free;
    }
    else if (solver->recon_target == MPF_SP_FA)
    {
      solver->outer_alloc_function = &mpf_batch_solve_alloc;
      solver->pre_process_function = &mpf_spai_init;
      solver->generate_rhs_function = &mpf_z_generate_B;
      solver->generate_initial_solution_function = &mpf_z_zeros;
      solver->reconstruct_function = &mpf_spai_zsy_reconstruct;
      solver->post_process_function = &mpf_sparse_csr_z_order;
      solver->outer_free_function = &mpf_batch_solve_free;
    }

    /* assign preconditining functions */

    switch (solver->precond_type)
    {
      case MPF_PRECOND_BLK_DIAG:
        //solver->precond_alloc_function = ;
        //solver->precond_apply_function = ;
        //solver->precond_init_function = ;
        //solver->precond_update_function = ;
        //solver->precond_free_function = ;
        break;

      case MPF_PRECOND_SPAI:
        //solver->precond_alloc_function = ;
        //solver->precond_apply_function = ;
        //solver->precond_init_function = ;
        //solver->precond_update_function = ;
        //solver->precond_free_function = ;
        break;

      case MPF_PRECOND_NONE:
        break; 
    }
  } 

  if (context->solver.framework == MPF_SOLVER_FRAME_MPF)
  {
    mkl_set_num_threads(context->solver.inner_nthreads);
  }
  context->args.n_outer_solve = 4;
}

/* 2 pass parallel version */
void mpf_batch_2pass_init
(
  MPF_Solver *solver,
  MPF_Int blk
)
{
  solver->outer_type = MPF_SOLVER_BATCH_2PASS;
  //solver->outer_function = &mpf_batch_2pass_solve;
}

void mpf_batch_spbasis_init
(
  MPF_Solver *solver
)
{
  //solver->outer_type = MPF_BATCH;
  //solver->outer_function = &mpf_batch_spbasis;
}

//MPF_SolverOuter_Pthreads *mpf_outer_pthreads_array_create
//(
//  MPF_Context *context
//)
//{
//  MPF_Int i = 0;
//  MPF_Int n_threads = context->n_threads_solver;
//  MPF_SolverOuter_Pthreads *context_pthreads_array
//    = (MPF_SolverOuter_Pthreads *) mpf_malloc(sizeof(MPF_ContextPthreads)*n_threads);
//  for (i = 0; i < context->n_threads_solver; i++)
//  {
//      context_pthreads_array[i].shared_context = context;
//  }
//  return context_pthreads_array;
//}
//
//void mpf_outer_pthreads_array_destroy
//(
//  MPF_SolverOuter_Pthreads *context_pthreads_array
//)
//{
//  MPF_Int i = 0;
//  MPF_Int n_threads = context_pthreads_array[0].shared_context->n_threads_solver;
//  for (i = 0; i < n_threads; ++i)
//  {
//    context_pthreads_array[i].shared_context = NULL;
//    context_pthreads_array[i].thread_id = 0;
//  }
//  mpf_free(context_pthreads_array);
//}
//
void mpf_krylov_free
(
  MPF_Solver *solver
)
{
  if (solver->inner_mem == NULL)
  {
    mpf_free(solver->inner_mem);
    solver->inner_mem = NULL;
  }
}

void mpf_preprocess_diag
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  mpf_get_max_nrhs(probe, solver);
  solver->n_batches = (solver->n_max_B+solver->batch-1) / solver->batch;
  mpf_d_zeros((MPF_Dense*)solver->fA);
}

/* sort nonzero entries in fA (edges) */
void mpf_sparse_csr_d_order
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *A_in
)
{
  MPF_Sparse *A = (MPF_Sparse*)A_in;
  MPF_Int nz_max = mpf_sparse_csr_get_max_row_nz(A);

  MPF_HeapMin_Fibonacci T;
  mpf_heap_min_fibonacci_init(&T, nz_max, nz_max);
  mpf_heap_min_fibonacci_internal_alloc(&T);

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    MPF_Int start = A->mem.csr.rs[i];
    MPF_Int end = A->mem.csr.re[i]-1;
    mpf_d_id_heapsort(&T, end-start+1, &A->mem.csr.cols[start],
      &((double*)A->mem.csr.data)[start], (double*)solver->B.data);
  }
}

void mpf_sparse_csr_z_order
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *A_in
)
{
  MPF_Sparse *A = (MPF_Sparse*)A_in;
  MPF_Int nz_max = mpf_sparse_csr_get_max_row_nz(A);

  MPF_HeapMin_Fibonacci T;
  mpf_heap_min_fibonacci_init(&T, nz_max, nz_max);
  mpf_heap_min_fibonacci_internal_alloc(&T);

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    MPF_Int start = A->mem.csr.rs[i];
    MPF_Int end = A->mem.csr.re[i]-1;
    mpf_z_id_heapsort(&T, end-start+1, &A->mem.csr.cols[start],
      &((MPF_ComplexDouble*)A->mem.csr.data)[start], (MPF_ComplexDouble*)solver->B.data);
  }
}
template <typename T>
void mpf_inner_solve_wrapper
(
  MPF_Solver *solver,
  T A
)
{
  solver->inner_function(solver, A, &solver->B, &solver->X);
}

template <typename T>
void mpf_inner_solve_gko_wrapper
(
  MPF_Solver *solver,
  T A
  //gko::matrix::Csr<double> *A
)
{
  //gko::matrix::Csr<double> *A = (gko::matrix::Csr<double> *)A_in;
}

void mpf_dsy_precond_update
(
  MPF_Probe* probe,
  MPF_Solver* solver
)
{
  MPF_Int color = solver->current_rhs/solver->max_blk_fA;
  for (MPF_Int i = 0; i < solver->color_to_node_map.bins_size[color]; ++i)
  {
    for (MPF_Int j = 0; j < solver->max_blk_fA/solver->blk_fA; ++j)
    {
      for (MPF_Int k = 0; k < solver->blk_fA; ++k)
      {
        for (MPF_Int l = 0; l < solver->blk_fA; ++l)
        {
          
        }
      }
    }
  }
}

void mpf_blkdiag_dsy_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Sparse *P = solver->Pmask;
  MPF_Sparse *fA = (MPF_Sparse*)solver->fA;
  MPF_Int BLK_MAX_fA = solver->max_blk_fA;
  MPF_BucketArray *H = &solver->color_to_node_map;
  MPF_Int *buffer = (MPF_Int*)solver->buffer;
  MPF_Int *buffer_rev = &buffer[solver->ld];

  for (MPF_Int i = 0; i < solver->batch; ++i)
  {
    MPF_Int color_start = (i+solver->current_rhs)/BLK_MAX_fA;
    MPF_Int color_end = (i+solver->current_rhs+solver->current_batch-1)/BLK_MAX_fA;

    for (MPF_Int color = color_start; color <= color_end; ++color)
    {
      /* same-color nodes */
      for (MPF_Int node = H->bins_start[color]; node != -1; node = H->next[node])
      {
        MPF_Int r_prev = H->values[node];
        MPF_Int r_A = r_prev*BLK_MAX_fA+i;

        for (MPF_Int p = 0; p < solver->blk_fA; ++p)
        {
          /* testing now */
          solver->M.mem.csr.cols[solver->M.mem.csr.re[r_A]] = ((double*)solver->fA)[r_A+p];
          solver->M.mem.csr.re[r_A] += 1;
        }
      }
    }
  }
}
