#include "mpf.h"

void mpf_dsy_gbl_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  double B_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', m_B, blk, B, m_B);
  double r_norm = 1.0;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* memory cpu*/
  double *Rold = (double*)solver->inner_mem;
  double *Rnew = &Rold[m_B*blk];
  double *Dvec = &Rnew[m_B*blk];
  double *Dvec_temp = &Dvec[m_B*blk];
  double *Rtemp = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold) * m_B * blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle,
    A->descr, sparse_layout, X, blk, m_B, 1.0, Rold, m_B);
  r_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rold, m_B)/B_norm;
  memcpy(Dvec, Rold, (sizeof *Dvec) * m_B * blk);

  /* main loop */
  while ((i < solver->iterations) && (r_norm > solver->tolerance))
  {
    /* computes alpha and gamma scalars */
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      sparse_layout, Dvec, blk, m_B, 0.0, Dvec_temp, m_B);

    trace_r_old = 0.0;
    gamma = 0.0;
    for (j = 0; j < blk; ++j)
    {
      gamma += mpf_ddot(m_B, &Dvec_temp[m_B*j], 1, &Dvec[m_B*j], 1);
      trace_r_old += mpf_ddot(m_B, &Rold[m_B*j], 1, &Rold[m_B*j], 1);
    }
    alpha = trace_r_old/gamma;

    /* updates X (solution) and residual vecblk */
    mpf_daxpy(m_B*blk, alpha, Dvec, 1, X, 1);
    memcpy(Rnew, Rold, (sizeof *Rnew)*m_B*blk);
    mpf_daxpy(m_B*blk, -alpha, Dvec_temp, 1, Rnew, 1);

    /* computes beta scalar */
    trace_r_new = 0.0;
    for (j = 0; j < blk; ++j)
    {
      trace_r_new += mpf_ddot(m_B, &Rnew[m_B*j], 1, &Rnew[m_B*j], 1);
    }
    beta = trace_r_new/trace_r_old;
    /* updates direction vecblk */
    mpf_dscal(m_B*blk, beta, Dvec, 1);
    mpf_daxpy(m_B*blk, 1.0, Rnew, 1, Dvec, 1);
    r_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rnew, m_B)/B_norm;

    /* swaps old with new residual vecblks */
    Rtemp = Rnew;
    Rnew = Rold;
    Rold = Rtemp;
    i += 1;
  }
}

void mpf_gbl_cg_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations
)
{
  MPF_Solver* solver = &context->solver;
  MPF_Int ld = context->A.m;
  solver->ld = ld;
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;
  solver->framework = MPF_SOLVER_FRAME_MPF;
  context->args.n_inner_solve = 3;

  if ((solver->precond_type == MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DSY_CG;
      solver->device = MPF_DEVICE_CPU;
      solver->inner_function = &mpf_dsy_gbl_cg;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_CG;
      //solver->inner_function = &mpf_zsy_gbl_cg;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZHE_CG;
      //solver->inner_function = &mpf_zhe_gbl_cg;
      solver->device = MPF_DEVICE_CPU;
    }
  }
  else if ((solver->precond_type != MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DSY_CG;
      solver->device = MPF_DEVICE_CPU;
        solver->inner_function = &mpf_dsy_pcg;

      mpf_d_precond_init(solver);
    }
  }

  solver->inner_alloc_function = &mpf_cg_alloc;
  solver->inner_free_function = &mpf_krylov_free;
  solver->inner_get_mem_size_function = &mpf_gbl_cg_get_mem_size;
}

void mpf_gbl_cg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
        (n*blk*2  /* size_B and size_X */
        +n*blk    /* size_residual_old */
        +n*blk    /* size_residual_new */
        +n*blk    /* size_direction */
        +n*blk);  /* size_direction_new */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
        (n*blk*2  /* size_B and size_X */
        +n*blk    /* size_residual_old */
        +n*blk    /* size_residual_new */
        +n*blk    /* size_direction */
        +n*blk);  /* size_direction_new */
  }
  else if (solver->data_type == MPF_COMPLEX_32)
  {
    solver->inner_bytes = sizeof(MPF_Complex)*
        (n*blk*2  /* size_B and size_X */
        +n*blk    /* size_residual_old */
        +n*blk    /* size_residual_new */
        +n*blk    /* size_direction */
        +n*blk);  /* size_direction_new */
  }
}
