#include "mpf.h"

void mpf_solve_gbl_pcg_dsy
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;


  double *Rold = (double*)solver->inner_mem;
  double *Rnew = &Rold[m*blk];
  double *Dvec = &Rnew[m*blk];
  double *Dvec_temp = &Dvec[m*blk];
  double *M = &Dvec_temp[m*blk];
  //double *T = &M[m*blk];
  double *Rtemp = NULL;
  double B_norm = mpf_dlange(MPF_COL_MAJOR, 'F', m, blk, B, m);
  double r_norm = 1.0;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;

  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;
  mpf_convert_layout_to_sparse(layout, &sparse_layout);

  memcpy(Rold, B, (sizeof *Rold)*m*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, m, 1.0, Rold, m);

  memcpy(Dvec, Rold, (sizeof *Dvec)*m*blk);
//solver->precond_function(solver, m, Rold, solver->M);
  //mpf_sparse_d_mm_wrapper(solver-> m, Rold, M);

  MPF_Int i = 0;
  while ((i < solver->iterations) && (r_norm > solver->tolerance))
  {
    /* computes alpha, gamma */
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle,
      A->descr, sparse_layout, Dvec, blk, m, 0.0, Dvec_temp, m);

    trace_r_old = mpf_ddot(m*blk, Rold, 1, M, 1);
    gamma = mpf_ddot(m*blk, Dvec_temp, 1, Dvec, 1);
    alpha = trace_r_old/gamma;

    /* updates X and residual vecblk */
    mpf_daxpy(m*blk, alpha, Dvec, 1, X, 1);
    memcpy(Rnew, Rold, (sizeof *Rnew)*m*blk);
    mpf_daxpy(m*blk, -alpha, Dvec_temp, 1, Rnew, 1);

    /* applies preconditioning */
//solver->precond_function(solver, m, Rnew, solver->M);

    /* computes beta */
    //trace_r_new = mpf_ddot(m*blk, Rnew, 1, Rnew, 1);
    trace_r_new = mpf_ddot(m*blk, Rnew, 1, M, 1);
    beta = trace_r_new/trace_r_old;

    /* update direction vecblk */
    mpf_dscal(m*blk, beta, Dvec, 1);
    //mpf_daxpy(m*blk, 1.0, Rnew, 1, Dvec, 1);
    mpf_daxpy(m*blk, 1.0, M, 1, Dvec, 1);
    r_norm = mpf_dlange(CblasColMajor, 'F', m, blk, Rnew, m)/B_norm;
    //printf("i; %d, r_norm: %1.4E, X[0]: %1.4E, B_norm: %1.4E\n",
    //  i, r_norm, X[0], B_norm);
    //printf("X[0]: %1.4E\n", X[0]);

    /* swaps old with new residual vecblks*/
    Rtemp = Rnew;
    Rnew = Rold;
    Rold = Rtemp;
    i += 1;
  }
}

void mpf_gbl_pcg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  char *precond,
  char *filename
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;

  if (strcmp(precond, "spai") == 0)
  {
    solver->precond_type = MPF_PRECOND_SPAI;
    solver->precond_apply_function = &mpf_sparse_d_mm_wrapper;
  }
  else if (strcmp(precond, "jacobi") == 0)
  {
    solver->precond_type = MPF_PRECOND_JACOBI;
    solver->precond_apply_function = &mpf_sparse_d_mm_wrapper;
  }

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_GBL_PCG;
    //solver->inner_function = &mpf_dsy_gbl_cg;
    solver->inner_function = &mpf_solve_gbl_pcg_dsy;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_GBL_CG;
    //solver->inner_function = &mpf_zsy_gbl_cg;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_GBL_CG;
    //solver->inner_function = &mpf_zhe_gbl_cg;
  }
  //solver->inner_function = &mpf_solve_gbl_pcg_dsy;

  //mpf_gbl_pcg_get_mem_size(solver);
}

void mpf_dsy_gbl_pcg_constrained
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int m = A->m;
  MPF_Int blk = solver->batch;
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  double *Rold = (double*)solver->inner_mem;
  double *Rnew = &Rold[m*blk];
  double *Dvec = &Rnew[m*blk];
  double *Dvec_temp = &Dvec[m*blk];
  double *M = &Dvec_temp[m*blk];

  //double *T = &M[m*blk];
  double *Rtemp = NULL;
  double B_norm = mpf_dlange(MPF_COL_MAJOR, 'F', m, blk, B, m);
  double r_norm = 1.0;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;
  mpf_convert_layout_to_sparse(layout, &sparse_layout);

  memcpy(Rold, B, (sizeof *Rold)*m*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, m, 1.0, Rold, m);
  vdMul(m*blk, B, Rold, Dvec);

  //memcpy(Dvec, Rold, (sizeof *Dvec)*m*blk);
//solver->precond_function(solver, m, Rold, solver->M);
  //mpf_sparse_d_mm_wrapper(solver-> m, Rold, M);
  //printf("X[0]: %1.4E, Rold[0]: %1.4E\n", X[0], Rold[0]);
  MPF_Int i = 0;
  while ((i < solver->iterations) && (r_norm > solver->tolerance))
  {
    /* computes alpha, gamma */
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      sparse_layout, Dvec, blk, m, 0.0, Dvec_temp, m);

    trace_r_old = mpf_ddot(m*blk, Rold, 1, M, 1);
    gamma = mpf_ddot(m*blk, Dvec_temp, 1, Dvec, 1);
    alpha = trace_r_old/gamma;

    /* updates X and residual vecblk */
    mpf_daxpy(m*blk, alpha, Dvec, 1, X, 1);
    vdMul(m*blk, B, Rold, Rnew);
    //memcpy(Rnew, Rold, (sizeof *Rnew)*m*blk);
    mpf_daxpy(m*blk, -alpha, Dvec_temp, 1, Rnew, 1);

    /* applies preconditioning */
//    solver->precond_function(solver, m, Rnew, M);

    /* computes beta */
    //trace_r_new = mpf_ddot(m*blk, Rnew, 1, Rnew, 1);
    trace_r_new = mpf_ddot(m*blk, Rnew, 1, M, 1);
    beta = trace_r_new/trace_r_old;

    /* update direction vecblk */
    mpf_dscal(m*blk, beta, Dvec, 1);
    //mpf_daxpy(m*blk, 1.0, Rnew, 1, Dvec, 1);
    mpf_daxpy(m*blk, 1.0, M, 1, Dvec, 1);
    r_norm = mpf_dlange(CblasColMajor, 'F', m, blk, Rnew, m)/B_norm;

    /* swaps old with new residual vecblks*/
    Rtemp = Rnew;
    Rnew = Rold;
    Rold = Rtemp;
    i += 1;
  }
}

void mpf_global_pcg_get_mem
(
  MPF_Solver *solver
)
{
  MPF_Int blk = solver->batch;
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
        (n*blk*2  /* size_B and size_X */
        +n*blk    /* size_residual_old */
        +n*blk    /* size_residual_new */
        +n*blk    /* size_direction */
        +n*blk    /* size_M */
        +n*blk    /* size_T */
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
