#include "mpf.h"

/* qr_eig */
/* eigenvalue computation using the qr algorithm */
/* */
void mpf_qr_ev
(
  const MPF_Layout layout,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *R,
  double *Z,
  MPF_Int max_iters,
  double tol
)
{
  MPF_Int i = 0;
  MPF_Int k_curr = 0;
  double *H_handle = H;
  double er = 1.0;

  while ((er > tol) && (i < max_iters))
  {
    mpf_matrix_d_diag_set(layout, m_H, n_H, R, m_H, 1.0);
    mpf_qr_dsy_givens(m_H, n_H, n_B, H_handle, ld_H, R);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_H, n_H, n_H, 1.0,
      R, m_H, H_handle, m_H, 0.0, Z, m_H);
    H_handle = Z;
    er = fabs(H[m_H*k_curr+k_curr+1])
       /(fabs(H[m_H*k_curr+k_curr])+fabs(H[m_H*(k_curr+1)+k_curr]));
    i += 1;
  }
}

void mpf_sparse_dsy_ev_min_iterative
(
  MPF_Solver *solver,

  /* solver parameters */
  VSLStreamStatePtr stream,

  /* data */
  MPF_Sparse *A,
  double *ev_min
)
{
  MPF_Int m_A = solver->ld;
  MPF_Dense B;
  B.data = solver->inner_mem;
  MPF_Dense X;
  X.data = &((double*)B.data)[m_A];
  void *swap = NULL;
  //double *memory_ev = &((double*)X.data)[m_A];
  //double MEAN = 0.0;
  //double STD_DEVIATION = 1.0;
  //double curr_ev = 0.0; /* do not use it currently */
  double norm_evec = 0.0;
  MPF_Int iterations_ev = solver->iterations;

  //vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, m_A, B, MEAN,
  //  STD_DEVIATION);
  mpf_matrix_d_set(MPF_COL_MAJOR, m_A, 1, (double*)B.data, m_A, 1.0);
  norm_evec = mpf_dnrm2(m_A, (double*)B.data, 1);
  mpf_dscal(m_A, 1/norm_evec, (double*)B.data, 1);

  printf("iterations_ev: %d\n", iterations_ev);
  iterations_ev = 100;
  for (MPF_Int i = 0; i < iterations_ev; ++i)
  {
    /* computes inverse iteration step Ax = X*/
    mpf_zeros_d_set(MPF_COL_MAJOR, m_A, 1, (double*)X.data, m_A);
    mpf_dsy_lanczos(solver, A, &B, &X);

    /* computes eigenvalue and normalizes eigenvector */
    norm_evec = mpf_dnrm2(m_A, (double*)X.data, 1);
    mpf_dscal(m_A, 1/norm_evec, (double*)X.data, 1);
    //curr_ev = mpf_ddot(m_A, B, 1, X, 1); /* do not use it currently */

    /* applies defl using the generated Krylov basis */
    //mpf_dsy_defl(m_A, solver->blk, solver->iterations*solver->blk, memory,
    //  m_A, X, m_A, B, m_A);

    /* initializes B for next iteration */
    swap = X.data;
    X.data = B.data;
    B.data = swap;
  }

  /* extract eigenvalue */
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, (double*)B.data, 1, m_A, 0.0, (double*)X.data, m_A);
  *ev_min = mpf_ddot(m_A, (double*)B.data, 1, (double*)X.data, 1);
  vslDeleteStream(&stream);
}

void mpf_sparse_dsy_ev_max_iterative
(
  MPF_Solver *solver,
  VSLStreamStatePtr stream,

  /* data */
  MPF_Sparse *A,

  MPF_Int iterations_ev,
  double *ev_max
)
{
  MPF_Int i = 0;
  MPF_Int m_A = solver->ld;
  double *B = (double*)solver->inner_mem;
  double *X = &B[m_A];
  //double *W = &X[m_A];
  double *swap = NULL;
  double curr_ev = 0.0;
  double norm_evec = 0.0;
  //double norm_r = 0.0;
  double MEAN = 0.0;
  double STD_DEVIATION = 1.0;

  vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, m_A, B, MEAN,
    STD_DEVIATION);
  mpf_matrix_d_set(MPF_COL_MAJOR, m_A, 1, B, m_A, 1.0);
  norm_evec = mpf_dnrm2(m_A, B, 1);
  mpf_dscal(m_A, 1/norm_evec, B, 1);

  iterations_ev = 200;
  while ((i < iterations_ev))
  {
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, B, 0.0, X);
    curr_ev = mpf_ddot(m_A, B, 1, X, 1);
    norm_evec = mpf_dnrm2(m_A, B, 1);

    curr_ev = curr_ev/norm_evec;
    norm_evec = mpf_dnrm2(m_A, X, 1);
    mpf_dscal(m_A, 1/norm_evec, X, 1);

    /* swaps B and X */
    swap = B;
    B = X;
    X = swap;
    i += 1;
  }
  *ev_max = curr_ev;
}

void mpf_ev_max_get_mem
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
      (n   /* B */
      +n   /* X */
      +n); /* W */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      (n   /* B */
      +n   /* X */
      +n); /* W */
  }
}
