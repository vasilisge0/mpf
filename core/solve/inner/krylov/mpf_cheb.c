#include "mpf.h"

/* ----------------------- Chebyshev approximations ------------------------- */

void mpf_sparse_dsy_cheb
(
  MPF_Solver *solver,

  const MPF_Int iters,
  const double lmin,
  const double lmax,
  const double *c,

  /* data */
  const MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int i = 0;
  MPF_Int M = iters;
  MPF_Int m_B = B_dense->m;
  MPF_Int n_B = B_dense->n;
  MPF_Int skip_ahead = M*M
                   + M
                   + M
                   + M
                   + M
                   + M;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  double *V = &((double*)solver->inner_mem)[skip_ahead];
  double *W = &V[m_B*n_B];
  double *swap = NULL;
  memcpy(V, B, (sizeof *V)*m_B*n_B);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 2.0/(lmax-lmin), A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, V, n_B, m_B, 0.0, W, m_B);
  mpf_daxpy(m_B*n_B, -(1.0+2.0*lmin/(lmax-lmin)), V, 1, W, 1);
  memcpy(X, V, (sizeof *V)*m_B*n_B);
  mpf_dscal(m_B*n_B, c[0], X, 1);
  mpf_daxpy(m_B*n_B, c[1], W, 1, X, 1);
  for (i = 2; i < iters; ++i)
  {
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 2.0*(2.0/(lmax-lmin)),
      A->handle, A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, W, n_B, m_B, -1.0, V, m_B);
    mpf_daxpy(m_B*n_B, -2.0*(1.0+2.0*lmin/(lmax-lmin)), W, 1, V, 1);
    mpf_daxpy(m_B*n_B, c[i], V, 1, X, 1);
    swap = V;
    V = W;
    W = swap;
  }
}

void mpf_dsy_cheb_poly
(
  MPF_Int M,     /* number of points = polynomial_degree-1 */
  double lmin, /* minimum eigenvalue */
  double lmax, /* maximum eigenvalue */
  void (*target_func)(MPF_Int, double*, double*),
  double *memory,
  double *cheb_coeffs
)
{
  MPF_Int i = 0;
  double *T = memory;
  double *c = &T[M*M];
  double *fapprox = &c[M];
  double *z = &fapprox[M];
  double *y = &z[M];
  double *fy = &y[M];

  /* generates Chebyshev nodes of the second kind in z */
  for (i = 0; i < M; ++i)
  {
    z[i] = cos(PI*(2*i+1)/(2*M));
  }

  /* initializes y */
  for (i = 0; i < M; ++i)
  {
    y[i] = (z[i]+1)*(lmax-lmin)/2 + lmin;
  }

  printf("z[0]: %1.4E, z[0]+1: %1.4E, (lmax-lmin)/2: %1.4E\n",
    z[0], z[0]+1, (lmax-lmin)/2);
  printf("y[0]: %1.4E\n", y[0]);

  mpf_matrix_d_announce(y, M, 1, M, "y");

  /* initializes T(:, 1) and T(:, 2) */
  mpf_matrix_d_set(MPF_COL_MAJOR, M, 1, T, M, 1.0);
  memcpy(&T[M], z, (sizeof *T)*M);

  for (i = 2; i < M; ++i)
  {
    vdMul(M, &T[M*(i-1)], &T[M], &T[M*i]);
    mpf_dscal(M, 2.0, &T[M*i], 1);
    mpf_daxpy(M, -1.0, &T[M*(i-2)], 1, &T[M*i], 1);
  }

  mpf_matrix_d_announce(T, M, M, M, "T");

  target_func(M, y, fy);

  printf("lmax: %1.4E\n", lmax);
  printf("lmin: %1.4E\n", lmin);

  mpf_matrix_d_announce(z, M, 1, M, "z");
  mpf_matrix_d_announce(fy, M, 1, M, "fy");

  mpf_dgemv(CblasColMajor, CblasTrans, M, M, 1.0, T, M, fy, 1, 0.0, y, 1);
  for (i = 0; i < M; ++i)
  {
    z[i] = mpf_ddot(M, &T[M*i], 1, &T[M*i], 1);
  }

  mpf_matrix_d_announce(z, M, 1, M, "dnrm");
  mpf_matrix_d_announce(y, M, 1, M, "y"); 
  /* computes coefficients */
  vdDiv(M, y, z, c);

  printf("c[0]: %1.4E\n", c[0]);
  printf("c[1]: %1.4E\n", c[1]);
  printf("c[2]: %1.4E\n", c[2]);
  printf("c[3]: %1.4E\n", c[3]);
  printf("c[4]: %1.4E\n", c[4]);

  /* computes approximate function */
  mpf_dgemv(CblasColMajor, CblasNoTrans, M, M, 1.0, T, M, c, 1, 0.0, fapprox, 1);

  /* set output */
  cheb_coeffs = c;
}

void mpf_cheb_get_mem
(
  MPF_Solver *solver
)
{
  MPF_Int M = solver->n_defl;
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
      (
      +M*M /* T */
      +M   /* c */
      +M   /* fapprox */
      +M   /* z */
      +M   /* y */
      +M   /* fy */
      +n   /* B */
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

void mpf_dsy_inv_1D
(
  MPF_Int n,
  double *x,
  double *fx
)
{
  MPF_Int i = 0;
  for (i = 0; i < n; ++i)
  {
    fx[i] = 1/x[i];
  }
}

void mpf_zsy_inv_1D
(
  MPF_Int n,
  double *x,
  double *fx
)
{
  MPF_Int i = 0;
  for (i = 0; i < n; ++i)
  {
    fx[i] = 1/x[i];
  }
}


void mp_cheb_init
(
  MPF_Solver *solver,
  char target_function[MPF_MAX_STRING_SIZE],
  MPF_Int cheb_M,
  MPF_Int iterations
)
{
  //solver->cheb_ev_iterations = iterations;
  //if (solver->data_type == MP_REAL)
  //{
  //  solver->inner_type = MP_SOLVER_CPU_DSY_CHEB;
  //  solver->inner_function = &mp_sparse_dsy_cheb;
  //}
  //else if ((solver->data_type == MP_COMPLEX) &&
  //         (solver->A_type == MP_MATRIX_SYMMETRIC))
  //{
  //  solver->solver_inner_type = MP_SOLVER_CPU_ZSY_LANCZOS;
  //  solver->solver_inner_function = &mp_zsy_lanczos;
  //}
  //else if ((solver->data_type == MP_COMPLEX) &&
  //         (solver->A_type == MP_MATRIX_HERMITIAN))
  //{
  //  solver->solver_inner_type = MP_SOLVER_CPU_ZHE_LANCZOS;
  //  solver->solver_inner_function = &mp_zhe_lanczos;
  //}
 // if (strcmp(target_function, "INV") == 0)
 // {
 //   solver->target_func = &mp_dsy_inv_1D;
 // }

 // solver->solver_ev_max_func = &mp_sparse_dsy_ev_max_iterative;
 // solver->solver_ev_min_func = &mp_sparse_dsy_ev_min_iterative;
 // solver->cheb_M = cheb_M;

 // //mp_sparse_ev_max_memory_get(solver->data_type, context->cheb_ev_iterations,
 // //  solver->m_A, &context->bytes_ev_max);
 // //mp_sparse_ev_min_memory_get(solver->data_type, context->cheb_ev_iterations,
 // //  solver->m_A, &context->bytes_ev_min);
 // mp_cheb_memory_get(solver->data_type, context->cheb_M, context->cheb_ev_iterations,
 //   solver->m_A, &context->bytes_inner);
}//
