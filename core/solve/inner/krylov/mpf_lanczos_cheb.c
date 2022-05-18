#include "mpf.h"

void mpf_dsy_lanczos_cheb
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* context */
  MPF_Int k = 0;
  MPF_Int j = 0;
  double b_norm;
  double r_norm;
  double h_temp;

  double *b = (double*)b_dense->data;
  double *x = (double*)x_dense->data;

  /* solver->*/
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int m_B = solver->ld;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;

  /* assign memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m_B*(m_H+1)];
  double *br = &H[m_H*n_H];
  double *r = &br[m_H];
  /* map handles to allocated memory */
  double *w = NULL;
  double *vprev = NULL;
  double *vcurr = NULL;
  /* first krylov iteration */
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(r, b, sizeof(double)*m_B);
  mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x, 1.0, r);
  b_norm = mpf_dnrm2(m_B, b, 1);
  r_norm = mpf_dnrm2(m_B, r, 1);
  memcpy(V, r, (sizeof *V)*m_B);

  /* -- outer-loop (restarts) -- */
  for (k = 0; k < outer_iterations; ++k)
  {
    mpf_dscal(m_B, 1/r_norm, V, 1);
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = r_norm;
    w = &V[m_B];
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, V, 0.0, w);
    H[0] = mpf_ddot(m_B, w, 1, V, 1);
    mpf_daxpy(m_B, -H[0], V, 1, w, 1);
    h_temp = mpf_dnrm2(m_B, w, 1);
    if (h_temp < 1e-12)
    {
      inner_iterations = 1;
      break;
    }
    H[1] = h_temp;
    mpf_dscal(m_B, 1/H[1], w, 1);
    for (j = 1; j < inner_iterations; ++j)
    {
      w = &V[m_B*(j+1)];
      vcurr = &V[m_B*j];
      vprev = &V[m_B*(j-1)];
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle,
        A->descr, vcurr, 0, w);
      H[m_H*j+j] = mpf_ddot(m_B, w, 1, vcurr, 1);
      mpf_daxpy(m_B, -H[m_H*j+j], vcurr, 1, w, 1);
      H[m_H*j+j-1] = H[m_H*(j-1)+j];
      mpf_daxpy(m_B, -H[m_H*j+j-1], vprev, 1, w, 1);
      h_temp = mpf_dnrm2(m_B, w, 1);
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }
      H[m_H*j+j+1] = h_temp;
      mpf_dscal(m_B, 1/H[m_H*j+j+1], w, 1);
    }

    /*solves linear system of equations and checks termination condition */
    mpf_qr_dsy_givens(m_H, n_H, 1, H, ld_H, br);
    //mpf_dsy_cheb(inner_iterations, );
    //mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    //  m_H, 1, 1.0, H, n_H, br, m_H);
    mpf_dgemv(CblasColMajor, CblasNoTrans, m_B, m_H, 1.0, V, m_B, br, 1, 0.0,
      x, 1);
    memcpy(r, b, (sizeof *r)*m_B);
    mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      x, 1.0, r);
    r_norm = mpf_dnrm2(m_B, r, 1);

    #if DEBUG == 1
        printf("relative residual: %1.4E\n", r_norm/b_norm);
    #endif
    if ((r_norm / b_norm <= solver->tolerance) && (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      /* restart */
      memcpy(V, r, (sizeof *V)*m_B);
    }
  }

  w = NULL;
  vprev = NULL;
  vcurr = NULL;
  V = NULL;
  H = NULL;
  br = NULL;
  r = NULL;
}
