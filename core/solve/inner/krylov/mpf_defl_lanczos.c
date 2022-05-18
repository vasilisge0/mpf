#include "mpf.h"

void mpf_dsy_defl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int defl_flag = 1; /* replace in MPF_Solver */

  /* context */
  double b_norm;
  double r_norm;
  double h_temp;

  /* solver->*/
  MPF_Int blk = solver->batch;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int m_B = solver->ld;
  MPF_Int n_B = blk;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;
  MPF_Int n_V = solver->iterations;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* assign memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m_B*(m_H+1)];
  double *br = &H[m_H*n_H];
  double *r = &br[m_H];
  double *xt = &r[m_B];
  double *refs_array = &xt[m_B];

  /* map handles to allocated memory */
  double *w = NULL;
  double *vprev = NULL;
  double *vcurr = NULL;

  /* unpacking memory_defl */
  double *Vdefl = (double*)solver->inner_mem;
  double *Hdefl = &Vdefl[m_B*solver->iterations*blk];
  double *refs_defl_array = &Hdefl[m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m_B*blk];

  /* first krylov iteration */
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  if (defl_flag)
  {
    /* updates residual */
    memcpy(r, B, (sizeof *r)*m_B);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, 1, m_B, 1.0, Vdefl,
      m_B, r, m_B, 0.0, Tdefl, n_V);
    mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
      Vdefl, m_B, Tdefl, n_V, 0.0, r, m_B);

    /* computes initial residual */
    mpf_daxpy(m_B, 1.0, r, 1, X, 1);
    memcpy(r, B, (sizeof *r)*m_B);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, X,
      1.0, r);
    memcpy(V, r, (sizeof *V)*m_B);
    b_norm = mpf_dnrm2(m_B, B, 1); //@FIX: fix this to consider projector as well
    r_norm = mpf_dnrm2(m_B, r, 1);
    printf("[start] relative residual: %1.4E\n", r_norm/b_norm);

    /* outer-loop (restarts) */
    for (MPF_Int k = 0; k < outer_iterations; ++k)
    {
      mpf_dscal(m_B, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;

      /* update w */
      w = &V[m_B];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, V, n_B, m_B, 0.0, w, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, n_B, m_B, 1.0,
        Vdefl, m_B, w, m_B, 0.0, Tdefl, n_V);
      /* solves linear system */
      mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, n_B, n_V, 1.0,
        Vdefl, m_B, Tdefl, m_B, 0.0, Mdefl, m_B);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_B, 1.0, w, m_B);

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
      for (MPF_Int j = 1; j < inner_iterations; ++j)
      {
        w = &V[m_B*(j+1)];
        vcurr = &V[m_B*j];
        vprev = &V[m_B*(j-1)];

        /* update w */
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, vcurr, n_B, m_B, 0.0, w, m_B);
        mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, 1, m_B, 1.0,
          Vdefl, m_B, w, m_B, 0.0, Tdefl, n_V);
        /* solves linear system */
        mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
        mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
          CblasNonUnit, m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
          Vdefl, m_B, Tdefl, m_B, 0.0, Mdefl, m_B);
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_B, 1.0, w, m_B);

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

      /* solves linear system of equations and checks termination condition */
      mpf_matrix_d_announce(H, m_H, n_H, m_H, "H");
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);

      /* reconstructs x */
      mpf_dgemv(CblasColMajor, CblasNoTrans, m_B, m_H, 1.0, V, m_B, br, 1, 0.0,
        xt, 1);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, xt, n_B, m_B, 0.0, Mdefl, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, 1, m_B, 1.0,
        Vdefl, m_B, Mdefl, m_B, 0.0, Tdefl, n_V);
      /* solves linear system */
      mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, Hdefl, n_H, Tdefl, n_V);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
        Vdefl, m_B, Tdefl, n_V, 0.0, Mdefl, m_B);
      mpf_daxpy(m_B, -1.0, Mdefl, 1, xt, 1);
      mpf_daxpy(m_B, 1.0, xt, 1, X, 1);

      /* computes residual */
      memcpy(r, B, (sizeof *r)*m_B);
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        X, 1.0, r);
      r_norm = mpf_dnrm2(m_B, r, 1);

      printf("[end] relative residual: %1.4E\n", r_norm/b_norm);
      #if DEBUG == 1
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
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }
  else
  {
    memcpy(r, B, sizeof(double)*m_B);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, X,
      1.0, r);
    b_norm = mpf_dnrm2(m_B, B, 1);
    r_norm = mpf_dnrm2(m_B, r, 1);
    memcpy(V, r, (sizeof *V)*m_B);

    /* outer-loop (restarts) */
    for (MPF_Int k = 0; k < outer_iterations; ++k)
    {
      mpf_dscal(m_B, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;
      w = &V[m_B];
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, V,
        0.0, w);
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
      for (MPF_Int j = 1; j < inner_iterations; ++j)
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

      /* solves linear system of equations and checks termination condition */
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);
      mpf_dgemv(CblasColMajor, CblasNoTrans, m_B, m_H, 1.0, V, m_B, br, 1,
        1.0, X, 1);

      memcpy(r, B, (sizeof *r)*m_B);
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        X, 1.0, r);
      r_norm = mpf_dnrm2(m_B, r, 1);

      #if DEBUG == 1
        printf("relative residual: %1.4E\n", r_norm/b_norm);
      #endif
      if ((r_norm / b_norm <= solver->tolerance) || (k == outer_iterations-1))
      {
        outer_iterations = k;
        break;
      }
      else
      {
        /* restart */
        memcpy(V, r, (sizeof *V)*m_B);
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }

  /* copy krylov subspace to memory_defl */
  memcpy(Hdefl, H, (sizeof *Hdefl)*m_H*n_H);
  memcpy(Vdefl, V, (sizeof *V)*m_B*inner_iterations);
  memcpy(refs_defl_array, refs_array, (sizeof *refs_defl_array)
    *(inner_iterations-1));

  w = NULL;
  vprev = NULL;
  vcurr = NULL;
  V = NULL;
  H = NULL;
  br = NULL;
  r = NULL;

  Vdefl = NULL;
  Hdefl = NULL;
  Tdefl = NULL;
  Mdefl = NULL;
}

void mpf_dsy_defl_lanczos_2
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
  MPF_Int n_B = solver->batch;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;
  MPF_Int n_V = solver->iterations;

  /* assign memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m_B*(m_H+1)];
  double *br = &H[m_H*n_H];
  double *r = &br[m_H];
  double *xt = &r[m_B];
  double *refs_array = &xt[m_B];

  /* map handles to allocated memory */
  double *w = NULL;
  double *vprev = NULL;
  double *vcurr = NULL;

  /* unpacking memory_defl */
  double *Vdefl = (double*)solver->inner_mem;
  double *Hdefl = &Vdefl[m_B*solver->iterations*solver->batch];
  double *refs_defl_array = &Hdefl[4*m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m_B*solver->batch];
  double *e_vecs = &Mdefl[m_B*solver->batch];
  double *e_vals = &e_vecs[m_B*solver->iterations];

  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  if (solver->use_defl)
  {
    /*-----------------------------------------*/
    /* projects residual to invariant subspace */
    /*-----------------------------------------*/
    /* updates residual */
    memcpy(r, b, (sizeof *r)*m_B);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, 1, m_B, 1.0, Vdefl,
      m_B, r, m_B, 0.0, Tdefl, n_V);
    mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
      Vdefl, m_B, Tdefl, n_V, 0.0, r, m_B);
    /* computes initial residual */
    mpf_daxpy(m_B, 1.0, r, 1, x, 1);
    memcpy(r, b, (sizeof *r)*m_B);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, x, 1.0, r);
    memcpy(V, r, (sizeof *V)*m_B);
    b_norm = mpf_dnrm2(m_B, b, 1); //@FIX: fix this to consider projector as well
    r_norm = mpf_dnrm2(m_B, r, 1);
    printf("[start] relative residual: %1.4E\n", r_norm/b_norm);

    /*------------------------*/
    /* low rank decomposition */
    /*------------------------*/
    /* outer-loop (restarts) */
    for (k = 0; k < outer_iterations; ++k)
    {
      mpf_dscal(m_B, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;
      /* update w */
      w = &V[m_B];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, V, n_B, m_B, 0.0, w, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, n_B, m_B, 1.0,
        Vdefl, m_B, w, m_B, 0.0, Tdefl, n_V);
      /* solves linear system */
      mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, n_B, n_V, 1.0,
        Vdefl, m_B, Tdefl, m_B, 0.0, Mdefl, m_B);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_B, 1.0, w, m_B);
      /* computes column of H */
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

      /* inner iterations */
      for (j = 1; j < inner_iterations; ++j)
      {
        w = &V[m_B*(j+1)];
        vcurr = &V[m_B*j];
        vprev = &V[m_B*(j-1)];
        /* update w */
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, vcurr, n_B, m_B, 0.0, w, m_B);
        mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, 1, m_B, 1.0,
          Vdefl, m_B, w, m_B, 0.0, Tdefl, n_V);
        mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
        mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
          CblasNonUnit, m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
          Vdefl, m_B, Tdefl, m_B, 0.0, Mdefl, m_B);
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_B, 1.0, w, m_B);
        /* computes new column of H */
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

      /*--------------------------------------------------------------------*/
      /* solves linear system of equations and checks termination condition */
      /*--------------------------------------------------------------------*/
      mpf_matrix_d_announce(H, m_H, n_H, m_H, "H");
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);
      /* reconstructs x */
      mpf_dgemv(CblasColMajor, CblasNoTrans, m_B, m_H, 1.0, V, m_B, br, 1, 0.0,
        xt, 1);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, xt, n_B, m_B, 0.0, Mdefl, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, 1, m_B, 1.0,
        Vdefl, m_B, Mdefl, m_B, 0.0, Tdefl, n_V);
      mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, Hdefl, n_H, Tdefl, n_V);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_V, 1.0,
        Vdefl, m_B, Tdefl, n_V, 0.0, Mdefl, m_B);
      mpf_daxpy(m_B, -1.0, Mdefl, 1, xt, 1);
      /* updates x */
      mpf_daxpy(m_B, 1.0, xt, 1, x, 1);

      /*------------------------*/
      /* eigenvalue computation */
      /*------------------------*/
      /* sets (2*m_H) x (2*n_H) augmented hessnberg matrix */
      mpf_diag_d_set(MPF_COL_MAJOR, m_H, n_H, H, 2*m_H, e_vals);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, &H[m_H], 2*m_H);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_B, n_H, 1.0,
        Vdefl, m_B, V, m_B, 0.0, &Hdefl[(2*m_H)*m_H], 2*m_H);
      mpf_domatcopy('C', 'N', m_H, n_H, 1.0, H, m_H,
        &Hdefl[(2*m_H)*m_H+m_H], m_H);

      /*-------------------*/
      /* computes residual */
      /*-------------------*/
      memcpy(r, b, (sizeof *r)*m_B);
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, x,
        1.0, r);
      r_norm = mpf_dnrm2(m_B, r, 1);
      #if DEBUG == 1
        printf("[end] relative residual: %1.4E\n", r_norm/b_norm);
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
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }
  else
  {
    memcpy(r, b, (sizeof *b)*m_B);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x,
      1.0, r);
    b_norm = mpf_dnrm2(m_B, b, 1);
    r_norm = mpf_dnrm2(m_B, r, 1);
    memcpy(V, r, (sizeof *V)*m_B);

    /* outer-loop (restarts) */
    for (k = 0; k < outer_iterations; ++k)
    {
      mpf_dscal(m_B, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;
      w = &V[m_B];
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, V,
        0.0, w);
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

      /* solves linear system of equations and checks termination condition */
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);
      mpf_dgemv(CblasColMajor, CblasNoTrans, m_B, m_H, 1.0, V, m_B, br, 1,
        1.0, x, 1);

      memcpy(r, b, (sizeof *r)*m_B);
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        x, 1.0, r);
      r_norm = mpf_dnrm2(m_B, r, 1);

      #if DEBUG == 1
        printf("relative residual: %1.4E\n", r_norm/b_norm);
      #endif
      if ((r_norm / b_norm <= solver->tolerance) || (k == outer_iterations-1))
      {
        outer_iterations = k;
        break;
      }
      else
      {
        /* restart */
        memcpy(V, r, (sizeof *V)*m_B);
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }

  /* copy krylov subspace to memory_defl */
  memcpy(Hdefl, H, (sizeof *Hdefl)*m_H*n_H);
  memcpy(Vdefl, V, (sizeof *V)*m_B*inner_iterations);
  memcpy(refs_defl_array, refs_array, (sizeof *refs_defl_array)
    *(inner_iterations-1));

  w = NULL;
  vprev = NULL;
  vcurr = NULL;
  V = NULL;
  H = NULL;
  br = NULL;
  r = NULL;

  Vdefl = NULL;
  Hdefl = NULL;
  Tdefl = NULL;
  Mdefl = NULL;
}

void mpf_defl_lanczos_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;
  solver->use_defl = 1;

  if (solver->device == MPF_DEVICE_CPU)
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DSY_LANCZOS;
      solver->inner_function = &mpf_dsy_defl_lanczos;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_LANCZOS;
      solver->inner_function = &mpf_zsy_lanczos;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZHE_LANCZOS;
      solver->inner_function = &mpf_zhe_lanczos;
      solver->device = MPF_DEVICE_CPU;
    }
  }

  mpf_defl_lanczos_get_mem_size(solver);
}

void mpf_defl_lanczos_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int blk = solver->batch;
  MPF_Int iterations = solver->iterations;
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)
      * (n*blk                 /* size_B */
        +n*blk                 /* size_X */
        +n*(iterations+1)      /* size_V */
        +iterations*iterations /* size_H */
        +iterations            /* size_br */
        +n                     /* size_residual */
        +iterations);          /* refs_array */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)
      * (n*blk                 /* size_B */
        +n*blk                 /* size_X */
        +n*(iterations+1)      /* size_V */
        +iterations*iterations /* size_H */
        +iterations            /* size_br */
        +n);                   /* size_residual */
  }
}
