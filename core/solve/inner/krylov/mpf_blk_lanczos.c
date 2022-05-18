#include "mpf.h"

void mpf_dsy_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* solver */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_B = B_dense->m;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* assign cpu memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m_B*(m_H+blk)];
  double *Br = &H[m_H*n_H];
  double *reflectors_array = &Br[m_H*blk];
  double *B_norms_array = &reflectors_array[blk];

  /* assign handles to cpu memory */
  double *V_first_vecblk = V;
  double *Vlast = &V[(m_B*blk)*inner_iterations];
  double *R = Vlast;
  double *W = NULL;
  double *Vprev = NULL;
  double *Hblk = NULL;
  double *Hblk_dest = NULL;

  /* */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dnrm2(m_B, &((double*)B)[m_B*i], 1);
  }
  memcpy(V, B, (sizeof *V)*m_B*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m_B, 1.0, V, m_B);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    r_norm = mpf_dnrm2(m_B, &V[m_B*i], 1) / B_norms_array[i];
    if (r_norm > r_norm_max)
    {
        r_norm_max = r_norm;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif

  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, V, m_B, reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m_B, blk, V, m_B, Br, ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, V, m_B, reflectors_array);
    W = &V[(m_B*blk)];
    Vprev = V;
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, 0.0, W, m_B);

    Hblk = H;
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0, Vprev,
      m_B, W, m_B, 0.0, Hblk, ld_H);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
      Vprev, m_B, Hblk , ld_H, 1.0, W, m_B);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W , m_B, reflectors_array);

    Hblk = H + blk;
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
    H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);

    /* inner iterations */
    for (MPF_Int j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H*blk)*j + blk*j];
      W = &V[(m_B*blk)*(j+1)];
      Vprev = &V[(m_B*blk*j)];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, 0.0, W, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
        Vprev, m_B, W, m_B, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m_B * blk)*(j-1)];
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W, m_B, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
      H_IJ_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk * (j+1);
        n_H = blk * j;
        break;
      }
      mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
    }

    if (H_IJ_norm > 1e-12)
    {
      MPF_Int j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, 0.0, W, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
        Vprev, m_B, W, m_B, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V_first_vecblk[(m_B * blk)*(j-1)];
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);
    }

    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_dsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, 1.0, H, ld_H, Br, ld_H);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, n_H, 1.0,
      V_first_vecblk, m_B, Br, m_H, 1.0, X, m_B);
    memcpy(R, B, (sizeof *R) * m_B * blk);
    mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, m_B, 1.0, R, m_B);
    r_norm_max = 0.0;

    /* finds max relative residual norm norm */
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dnrm2(m_B, &R[m_B*i], 1)/B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG == 1
        printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_max, k);
    #endif

    /* check if relative residual norm is small enough */
    if (r_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m_B*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mpf_zsy_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble COMPLEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);

  /* context */
  MPF_ComplexDouble H_IJ_norm = ZERO_C;
  double H_IJ_norm_abs = 0.0;
  MPF_ComplexDouble r_norm = ZERO_C;
  double r_norm_abs = 0.0;
  double r_norm_abs_max = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = solver->iterations*blk;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* allocates memory for mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble *) solver->inner_mem;
  MPF_ComplexDouble *H = &V[m*(m_H + blk)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  MPF_ComplexDouble *B_norms_array = &Br[m_H*blk];

  /* assigns handles on cpu memory */
  MPF_ComplexDouble *V_first_vecblk = V;
  MPF_ComplexDouble *Vlast = &V[(m*blk)*inner_iterations];
  MPF_ComplexDouble *R = Vlast;
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Hblk_dest = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
  memcpy(V, B, (sizeof *V_first_vecblk)*m*blk);
  mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, ONE_C, V_first_vecblk, m);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m, &ONE_C,
      B, m, B, m, &ZERO_C, &B_norms_array[i], ld_H);
    mpf_vectorized_z_sqrt(1, &B_norms_array[i], &B_norms_array[i]);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m, &ONE_C,
      &V[m*i], m, &V[m*i], m, &ZERO_C, &r_norm, ld_H);

    mpf_vectorized_z_sqrt(1, &r_norm, &r_norm);
    r_norm = mpf_scalar_z_divide(r_norm, B_norms_array[i]);
    mpf_vectorized_z_abs(1, &r_norm, &r_norm_abs);
    if (r_norm_abs > r_norm_abs_max)
    {
      r_norm_abs_max = r_norm_abs;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_abs_max);
  #endif

  if (r_norm_abs_max <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_gram_schmidt_zge(m, blk, V, Br, m_H);
    W = &V[(m * blk)];
    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
      A->descr, sparse_layout, Vprev, blk, m, ZERO_C, W, m);

    Hblk = H;
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m,
      &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
      &COMPLEX_MINUS_ONE, Vprev, m, Hblk , ld_H, &ONE_C, W, m);

    Hblk = H + blk;
    mpf_gram_schmidt_zge(m, blk, W, Hblk, ld_H);
    for (MPF_Int i = 0; i < blk; ++i)
    {
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, blk, &ONE_C,
        &Hblk[ld_H*i], ld_H, &Hblk[ld_H*i], ld_H, &ONE_C, &H_IJ_norm, 1);
    }

    mpf_vectorized_z_sqrt(1, &H_IJ_norm, &H_IJ_norm);
    mpf_vectorized_z_abs(1, &H_IJ_norm, &H_IJ_norm_abs);
    if (H_IJ_norm_abs <= 1e-12)
    {
      inner_iterations = 1;
      m_H = blk;
      n_H = blk;
      break;
    }

    for (MPF_Int j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W,
        m);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy ('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest,
        ld_H);
      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_gram_schmidt_zge(m, blk, W, Hblk, ld_H);

      for (MPF_Int i = 0; i < blk; ++i)
      {
        mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, blk,
          &ONE_C, &Hblk[ld_H*i], ld_H, &Hblk[ld_H*i], ld_H, &ONE_C,
          &H_IJ_norm, 1);
      }
      mpf_vectorized_z_sqrt(1, &H_IJ_norm, &H_IJ_norm);
      vzAbs(1, &H_IJ_norm, &H_IJ_norm_abs);
      if (H_IJ_norm_abs <= 1e-12)
      {
        inner_iterations = (j+1);
        m_H = blk*j;
        n_H = blk*j;
        break;
      }
    }

    if (H_IJ_norm_abs > 1e-12)
    {
      MPF_Int j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W,
        m);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy ('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest,
        ld_H);
    }

    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_zsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, &ONE_C, H, ld_H, Br, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, n_H,
      &ONE_C,  V_first_vecblk, m, Br, m_H, &ONE_C, X, m);
    memcpy(R, B, (sizeof *R)*m*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle,
      A->descr, sparse_layout, X, blk, m, ONE_C, R, m);

    r_norm_abs_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m, &ONE_C,
        &R[m*i], m, &R[m*i], m, &ZERO_C, &r_norm, 1);
      mpf_vectorized_z_sqrt(1, &r_norm, &r_norm);
      r_norm = mpf_scalar_z_divide(r_norm, B_norms_array[i]);
      mpf_vectorized_z_abs(1, &r_norm, &r_norm_abs);
      if (r_norm_abs > r_norm_abs_max)
      {
        r_norm_abs_max = r_norm_abs;
      }
    }

    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_abs_max,
        k);
    #endif

    if ((r_norm_abs_max <= solver->tolerance) || (k == solver->restarts))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mpf_zhe_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble COMPLEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);

  /* context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* allocates memory for mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[m*(m_H + blk)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  double *B_norms_array = (double *) &Br[m_H*blk];

  /* assigns handles on cpu memory */
  MPF_ComplexDouble *Vlast = &V[(m*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Hblk_dest = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
  memcpy(V, B, (sizeof *V)*m*blk);
  mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, ONE_C, V, m);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dznrm2(m, &B[m*i], 1);
    r_norm = mpf_dznrm2(m, &V[m*i], 1);
    r_norm = r_norm / B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }
  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif

  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_gram_schmidt_zhe(m, blk, V, Br, m_H);
    W = &V[(m*blk)];
    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
      A->descr, sparse_layout, Vprev, blk, m, ZERO_C, W, m);

    Hblk = H;
    mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m,
      &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
      &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

    Hblk = H + blk;
    mpf_gram_schmidt_zhe(m, blk, W, Hblk, ld_H);
    H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
    if (H_IJ_norm <= 1e-12)
    {
      inner_iterations = 0;
      m_H = 1;
      n_H = 1;
      break;
    }

    for (MPF_Int j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m*blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W,
        m);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);

      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy('C', 'C', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest,
        ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_gram_schmidt_zhe(m, blk, W, Hblk, ld_H);

      H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = (j+1);
        m_H = blk*j;
        n_H = blk*j;
        break;
      }
    }

    if (H_IJ_norm > 1e-12)
    {
      MPF_Int j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W,
        m);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, blk, blk,
        &COMPLEX_MINUS_ONE, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy('C', 'C', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest,
        ld_H);
    }

    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_zsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, &ONE_C, H, ld_H, Br, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, n_H,
      &ONE_C, V, m, Br, m_H, &ONE_C, X, m);
    memcpy(R, B, (sizeof *R) * m * blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle,
      A->descr, sparse_layout, X, blk, m, ONE_C, R, m);

    r_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dznrm2(m, &R[m*i], 1);
      r_norm = r_norm / B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_max, k);
    #endif

    if ((r_norm_max <= solver->tolerance) || (k == solver->restarts))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mpf_blk_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
)
{
  MPF_Solver *solver = &context->solver;
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;
  context->args.n_inner_solve = 4;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_BLK_LANCZOS;
    solver->inner_function = &mpf_dsy_blk_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_BLK_LANCZOS;
    solver->inner_function = &mpf_zsy_blk_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_BLK_LANCZOS;
    solver->inner_function = &mpf_zhe_blk_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  solver->inner_alloc_function = &mpf_krylov_alloc;
  solver->inner_free_function = &mpf_krylov_free;
  mpf_blk_lanczos_get_mem_size(solver);
}

void mpf_blk_lanczos_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int iterations = solver->iterations;
  MPF_Int blk = solver->batch;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)
      * (n*(iterations+1)*blk           /* size_V */
        +iterations*blk*iterations*blk  /* size_H */
        +iterations*blk*blk             /* size_Br */
        +blk                            /* size reflectors */
        +blk);                          /*?*/
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)
      * (n*(iterations+1)*blk           /* size_V */
        +iterations*blk*iterations*blk  /* size_H */
        +iterations*blk*blk             /* size_Br */
        +blk                            /* size reflectors */
        +blk);                          /*?*/
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
     solver->inner_bytes = sizeof(MPF_ComplexDouble) *
          (n*(iterations+1)*blk           /* size_V */
          +iterations*blk*iterations*blk  /* size_H */
          +iterations*blk*blk             /* size_Br */
          +blk                            /* size reflectors */
          +blk);                          /*?*/
  }
}
