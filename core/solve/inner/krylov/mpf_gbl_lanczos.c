#include "mpf.h"

void mpf_dsy_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* solver context */
  double B_norm = 0.0;
  double R_norm = 0.0;
  double trace = 0.0;
  double h_temp = 0.0;

  double *B = (double*) B_dense->data;
  double *X = (double*) X_dense->data;

  /* solver->*/
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1 + solver->restarts;
  MPF_Int ld_H = solver->iterations;
  MPF_Int max_n_H = solver->iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = max_n_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* assign cpu memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m*blk*(m_H+1)];
  double *Hblk = &H[m_H*n_H];
  double *Br = &Hblk[blk*blk];

  /* handles to allocated cpu memory */
  double *W = NULL;
  double *Vprev = NULL;
  double *Vlast = &V[(m * blk) * solver->iterations];
  double *R = Vlast;

  /* compute residual vectors block using initial approximation of solution */
  /* block vectors */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V) * m * blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, 1.0, V, m);
  B_norm = mpf_dlange (CblasColMajor, 'F', m, blk, B, m);
  R_norm = mpf_dlange(CblasColMajor, 'F', m, blk, V, m);

  #if DEBUG
    printf("relative residual frobenious norm: %1.4E\n", R_norm/B_norm);
  #endif

  if (R_norm/B_norm <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_dscal(m*blk, 1/R_norm, V, 1);
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0] = B_norm;
    W = &V[(m * blk)];
    Vprev = V;
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      sparse_layout, Vprev, blk, m, 0.0, W, m);

    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, 1.0,
      Vprev, m, W, m, 0.0, Hblk, blk);
    trace = 0.0;
    for (MPF_Int t = 0; t < blk; ++t)
    {
      trace = trace + Hblk[blk*t + t];
    }
    H[0] = trace;
    mpf_daxpy(m*blk, -trace, Vprev, 1, W, 1);
    H[1] = LAPACKE_dlange (LAPACK_COL_MAJOR, 'F', m, blk, W, m);
    mpf_dscal(m*blk, 1/H[1], W, 1);

    for (MPF_Int j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];

      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, m, 0.0, W, m);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, 1.0,
        Vprev, m, W, m, 0.0, Hblk, blk);

      trace = 0.0;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = trace + Hblk[blk*t + t];
      }

      H[ld_H*j + j] = trace;
      mpf_daxpy(m*blk, -trace, Vprev , 1, W, 1);
      Vprev = &V[(m*blk)*(j-1)];
      mpf_daxpy(m*blk, -H[m_H*(j-1) + j], Vprev , 1, W, 1);
      h_temp = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, blk, W, m);
      H[ld_H*j + j-1] = H[ld_H*(j-1) + j];
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      H[ld_H*j + j+1] = h_temp;
      mpf_dscal(m*blk, 1/H[m_H*j + j+1], W, 1);
    }

    /* solves upper triangular linear system and evaluates
       termination criteria */
    mpf_qr_dsy_givens(m_H, n_H, 1, H, ld_H, Br);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[m*blk*i];
      mpf_daxpy(m*blk, Br[i], W, 1, X, 1);
    }

    #if DEBUG
      mpf_matrix_d_announce(Br, m_H, 1, ld_H, "Br");
    #endif

    memcpy(R, B, (sizeof *R) * m * blk);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, m, 1.0, R, m);
    R_norm = mpf_dlange(CblasColMajor, 'F', m, blk, R, m);

    #if DEBUG
      printf("residual norm: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm / B_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m*blk);
    }
  }

  V = NULL;
  H = NULL;
  Hblk = NULL;
  Br = NULL;
  W = NULL;
  Vprev = NULL;
  Vlast = NULL;
  R = NULL;
}

void mpf_zsy_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants*/
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble COMPF_LEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);

  /* solver context */
  MPF_ComplexDouble B_norm = ZERO_C;
  MPF_ComplexDouble R_norm = ZERO_C;
  MPF_ComplexDouble trace = ZERO_C;
  MPF_ComplexDouble tempf_complex;
  MPF_ComplexDouble h_temp;
  double B_norm_abs;
  double R_norm_abs;
  double h_abs;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* solver->*/
  MPF_Int m_B = solver->ld;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int ld_H = inner_iterations;
  //MPF_Int max_n_H = inner_iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = ld_H;
  MPF_Int blk = solver->batch;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[m_B*blk*(m_H+1)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  MPF_ComplexDouble *Hblk = &Br[m_H];
  /* handles on cpu memory */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Vlast = &V[(m_B*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;

  /* compute residual vectors block using initial approximation of
     solution block vectors*/
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V)*m_B*blk);
  mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, COMPF_LEX_MINUS_ONE, A->handle,
    A->descr, sparse_layout, X, blk, m_B, ONE_C, V, m_B);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
    B, m_B*blk, B, m_B * blk, &ZERO_C, &B_norm, 1);
  mpf_vectorized_z_sqrt(1, &B_norm, &B_norm);
  mpf_vectorized_z_abs(1, &B_norm, &B_norm_abs);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B * blk, &ONE_C,
    V, m_B*blk, V, m_B*blk, &ZERO_C, &R_norm, 1);
  mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
  mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);

  #if DEBUG == 1
    printf("relative residual frobenious norm: %1.4E\n", R_norm_abs/B_norm_abs);
  #endif

  if (R_norm_abs/B_norm_abs <= solver->tolerance)
  {
    return;
  }

  Br[0] = R_norm;
  for (MPF_Int i = 1; i < m_H; ++i)
  {
    Br[i] = ZERO_C;
  }

  /* main loop */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    tempf_complex = mpf_scalar_z_divide(ONE_C, R_norm);
    mpf_zscal(m_B * blk, &tempf_complex, V, 1);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0] = R_norm;
    W = &V[(m_B * blk)];
    Vprev = V;

    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      Vprev, m_B*blk, W, m_B * blk, &ZERO_C, &trace, 1);

    H[0] = trace;
    tempf_complex = mpf_scalar_z_invert_sign(trace);
    mpf_zaxpy(m_B*blk, &tempf_complex, Vprev, 1, W, 1);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      W, m_B*blk, W, m_B*blk, &ZERO_C, &H[1], 1);
    mpf_vectorized_z_sqrt(1, &H[1], &H[1]);
    tempf_complex = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m_B*blk, &tempf_complex, W, 1);

    for (MPF_Int j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m_B*blk)*(j+1)];
      Vprev = &V[(m_B*blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
        Vprev, m_B, W, m_B, &ZERO_C, Hblk, blk);

      trace = ZERO_C;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
      }
      H[ld_H*j+j] = trace;
      tempf_complex = mpf_scalar_z_invert_sign(trace);
      mpf_zaxpy(m_B*blk, &tempf_complex, Vprev, 1, W, 1);
      Vprev = &V[(m_B*blk)*(j-1)];
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*(j-1) + j]);
      mpf_zaxpy(m_B*blk, &tempf_complex, Vprev , 1, W, 1);

      H[ld_H*j+j-1] = H[ld_H*(j-1) + j];
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk,
        &ONE_C, W, m_B*blk, W, m_B*blk, &ZERO_C, &h_temp, 1);

      mpf_vectorized_z_sqrt(1, &h_temp, &h_temp);
      mpf_vectorized_z_abs(1, &h_temp, &h_abs);
      if ((h_abs <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }
      H[ld_H*j+j+1] = h_temp;
      tempf_complex = mpf_scalar_z_divide(ONE_C, H[m_H*j+j+1]);
      mpf_zscal(m_B*blk, &tempf_complex, W, 1);
    }

    /* solves system of equations using qr decomposition and constructs
       solution to the linear system of equations */
    mpf_qr_zsy_givens(H, Br, ld_H, n_H, 1);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[m_B*blk*i];
      mpf_zaxpy(m_B*blk, &Br[i], W, 1, X, 1);
    }

    memcpy(R, B, (sizeof *R)*m_B*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, COMPF_LEX_MINUS_ONE,
      A->handle, A->descr, sparse_layout, X, blk, m_B, ONE_C, R, m_B);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      R, m_B*blk, R, m_B*blk, &ZERO_C, &R_norm, 1);

    mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
    mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);

    #if DEBUG
      printf(">relative residual frobenious norm: %1.4E\n",
        R_norm_abs/B_norm_abs);
    #endif

    if (R_norm_abs/B_norm_abs <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m_B*blk);
    }
  }

  V = NULL;
  H = NULL;
  Br = NULL;
  W = NULL;
  Vprev = NULL;
  Vlast = NULL;
  R = NULL;
}

void mpf_zhe_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants*/
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble COMPLEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);

  /* solver context */
  double B_norm = 0.0;
  double R_norm = 0.0;
  MPF_ComplexDouble trace = ZERO_C;
  MPF_ComplexDouble tempf_complex;
  MPF_ComplexDouble h_temp;
  double h_abs;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_B = solver->ld;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = ld_H;
  MPF_Int blk = solver->batch;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X;

  /* cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[m_B*blk*(m_H+1)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  MPF_ComplexDouble *Hblk = &Br[m_H];

  /* handles on cpu memory */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Vlast = &V[(m_B*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;

  /* compute residual vectors block using initial approximation of solution
     block vectors */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V)*m_B*blk);
  mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle, A->descr,
    sparse_layout, X, blk, m_B, ONE_C, V, m_B);
  B_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, B, m_B);
  R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, V, m_B);

  #if DEBUG == 1
    printf("relative residual frobenious norm: %1.4E\n", R_norm/B_norm);
  #endif

  if (R_norm/B_norm <= solver->tolerance)
  {
    return;
  }
  Br[0] = mpf_scalar_z_init(R_norm, 0.0);
  for (MPF_Int i = 1; i < m_H; ++i)
  {
    Br[i] = ZERO_C;
  }

  /* main loop */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    tempf_complex = mpf_scalar_z_normalize(ONE_C, R_norm);
    mpf_zscal(m_B*blk, &tempf_complex, V, 1);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0] = mpf_scalar_z_init(R_norm, 0.0);
    W = &V[(m_B * blk)];
    Vprev = V;

    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
    mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      Vprev, m_B*blk, W, m_B*blk, &ZERO_C, &trace, 1);

    H[0] = trace;
    tempf_complex = mpf_scalar_z_invert_sign(trace);
    mpf_zaxpy(m_B*blk, &tempf_complex, Vprev, 1, W, 1);
    mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      W, m_B*blk, W, m_B*blk, &ZERO_C, &H[1], 1);
    mpf_vectorized_z_sqrt(1, &H[1], &H[1]);
    tempf_complex = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m_B*blk, &tempf_complex, W, 1);

    for (MPF_Int j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m_B,
        &ONE_C, Vprev, m_B, W, m_B, &ZERO_C, Hblk, blk);

      trace = ZERO_C;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
      }

      tempf_complex = mpf_scalar_z_invert_sign(trace);
      H[ld_H*j+j] = trace;
      tempf_complex = mpf_scalar_z_invert_sign(trace);
      mpf_zaxpy(m_B*blk, &tempf_complex, Vprev, 1, W, 1);
      Vprev = &V[(m_B*blk)*(j-1)];
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*(j-1)+j]);
      mpf_zaxpy(m_B*blk, &tempf_complex, Vprev , 1, W, 1);
      H[ld_H*j + j-1] = H[ld_H*(j-1) + j];

      h_abs = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, W, m_B);
      if ((h_abs <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }

      h_temp = mpf_scalar_z_init(h_abs, 0.0);
      H[ld_H*j+j+1] = h_temp;
      tempf_complex = mpf_scalar_z_divide(ONE_C, H[m_H*j+j+1]);
      mpf_zscal(m_B*blk, &tempf_complex, W, 1);
    }

    /* solves system of equations using qr decomposition and constructs
       solution to the linear system of equations */
    mpf_qr_zsy_givens_2(H, Br, m_H, n_H, 1);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[m_B*blk*i];
      mpf_zaxpy(m_B*blk, &Br[i], W, 1, X, 1);
    }
    memcpy(R, B, (sizeof *R)*m_B*blk);

    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle,
      A->descr, sparse_layout, X, blk, m_B, ONE_C, R, m_B);
    R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, R, m_B);

    #if DEBUG == 1
      printf(">relative residual frobenious norm: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm / B_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m_B*blk);
    }
  }
//  V = NULL;
//  H = NULL;
//  Br = NULL;
//  W = NULL;
//  Vprev = NULL;
//  Vlast = NULL;
//  R = NULL;
//  printf("end\n");
}

void mpf_zsy_gbl_gmres
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
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  /* solver context */
  double R_norm_abs = 0.0;
  double h_tempf_abs = 0.0;
  double B_norm_abs = 0.0;
  MPF_ComplexDouble B_norm = ZERO_C;
  MPF_ComplexDouble R_norm = ZERO_C;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble h_temp = ZERO_C;

  /* solver metadata */
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1);
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;
  MPF_Int size_V = n*blk*m_H;
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_Br = m_H;
  MPF_Int size_Hblk = blk*blk;
  MPF_ComplexDouble trace = ZERO_C;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* map mathematical objects to cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[size_V];
  MPF_ComplexDouble *Br = &H[size_H];
  MPF_ComplexDouble *Hblk = &Br[size_Br];
  MPF_ComplexDouble *tempf_matrix = &Hblk[size_Hblk];
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *const Vlast = &V[(n*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;

  /* computes residual vectors block using initial approximation of solution
     block vectors */
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, V, n);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
    B, n*blk, B, n*blk, &ZERO_C, &B_norm, 1);
  mpf_vectorized_z_sqrt(1, &B_norm, &B_norm);
  mpf_vectorized_z_abs(1, &B_norm, &B_norm_abs);

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; k++)
  {
    /* first iteration */
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      V, n*blk, V, n*blk, &ZERO_C, &R_norm, 1);

    mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
    //R_norm = mpf_scalar_z_divide(R_norm, B_norm);
    tempf_complex = mpf_scalar_z_divide(ONE_C, R_norm);
    mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);
    if (R_norm_abs <= solver->tolerance)  /* checks terminating condition */
    {
      return;
    }

    mpf_zscal(n*blk, &tempf_complex, V, 1);
    Br[0] = B_norm;
    for (MPF_Int i = 1; i < m_H; i++)
    {
      Br[i] = ZERO_C;
    }

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C, W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Vprev = &V[(n*blk)*i];
        mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
          &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, blk);

        trace = ZERO_C;
        for (MPF_Int t = 0; t < blk; ++t)
        {
          trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
        }
        H[ld_H*j+i] = trace;
        tempf_complex = mpf_scalar_z_invert_sign(trace);
        mpf_zaxpy(n*blk, &tempf_complex, Vprev, 1, W, 1);
      }

      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk,
        &ONE_C, W, n*blk, W, n*blk, &ZERO_C, &h_temp, 1);
      mpf_vectorized_z_sqrt(1, &h_temp, &h_temp);
      mpf_vectorized_z_abs(1, &h_temp, &h_tempf_abs);

      if (h_tempf_abs <= 1e-12)
      {
        inner_iterations = j+1;
        m_H = (inner_iterations+1);
        n_H = inner_iterations;
        break;
      }
      tempf_complex = mpf_scalar_z_divide(ONE_C, h_temp);
      mpf_zscal(n*blk, &tempf_complex, W, 1);
      H[ld_H*j+j+1] = h_temp;
    }

    /* solve system of equations using qr decomposition */
    mpf_qr_zge_givens(H, Br, m_H, n_H, blk, tempf_matrix);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[n*blk*i];
      mpf_zaxpy(n*blk, &Br[i], W, 1, X, 1);
    }
    memcpy(R, B, sizeof(MPF_ComplexDouble)*n*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      R, n*blk, R, n*blk, &ZERO_C, &R_norm, 1);

    mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
    R_norm = mpf_scalar_z_divide(R_norm, B_norm);
    mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);

    #if DEBUG == 1
      printf("norm_frobenious_residual: %1.4E\n", R_norm_abs);
    #endif

    if ((R_norm_abs <= solver->tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *R)*n*blk);  /* copies residual to V(:, block(1)) */
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mpf_cpu_csy_global_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_Complex ONE_C = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex MINUS_ONE_C = mpf_scalar_c_init(-1.0, 0.0);

  /* solver context */
  float R_norm_abs = 0.0;
  float h_tempf_abs = 0.0;
  MPF_Complex B_norm = ZERO_C;
  MPF_Complex R_norm = ZERO_C;
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Complex h_temp = ZERO_C;

  MPF_Complex *B = (MPF_Complex*)B_dense->data;
  MPF_Complex *X = (MPF_Complex*)X_dense->data;

  /* solver solver->ata */
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = solver->iterations+1;
  MPF_Int size_V = n*blk*m_H;
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_Br = m_H;
  //MPF_Int size_Hblk = blk*blk;
  MPF_Complex trace = ZERO_C;

  /* map mathematical objects to cpu memory */
  MPF_Complex *V = (MPF_Complex*)solver->inner_mem;
  MPF_Complex *H = &V[size_V];
  MPF_Complex *Br = &H[size_H];
  MPF_Complex *Hblk = &Br[size_Br];
  //MPF_Complex *tempf_matrix = &Hblk[size_Hblk];
  MPF_Complex *W = NULL;
  MPF_Complex *Vprev = NULL;
  MPF_Complex *const Vfirst = V;
  MPF_Complex *const Vlast = &V[(n*blk)*(MPF_Int)solver->iterations];
  MPF_Complex *R = Vlast;

  /* computes residual vectors block using initial approximation of solution
     block vectors */
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, V, n);
  mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n * blk, &ONE_C,
    B, n*blk, B, n*blk, &ZERO_C, &B_norm, 1);
  mpf_vectorized_c_sqrt(1, &B_norm, &B_norm);  //@BUG: this might require a separate variable as output

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    /* first iteration */
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      V, n*blk, Vfirst, n*blk, &ZERO_C, &R_norm, 1);

    mpf_vectorized_c_sqrt(1, &R_norm, &R_norm);
    R_norm = mpf_scalar_c_divide(R_norm, B_norm);
    mpf_vectorized_c_abs(1, &R_norm, &R_norm_abs);
    if (R_norm_abs <= solver->tolerance)  /* checks terminating condition */
    {
      return;
    }

    tempf_complex = mpf_scalar_c_divide(ONE_C, R_norm);
    mpf_cscal(n*blk, &tempf_complex, V, 1);
    Br[0] = R_norm;
    for (MPF_Int i = 1; i < m_H; ++i)
    {
      Br[i] = ZERO_C;
    }

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      V = &V[(n*blk)*j];
      mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Vprev = &V[(n*blk)*i];
        mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
          &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, blk);

        trace = ZERO_C;
        for (MPF_Int t = 0; t < blk; ++t)
        {
          trace = mpf_scalar_c_add(trace, Hblk[blk*t+t]);
        }
        H[ld_H*j+i] = trace;
        tempf_complex = mpf_scalar_c_invert_sign(trace);
        mpf_caxpy(n*blk, &tempf_complex, Vprev, 1, W, 1);
      }
      mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk,
        &ONE_C, W, n*blk, W, n*blk, &ZERO_C, &h_temp, 1);

      mpf_vectorized_c_sqrt(1, &h_temp, &h_temp);
      mpf_vectorized_c_abs(1, &h_temp, &h_tempf_abs);
      if (h_tempf_abs <= 1e-12)
      {
        inner_iterations = j+1;
        m_H = (j+1);
        n_H = j;
        break;
      }

      tempf_complex = mpf_scalar_c_divide(ONE_C, h_temp);
      mpf_zscal(n*blk, &tempf_complex, W, 1);
      H[ld_H*j+j+1] = h_temp;
    }

    /* solve system of equations using qr decomposition */
    //@BUG: need to change function to include ld_H
    //mpf_qr_givens_cge_factorize(H, Br, m_H, n_H, blk, tempf_matrix); 
    mpf_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[n*blk*i];
      mpf_zaxpy(n*blk, &Br[i], W, 1, X, 1);
    }
    memcpy(R, B, sizeof(float)*n*blk);

    mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      R, n*blk, R, n*blk, &ZERO_C, &R_norm, 1);
    mpf_vectorized_c_sqrt(1, &R_norm, &R_norm);
    R_norm = mpf_scalar_c_divide(R_norm, B_norm);
    mpf_vectorized_c_abs(1, &R_norm, &R_norm_abs);

    #if DEBUG
      printf("norm_frobenious_residual: %1.4E\n", R_norm_abs);
    #endif

    if ((R_norm_abs <= solver->tolerance) || (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      /* copies residual to V(:, block(1)) */
      memcpy(Vfirst, R, (sizeof *R)*n*blk);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mpf_gbl_gmres_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
)
{
  MPF_Solver* solver = &context->solver;
  MPF_Int ld = context->A.m;
  solver->ld = ld;
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;
  solver->framework = MPF_SOLVER_FRAME_MPF;
  context->args.n_inner_solve = 4;

  if ((solver->precond_type == MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DGE_GMRES;
      solver->inner_function = &mpf_dge_gbl_gmres;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_GMRES;
      solver->inner_function = &mpf_zsy_gbl_gmres;
      solver->device = MPF_DEVICE_CPU;
  }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZGE_GMRES;
      solver->inner_function = &mpf_zge_gbl_gmres;
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
  solver->inner_get_mem_size_function = &mpf_gbl_gmres_get_mem_size;
}

void mpf_gbl_gmres_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int iterations = solver->iterations;
  MPF_Int blk = solver->batch;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
      (n*blk                      /* size_B */
      +n*blk                      /* size_X */
      +n*blk*(iterations+1)       /* size_V */
      +(iterations+1)*iterations  /* size_H */
      +iterations+1               /* size_Br */
      +blk*blk                    /* size_Hblk */
      +(iterations+1)*2);         /*size_tempf_matrix */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      (n*blk                      /* size_B */
      +n*blk                      /* size_X */
      +n*blk*(iterations+1)       /* size_V */
      +(iterations+1)*iterations  /* size_H */
      +iterations+1               /* size_Br */
      +blk*blk                    /* size_Hblk */
      +(iterations+1)*2);         /*size_tempf_matrix */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      (n*blk*(iterations+1)       /* size_V */
      +(iterations+1)*iterations  /* size_H */
      +iterations+1               /* size_Br */
      +blk*blk                    /* size_Hblk */
      +(iterations+1)*2);         /*size_tempf_matrix */
  }
}

void mpf_gbl_lanczos_init
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
    solver->inner_type = MPF_SOLVER_DSY_GBL_LANCZOS;
    solver->inner_function = &mpf_dsy_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_GBL_LANCZOS;
    solver->inner_function = &mpf_zsy_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_GBL_LANCZOS;
    solver->inner_function = &mpf_zhe_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  solver->inner_alloc_function = &mpf_krylov_alloc;
  solver->inner_free_function = &mpf_krylov_free;
  mpf_gbl_lanczos_get_mem_size(solver);
}

void mpf_gbl_lanczos_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int iterations = solver->iterations;
  MPF_Int blk = solver->batch;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
        (n*blk*(iterations+1)      /* size_V */
        +iterations*iterations     /* size_H */
        +iterations                /* size_Br */
        +blk*blk);                 /* size_Hblk */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
        (n*blk*(iterations+1)      /* size_V */
        +iterations*iterations     /* size_H */
        +iterations                /* size_Br */
        +blk*blk);                 /* size_Hblk */
  }
}

