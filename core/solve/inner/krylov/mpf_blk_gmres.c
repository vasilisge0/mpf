#include "mpf.h" 

/* (2) _block_gmres_real */
void mpf_dge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense, /* (input) rhs block of vectors */
  MPF_Dense *X_dense  /* (output) solution of AX = B */
)
{
  /* context */
  MPF_Int n = solver->ld;
  double H_IJ_norm = 0.0;
  double R_norm_max = 0.0;
  double R_norm = 0.0;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* solver->*/
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1)*blk;  /* rows_H */
  MPF_Int n_H = solver->iterations*blk;      /* cols_H */
  MPF_Int ld_H = (solver->iterations+1)*blk; /* rows_H */
  MPF_Layout layout = MPF_COL_MAJOR;  /* layout of B and X for dense BLAS */
  MPF_LayoutSparse sparse_layout;    /* layout of B and X for sparse BLAS */

  /* memory cpu */
  double *V = (double*)solver->inner_mem;
  double *H = &V[n*blk*(solver->iterations+1)];
  double *Br = &H[m_H*n_H]; /* reduced rhs = V^T*B*/
  double *reflectors_array = &Br[m_H*blk];
  double *Bnorms_array = &reflectors_array[blk];

  /* handles on allcoated cpu memory */
  double *W = NULL;
  double *Hblk = NULL;
  double *Vprev = NULL;
  double *R = &V[n*n_H];  /* residual block of vectors */

  /* first iteration (initialize V(:, 0)) */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(V, B, (sizeof *V)*n*blk);   /* copies V0 <- B */
  mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0,   /*  <- */
    A->handle, A->descr, sparse_layout, X, blk, n, 1.0, V, n);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    Bnorms_array[i] = cblas_dnrm2(n, &((double*)solver->B.data)[n*i], 1);
    R_norm = cblas_dnrm2(n, &V[n*i], 1)/Bnorms_array[i];
    if (R_norm > R_norm_max)
    {
      R_norm_max = R_norm;
    }
  }

  if (R_norm_max <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer iterations (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, ld_H);
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, blk, Br, ld_H);
    mpf_dgeqrf(LAPACK_COL_MAJOR, n, blk, V, n, reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, V, n, Br, m_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, n, blk, blk, V, n, reflectors_array);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle,
        A->descr, sparse_layout, Vprev, blk, n, 0.0, W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Hblk = &H[(m_H*blk)*j+blk*i];
        Vprev = &V[(n*blk)*i];
        mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n, 1.0,
          W, n, Vprev , n, 0.0, Hblk, m_H);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, blk, -1.0,
          Vprev, n, Hblk, m_H, 1.0, W, n);
      }
      Hblk = &H[m_H*blk*j + blk*(j+1)];
      mpf_dgeqrf(LAPACK_COL_MAJOR, n, blk, W, n, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, //change arg 3 to blk(?)
                     blk, W, n, Hblk, m_H);
      H_IJ_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, m_H);
      if ((H_IJ_norm <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = blk*(inner_iterations);
        n_H = blk*(inner_iterations);
        break;
      }
      mpf_dorgqr(LAPACK_COL_MAJOR, n, blk, blk, W, n, reflectors_array);
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */

    mpf_block_qr_dge_givens(m_H, blk, H, ld_H, Br, ld_H, blk);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, 1.0, H, ld_H, Br, ld_H);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, n_H, 1.0, V,
      n, Br, ld_H, 1.0, X, n);

    /* computes residual */
    memcpy(R, B, (sizeof *R)*n*blk);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, n, 1.0, R, n);
    R_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      R_norm = mpf_dnrm2(n, &R[n*i], 1)/Bnorms_array[i];
      if (R_norm > R_norm_max)
      {
        R_norm_max = R_norm;
      }
    }

    #if DEBUG == 1
      printf("   max_relative residual: %1.4E\n", R_norm_max);
    #endif

    if (R_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = (solver->iterations+1)*blk;
      n_H = solver->iterations*blk;
    }
  }

  Hblk = NULL;
  V = NULL;
  W = NULL;
  R = NULL;
  V = NULL;
  H = NULL;
  Br = NULL;
  reflectors_array = NULL;
  Bnorms_array = NULL;
}

void mpf_sge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* solver context */
  float H_IJ_norm = 0.0;
  float R_norm_max = 0.0;
  float R_norm = 0.0;

  /* solver solver->*/
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = 1+solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1)*blk;
  MPF_Int n_H = solver->iterations * blk;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  float *B = (float*)B_dense->data;
  float *X = (float*)X_dense->data;

  /* maps mathematical objects to cpu memory */
  float *V = (float*)solver->inner_mem;
  float *H = &V[n*blk*(solver->iterations+1)];
  float *Br = &H[m_H*n_H];
  float *reflectors_array = &Br[m_H*blk];
  float *B_norms_array    = &reflectors_array[blk];

  /* handles */
  float *W = NULL;
  float *Vprev = NULL;
  float *Hblk = NULL;
  float *Vlast = &V[n*n_H];
  float *R = Vlast;
  mpf_convert_layout_to_sparse(layout, &sparse_layout);

  /* first inner iteration (initialize V(:, 0)) */
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_s_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, n, 1.0, V, n);

  /* finds maximum relative residual norm */
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_snrm2(n, &((float*)B)[n*i], 1);
    R_norm = mpf_snrm2(n, &V[n*i], 1)/B_norms_array[i];
    if (R_norm > R_norm_max)
    {
      R_norm_max = R_norm;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %lf\n", R_norm_max);
  #endif

  if (R_norm_max <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer iterations (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_s_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
    mpf_zeros_s_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_sgeqrf(LAPACK_COL_MAJOR, n, blk, V, n, reflectors_array);
    mpf_slacpy(LAPACK_COL_MAJOR, 'U', blk, blk, V, n, Br, m_H);
    mpf_sorgqr(LAPACK_COL_MAJOR, n, blk, blk, V, n, reflectors_array);

    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, n, 0.0, W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Hblk = &H[(m_H*blk)*j+blk*i];
        Vprev = &V[(n*blk)*i];
        mpf_sgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, blk, blk,
          n, 1.0 , W, n, Vprev, n, 0.0, Hblk, m_H);
        mpf_sgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_NO_TRANS, MPF_BLAS_NO_TRANS, n, blk,
          blk, -1.0, Vprev, n, Hblk, m_H, 1.0, W, n);
      }
      Hblk = &H[m_H*blk*j + blk*(j+1)];

      //mpf_sgeqrf(LAPACK_COL_MAJOR, n, blk, W, B, reflectors_array);
      //mpf_slacpy(LAPACK_COL_MAJOR, 'U', B, // @BUG: change arg 3 to blk(?)
      // blk, W, n, Hblk, m_H);
      H_IJ_norm = mpf_slange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, n);
      if ((H_IJ_norm <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = blk*(inner_iterations+1);
        n_H = blk*inner_iterations;
        break;
      }
      mpf_sorgqr(LAPACK_COL_MAJOR, n, blk, blk, W, n, reflectors_array);
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    mpf_block_qr_sge_givens(n_H, blk, H, m_H, Br, m_H, blk);
    mpf_strsm(MPF_COL_MAJOR, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, 1.0, H, n, Br, m_H);
    mpf_sgemm(MPF_COL_MAJOR, CblasNoTrans, CblasNoTrans, n, blk, n_H, 1.0, V, n,
      Br, m_H, 1.0, X, n);

    /* computes residual norm */
    memcpy(R, B, (sizeof *R)*n*blk);
    mpf_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, n, 1.0, R, n);
    R_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      R_norm = mpf_snrm2(n, &R[n*i], 1) / B_norms_array[i];
      if (R_norm > R_norm_max)
      {
        R_norm_max = R_norm;
      }
    }

    #if DEBUG
      printf("   max_relative residual: %1.4E\n", R_norm_max);
      printf("              iterations: %d\n", solver->iterations);
      printf("max_num_outer_iterations: %d\n", outer_iterations);
    #endif

    /* checks if maximum relative residual norm is small enough */
    if (R_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = (solver->iterations+1)*blk;
      n_H = solver->iterations*blk;
    }

  }

  Hblk = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  V = NULL;
  H = NULL;
  Br = NULL;
  reflectors_array = NULL;
  B_norms_array = NULL;
}

void mpf_zge_block_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  double H_IJ_norm = 0.0;
  //MPF_ComplexDouble tempf_complex = ZERO_C;
  double R_norm = 0.0;
  double R_norm_max = 0.0;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* solver->*/
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int ld_H = (solver->iterations+1)*blk;
  MPF_Int m_H = (solver->iterations+1)*blk;
  MPF_Int n_H = solver->iterations*blk;

  /* maps mathematical objects to cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[n*blk*(solver->iterations+1)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  double *B_norms_array = (double*)&Br[m_H*blk];

  /* other memory handles */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Vlast = &V[n*n_H];
  MPF_ComplexDouble *R = Vlast;

  /* compute residual vectors block using initial approximation of solution
     block vectors */
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(R, B, (sizeof *R)*n*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
  R_norm_max = 0.0;

  //tempf_complex = ZERO_C;
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dznrm2(n, &B[n*i], 1);
    R_norm = mpf_dznrm2(n, &R[n*i], 1)/B_norms_array[i];
    if (R_norm > R_norm_max)
    {
      R_norm_max = R_norm;
    }
  }

  #if DEBUG == 1
    printf("[start] max relative residual: %1.4E\n", R_norm_max);
  #endif

  if (R_norm_max <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_gram_schmidt_zhe(n, blk, R, Br, ld_H);
    memcpy(V, R, (sizeof *V)*n*blk);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; j++)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);

      for (MPF_Int i = 0; i < j + 1; i++)
      {
        Hblk = &H[(m_H*blk)*j + blk*i];
        Vprev = &V[(n*blk)*i];
        mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, n,
         &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, m_H);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, blk,
          &MINUS_ONE_C, Vprev, n, Hblk, m_H, &ONE_C, W, n);
      }
      Hblk = &H[ld_H*blk*j + blk*(j+1)];

      /* reorthogonalize a batch of columns */
      mpf_gram_schmidt_zhe(n, blk, W, Hblk, ld_H);

      H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if ((H_IJ_norm <= 1e-12) || (j == inner_iterations-1))
      {
          inner_iterations = j+1;
          m_H = blk*(inner_iterations+1);
          n_H = blk*inner_iterations;
          break;
      }
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    // @BUG: test if this contains bugs for zhe versions
    mpf_block_qr_zge_givens(H, Br, m_H, n_H, blk, blk);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, &ONE_C, H, ld_H, Br, ld_H);
    mpf_matrix_z_announce(Br, 10, blk, n, "Br (af)");
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, n_H,
      &ONE_C, V, n, Br, ld_H, &ONE_C, X, n);
    memcpy(R, B, (sizeof *R)*n*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    R_norm_max = 0.0;

    //tempf_complex = ZERO_C;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      R_norm = mpf_dznrm2(n, &R[n*i], 1)/B_norms_array[i];
      if (R_norm > R_norm_max)
      {
        R_norm_max = R_norm;
      }
    }

    #if DEBUG == 1
      printf("k: %d, max_relative residual: %1.4E\n", k, R_norm_max);
    #endif

    if (R_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = (solver->iterations+1)*blk;
      n_H = solver->iterations*blk;
    }
  }

  Hblk = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  Br = NULL;
  V = NULL;
  H = NULL;
  V = NULL;
  B_norms_array = NULL;
}

void mpf_zge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  /* context */
  MPF_Int n = solver->ld;
  MPF_ComplexDouble H_IJ_norm = ZERO_C;
  double H_IJ_norm_abs = 0.0;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  double R_norm_abs = 0.0;
  double R_norm_abs_max = 0.0;

  /* meta */
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int ld_H = (solver->iterations+1)*blk;
  MPF_Int m_H = (solver->iterations+1)*blk;
  MPF_Int n_H = solver->iterations*blk;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* maps mathematical objects to cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[n*blk*(solver->iterations+1)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  MPF_ComplexDouble *B_norms_array = &Br[m_H*blk];
  /* other memory handles */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Vfirst = V;
  MPF_ComplexDouble *Vlast = &V[n*n_H];
  MPF_ComplexDouble *R = Vlast;

  /* compute residual vectors block using initial approximation of
     solution block vectors */
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(R, B, (sizeof *R)*n*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
  R_norm_abs_max = 0.0;
  tempf_complex = ZERO_C;

  for (MPF_Int i = 0; i < blk; ++i)
  {
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
             &R[n*i], n, &R[n*i], n, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
             &B[n*i], n, &B[n*i], n, &ZERO_C, &B_norms_array[i], 1);
    mpf_vectorized_z_sqrt(1, &B_norms_array[i], &B_norms_array[i]);
    tempf_complex = mpf_scalar_z_divide(tempf_complex, B_norms_array[i]);
    mpf_vectorized_z_abs(1, &tempf_complex, &R_norm_abs);
    if (R_norm_abs > R_norm_abs_max)
    {
      R_norm_abs_max = R_norm_abs;
    }
  }

  #if DEBUG == 1
    printf("[start] max relative residual: %1.4E\n", R_norm_abs_max);
  #endif

  if (R_norm_abs_max <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_gram_schmidt_zge(n, blk, R, Br, m_H);
    memcpy(V, R, (sizeof *V)*n*blk);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Hblk = &H[(m_H*blk)*j + blk*i];
        Vprev = &V[(n*blk)*i];
        mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
          &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, m_H);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, blk,
          &MINUS_ONE_C, Vprev, n, Hblk, m_H, &ONE_C, W, n);
      }
      Hblk = &H[m_H*blk*j + blk*(j+1)];
      mpf_gram_schmidt_zge(n, blk, W, Hblk, m_H);

      H_IJ_norm = ZERO_C;
      for (MPF_Int i = 0; i < blk; ++i)
      {
        mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, blk,
          &ONE_C, &Hblk[ld_H*i], ld_H, &Hblk[ld_H*i], ld_H,
          &ONE_C, &H_IJ_norm, 1);
      }

      mpf_vectorized_z_sqrt(1, &H_IJ_norm, &H_IJ_norm);
      mpf_vectorized_z_abs(1, &H_IJ_norm, &H_IJ_norm_abs);
      if ((H_IJ_norm_abs <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = blk*(inner_iterations+1);
        n_H = blk*inner_iterations;
        break;
      }
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    mpf_block_qr_zge_givens(H, Br, m_H, n_H, blk, blk);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, &ONE_C, H, m_H, Br, m_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, n_H,
      &ONE_C, Vfirst, n, Br, m_H, &ONE_C, X, n);

    memcpy(R, B, (sizeof *R)*n*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);

    R_norm_abs_max = 0.0;
    tempf_complex = ZERO_C;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
        &R[n*i], n, &R[n*i], n, &ZERO_C, &tempf_complex, 1);
      mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
      tempf_complex = mpf_scalar_z_divide(tempf_complex, B_norms_array[i]);
      mpf_vectorized_z_abs(1, &tempf_complex, &R_norm_abs);
      if (R_norm_abs > R_norm_abs_max)
      {
        R_norm_abs_max = R_norm_abs;
      }
    }
    #if DEBUG == 1
        printf("k: %d, max_relative residual: %1.4E\n", k, R_norm_abs_max);
    #endif

    if (R_norm_abs_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = (solver->iterations+1)*blk;
      n_H = solver->iterations*blk;
    }
  }

  Hblk = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  Br = NULL;
  V = NULL;
  H = NULL;
  V = NULL;
  B_norms_array = NULL;
}

void mpf_csy_block_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex ONE_C = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex MINUS_ONE_C = mpf_scalar_c_init(-1.0, 0.0);

  /* context */
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Complex H_IJ_norm = ZERO_C;
  float H_IJ_norm_abs = 0.0;
  MPF_Complex tempf_complex = ZERO_C;
  float R_norm_abs = 0.0;
  float R_norm_abs_max = 0.0;

  /* solver */
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1)*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Int size_V = n*blk*(solver->iterations+1);
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_Br = m_H*blk;

  MPF_Complex *B = (MPF_Complex*)B_dense->data;
  MPF_Complex *X = (MPF_Complex*)X_dense->data;

  /* map mathematical objects to cpu memory */
  MPF_Complex *V = (MPF_Complex*)solver->inner_mem;
  MPF_Complex *H = &V[size_V];
  MPF_Complex *Br = &H[size_H];
  MPF_Complex *B_norms_array = &Br[size_Br];
  /* other handles */
  MPF_Complex *W = NULL;
  MPF_Complex *Vprev = NULL;
  MPF_Complex *Hblk = NULL;
  MPF_Complex *Vlast = &V[n*n_H];
  MPF_Complex *R = Vlast;

  /* compute residual vectors block using initial approximation of solution
     block vectors */
  mpf_zeros_c_set(MPF_COL_MAJOR, m_H, n_H, H, ld_H);
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, V, n);
  R_norm_abs_max = 0.0;

  for (MPF_Int i = 0; i < blk; ++i)
  {
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
      &R[n*i], n, &R[n*i], n, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_c_sqrt(1, &tempf_complex, &tempf_complex);
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
      &B[n*i], n, &B[n*i], n, &ZERO_C, &B_norms_array[i], 1);

    mpf_vectorized_c_sqrt(1, &B_norms_array[i], &B_norms_array[i]);
    tempf_complex = mpf_scalar_c_divide(tempf_complex, B_norms_array[i]);
    mpf_vectorized_c_abs(1, &tempf_complex, &R_norm_abs);
    if (R_norm_abs > R_norm_abs_max)
    {
        R_norm_abs_max = R_norm_abs;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %1.4E\n", R_norm_abs_max);
  #endif

  if (R_norm_abs_max <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer iterations */
  for (k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_c_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_gram_schmidt_cge(n, blk, R, Br, m_H);
    /* inner iterations */
    for (j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n * blk)*(j+1)];
      Vprev = &V[(n * blk)*j];
      mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);
      for (i = 0; i < j+1; ++i)
      {
        Hblk = &H[(m_H*blk)*j + blk*i];
        Vprev = &V[(n*blk)*i];
        mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
          &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, m_H);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, blk,
          &MINUS_ONE_C, Vprev, n, Hblk, m_H, &ONE_C, W, n);
      }

      Hblk = &H[m_H*blk*j + blk*(j+1)];
      mpf_gram_schmidt_cge(n, blk, W, Hblk, m_H);
      for (i = 0; i < blk; ++i)
      {
        mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, blk,
          &ONE_C, &Hblk[ld_H*i], ld_H, &Hblk[ld_H*i], ld_H, &ONE_C,
          &H_IJ_norm, 1);
      }
      mpf_vectorized_c_sqrt(1, &H_IJ_norm, &H_IJ_norm);
      mpf_vectorized_c_abs(1, &H_IJ_norm, &H_IJ_norm_abs);
      if ((H_IJ_norm_abs <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j; //@BUG: maybe this should be j+1
        m_H = blk*(j+1);
        n_H = blk*j;
        break;
      }
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    mpf_block_qr_cge_givens(H, Br, m_H, n_H, blk, blk);
    mpf_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, &ONE_C, H, m_H, Br, m_H);
    mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, blk, n_H,
      &ONE_C, V, n, Br, m_H, &ONE_C, X, n);
    memcpy(R, B, (sizeof *R) * n * blk);
    mpf_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    R_norm_abs_max = 0.0;
    for (i = 0; i < blk; i++)
    {
      mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
        &R[n*i], n, &R[n*i], n, &ZERO_C, &tempf_complex, 1);
      mpf_vectorized_c_sqrt(1, &tempf_complex, &tempf_complex);
      tempf_complex = mpf_scalar_c_divide(tempf_complex, B_norms_array[i]);
      mpf_vectorized_c_abs(1, &tempf_complex, &R_norm_abs);
      if (R_norm_abs > R_norm_abs_max)
      {
        R_norm_abs_max = R_norm_abs;
      }
    }
    #if DEBUG == 1
      printf ("max_relative residual: %1.4E\n", R_norm_abs_max);
    #endif
    #if MODE == MPF_PROFILE

    #endif
    if (R_norm_abs <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = (solver->iterations+1)*blk;
      n_H = (solver->iterations)*blk;
    }
  }

  Hblk = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  Br = NULL;
  V = NULL;
  H = NULL;
  V = NULL;
  B_norms_array = NULL;
}

void mpf_blk_gmres_init
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
      solver->inner_function = &mpf_dge_blk_gmres;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_GMRES;
      solver->inner_function = &mpf_zge_blk_gmres;
      solver->device = MPF_DEVICE_CPU;
  }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZGE_GMRES;
      solver->inner_function = &mpf_zge_blk_gmres;
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
  solver->inner_get_mem_size_function = &mpf_blk_gmres_get_mem_size;
}

void mpf_blk_gmres_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int iterations = solver->iterations;
  MPF_Int blk = solver->batch;
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
      (n*blk*2                            /* size_B and size_X */
      +n*blk*(iterations+1)               /* size_V */
      +(iterations+1)*blk*iterations*blk  /* size_H */
      +(iterations+1)*blk*blk             /* size_Br */
      +blk                                /* size_reflectors */
      +blk);                              /* size_B_norms */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      (n*blk*(iterations+1)               /* size_V */
      +(iterations+1)*blk*iterations*blk  /* size_H */
      +(iterations+1)*blk*blk             /* size_Br */
      +blk                                /* size_reflectors */
      +blk);                              /* size_B_norms */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      //(n*blk*2                          /* size_B and size_X */
      (n*blk*(iterations+1)               /* size_V */
      +(iterations+1)*blk*iterations*blk  /* size_H */
      +(iterations+1)*blk*blk             /* size_Br */
      +blk                                /* size_reflectors */
      +blk);                              /* size_B_norms */
  }
}
