#include "mpf.h"

/* (3) _global_gmres_real */
void mpf_dge_gbl_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  double B_norm = 0.0;
  double R_norm = 0.0;
  double trace = 0.0;
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Int t = 0;

  /* solver->*/
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1);
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout = MPF_SPARSE_COL_MAJOR;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* access solver memory */
  double *V = (double*)solver->inner_mem;
  double *H = &V[n*m_H*blk];
  double *Hblk = &H[m_H*n_H];
  double *Br = &Hblk[blk*blk];
  double *tempf_matrix = &Br[m_H];
  /* handles to cpu memory */
  double *W = NULL;
  double *Vprev = NULL;
  double *Vlast = &V[(n*blk)*solver->iterations];
  double *R = Vlast;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, n, 1.0, V, n);
  B_norm = mpf_dlange(MPF_BLAS_COL_MAJOR, 'F', n, blk, X, n);
  R_norm = mpf_dlange(MPF_BLAS_COL_MAJOR, 'F', n, blk, V, n);
  #if DEBUG == 1
    printf("relative residual frobenious norm: %1.4E\n", R_norm/B_norm);
  #endif
  if (R_norm/B_norm <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    Br[0] = B_norm;
    for (i = 1; i < m_H; ++i)
    {
      Br[i] = 0.0;
    }
    mpf_dscal(n*blk, 1/R_norm, V, 1);
    printf("V[0]: %1.4E\n", V[0]);
    printf("R.norm: %1.4E\n", R_norm);
    /* inner iterations */
    for (j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, n, 0.0, W, n);

      for (i = 0; i < j+1; ++i)
      {
        Vprev = &V[(n*blk)*i];
        mpf_dgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, blk,
          blk, n, 1.0, W, n, Vprev, n, 0.0, Hblk, blk);
        trace = 0.0;
        for (t = 0; t < blk; ++t)
        {
          trace += Hblk[blk*t+t];
        }
        H[ld_H*j+i] = trace;
        mpf_daxpy(n*blk, -trace, Vprev, 1, W, 1);
      }
      H[ld_H*j+j+1] = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, blk, W, n);
      if ((fabs(H[ld_H*j+j+1]) <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = (inner_iterations+1);
        n_H = inner_iterations;
        break;
      }
      mpf_dscal(n*blk, 1/H[ld_H*j+j+1], W, 1);
    }

    /* solves system of equations using qr decomposition */
    mpf_qr_givens_dge(n_H, 1, H, m_H, Br, ld_H, tempf_matrix);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, Br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      W = &V[n*blk*i];
      mpf_daxpy(n*blk, Br[i], W, 1, X, 1);
    }
    memcpy(R, B, (sizeof *B)*n*blk);
    mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, n, 1.0, R, n);
    R_norm = mpf_dlange(CblasColMajor, 'F', m_H, blk, R, n);

    #if DEBUG == 1
      printf("norm_frobenious_residual: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm/B_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*n*blk);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  H = NULL;
  V = NULL;
  Br = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  tempf_matrix = NULL;
}

void mpf_sge_gbl_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* context */
  float B_norm = 0.0;
  float R_norm = 0.0;
  float trace = 0.0;

  /* solver->*/
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  float *B = (float*)B;
  float *X = (float*)X;

  /* mappings to cpu accessed memory */
  float *V = (float*)solver->inner_mem;
  float *H = &V[n*m_H*blk];
  float *Hblk = &H[m_H*n_H];
  float *Br = &Hblk[blk*blk];
  float *tempf_matrix = &Br[m_H];

  /* handles on cpu memory */
  float *W = NULL;
  float *Vprev = NULL;
  float *Vlast = &V[(n*blk)*solver->iterations];
  float *R = Vlast;
  mpf_convert_layout_to_sparse(layout, &sparse_layout);

  /* first iteration */
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, n, 1.0, V, n);
  B_norm = mpf_slange(MPF_BLAS_COL_MAJOR, 'F', n, blk, B, n);
  R_norm = mpf_slange(MPF_COL_MAJOR, 'F', n, blk, V, n);

  #if DEBUG == 1
      printf("relative residual frobenious norm: %lf\n", R_norm/B_norm);
  #endif


  if (R_norm/B_norm <= solver->tolerance)
  {
      return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; k++)
  {
    Br[0] = B_norm;
    for (MPF_Int i = 1; i < m_H; i++)
    {
        Br[i] = 0.0;
    }

    mpf_sscal(n*blk, 1/R_norm, V, 1);
    for (MPF_Int j = 0; j < inner_iterations; j++)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mpf_sparse_s_mm(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr,
       sparse_layout, Vprev,
                     blk, n, 0.0, W, n);

      for (MPF_Int i = 0; i < j+1; i++)
      {
        Vprev = &V[(n*blk)*i];
        mpf_sgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, blk, blk, 
          n, 1.0, W, n, Vprev, n, 0.0, Hblk, blk);

        trace = 0.0;
        for (MPF_Int t = 0; t < blk; t++)
        {
            trace += Hblk[blk*t+t];
        }
        H[ld_H*j+i] = trace;
        mpf_saxpy(n*blk, -trace, Vprev, 1, W, 1);
      }
      H[ld_H*j+j+1] = mpf_slange(LAPACK_COL_MAJOR, 'F', n, blk, W, n);

      if ((fabs(H[ld_H*j+j+1]) <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = (j+1);
        n_H = j;
        break;
      }

      mpf_sscal(n*blk, 1/H[ld_H*j+j+1], W, 1);
    }

    /* solve system of equations using qr decomposition */
    mpf_qr_givens_sge(n_H, 1, H, m_H, Br, ld_H, tempf_matrix);
    mpf_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, Br, ld_H);

    /* reconstruction */
    for (MPF_Int i = 0; i < n_H; i++)
    {
      W = &V[n*blk*i];
      mpf_saxpy(n*blk, Br[i], W, 1, X, 1);
    }

    memcpy(R, B, sizeof(float)*n*blk);
    mpf_sparse_s_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, n, 1.0, R, n);
    R_norm = mpf_slange(MPF_BLAS_COL_MAJOR, 'F', n, blk, R, n);

    #if DEBUG == 1
      printf("norm_frobenious_residual: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm/B_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *R)*n*blk);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  H = NULL;
  V = NULL;
  Br = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
  tempf_matrix = NULL;
}

void mpf_zge_gbl_gmres
(
  /* solver parameters */
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

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* solver context */
  double h_tempf_abs = 0.0;
  double B_norm = 0.0;
  double R_norm = 0.0;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  //MPF_ComplexDouble h_temp = ZERO_C;

  /* solver solver->ata */
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_H = (solver->iterations+1);
  MPF_Int n_H = solver->iterations;
  const MPF_Int ld_H = m_H;
  MPF_Int size_V = n* blk * m_H;
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_Br = m_H;
  MPF_Int size_Hblk = blk*blk;
  MPF_ComplexDouble trace = ZERO_C;

  /* map mathematical objects to cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[size_V];
  MPF_ComplexDouble *Br = &H[size_H];
  MPF_ComplexDouble *Hblk = &Br[size_Br];
  MPF_ComplexDouble *tempf_matrix = &Hblk[size_Hblk];
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *const Vfirst = V;
  MPF_ComplexDouble *const Vlast = &V[(n*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;

  /* computes residual vectors block using initial approximation of solution
     block vectors */
  memcpy(V, B, (sizeof *V)*n*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, Vfirst, n);
  B_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', n, blk, B, n);
  R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', n, blk, Vfirst, n);
  if (R_norm/B_norm <= solver->tolerance) /* checks terminating condition */
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    /* first iteration */
    tempf_complex = mpf_scalar_z_normalize(ONE_C, R_norm);
    mpf_zscal(n*blk, &tempf_complex, Vfirst, 1);
    Br[0] = mpf_scalar_z_init(R_norm, 0.0);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H-1, 1, &Br[1], m_H);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; j++)
    {
      W = &V[(n * blk)*(j+1)];
      Vprev = &V[(n * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        Vprev = &V[(n*blk)*i];
        //mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
        //         &ONE_C, W, n, Vprev, n, &ZERO_C,
        //         Hblk, blk);
        //trace = ZERO_C;
        //for (t = 0; t < blk; t++)
        //{
        //    trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
        //}
        mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, n*blk,
          &ONE_C, W, n*blk, Vprev, n*blk, &ZERO_C, &trace, 1);
        H[ld_H*j+i] = trace;
        tempf_complex = mpf_scalar_z_invert_sign(trace);
        mpf_zaxpy(n*blk, &tempf_complex, Vprev, 1, W, 1);
      }

      h_tempf_abs = mpf_zlange(LAPACK_COL_MAJOR, 'F', n, blk, W, n);
      if (h_tempf_abs <= 1e-12)
      {
        inner_iterations = j+1;
        m_H = (inner_iterations+1);
        n_H = inner_iterations;
        break;
      }
      tempf_complex = mpf_scalar_z_normalize(ONE_C, h_tempf_abs);
      mpf_zscal(n*blk, &tempf_complex, W, 1);
      H[ld_H*j+j+1] = mpf_scalar_z_init(h_tempf_abs, 0.0);
    }

    /* solve system of equations using qr decomposition */
    //@BUG: this needs to take as input the ld_H because m_H != ld_H
    mpf_qr_zge_givens(H, Br, m_H, n_H, blk, tempf_matrix);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (MPF_Int i = 0; i < n_H; ++i)
    {
      W = &V[n*blk*i];
      mpf_zaxpy(n*blk, &Br[i], W, 1, X, 1);
    }

    memcpy(R, B, (sizeof *B)*n*blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', n, blk, R, n);

    #if DEBUG == 1
      printf("norm_frobenious_residual/norm_frobenious_B: %1.4E\n",
        R_norm/B_norm);
    #endif

    if ((R_norm/B_norm <= solver->tolerance) || (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *R)*n*blk); /* copies residual to V(:, block(1)) */
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  Vprev = NULL;
  W = NULL;
  R = NULL;
}

