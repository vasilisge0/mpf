#include "mpf.h"

void mpf_dge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  double norm_b = 0.0;
  double r_norm = 0.0;

  /* solver->*/
  MPF_Int n = solver->ld;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;
  MPF_Int size_V = n*m_H;
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_tempf_matrix = 2*n_H;

  double *b = (double*) b_dense->data;
  double *x = (double*) x_dense->data;

  /* cpu accesible memory */
  double *V = (double*)solver->inner_mem;
  double *H = &V[size_V];
  double *tempf_matrix = &H[size_H];
  double *br = &tempf_matrix[size_tempf_matrix];

  /* handles on cpu accesible memory */
  double *vfirst = V;
  double *vlast = &V[n*solver->iterations];
  double *w = vfirst + n;
  double *vprev = vfirst;
  double *r = vlast;

  /* first iteration */
  for (MPF_Int i = 0; i < n_H; ++i)
  {
    for (MPF_Int j = 0; j < m_H; ++j)
    {
      H[m_H*i+j] = 0.0;
    }
  }

  /* initializes br vector */
  for (MPF_Int i = 0; i < m_H; i++)
  {
    br[i] = 0.0;
  }

  memcpy(r, b, (sizeof r)*n);
  mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x,
    1.0, r);
  norm_b = mpf_dnrm2(n, b, 1);

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    memcpy(V, r, (sizeof *V)*n);
    r_norm = mpf_dnrm2(n, r, 1);
    br[0] = r_norm;
    mpf_dscal(n, 1/r_norm, vfirst, 1);

    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      w = &vfirst[n*(j+1)];
      vprev = &vfirst[n*j];
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        vprev, 0.0, w);

      for (MPF_Int i = 0; i < j+1; ++i)
      {
        vprev = &vfirst[n*i];
        H[m_H*j+i] = mpf_ddot(n, w, 1, vprev, 1);
        mpf_daxpy(n, -H[m_H*j+i], vprev, 1, w, 1);
      }
      H[m_H*j+j+1] = mpf_dnrm2(n, w, 1);
      if (H[m_H*j+j+1] <= 1e-12)
      {
        inner_iterations = j+1;
        m_H = j+1;
        n_H = j;
        break;
      }
      mpf_dscal(n, 1/H[m_H*j+j+1], w, 1);
    }

    #if DEBUG == 1
      printf("m_H: %d, n_H: %d\n", m_H, n_H);
      mpf_matrix_d_announce(H, m_H, n_H, m_H, "H (gmres/before) 1st");
    #endif

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    mpf_qr_givens_dge(n_H, 1, H, m_H, br, m_H, tempf_matrix);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, br, ld_H);
    mpf_dgemv(CblasColMajor, CblasNoTrans, n, n_H, 1.0, V, n, br, 1, 1.0, x, 1);

    /* computes residual and residual norm */
    memcpy(r, b, (sizeof *r)*n);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x, 1.0, r);
    r_norm = mpf_dnrm2(n, r, 1);
    #if DEBUG == 1
      printf("relative residual: %1.4E\n", r_norm/norm_b);
      printf("       iterations: %d/%d\n", n_H, solver->iterations);
    #endif

    if (r_norm/norm_b <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }
  V = NULL;
  H = NULL;
  tempf_matrix = NULL;
  br = NULL;
  vfirst = NULL;
  vlast = NULL;
  w = NULL;
  vprev = NULL;
  r = NULL;
}

void mpf_sge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* context */
  float b_norm = 0.0;
  float r_norm = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;
  MPF_Int n = solver->ld;

  float *b = (float*)b_dense->data;
  float *x = (float*)x_dense->data;

  /* maps mathematical objects to cpu memory */
  float *V = (float*)solver->inner_mem;
  float *H = &V[n*m_H];
  float *tempf_matrix = &H[m_H*n_H];
  float *br = &tempf_matrix[2*n_H];

  /* handles on cpu memory */
  float *vfirst = V;
  float *vlast = &V[n*solver->iterations];
  float *w = &vfirst[n];
  float *vprev = vfirst;
  float *r = vlast;

  /* first iteration */
  for (MPF_Int i = 0; i < n_H; ++i)
  {
    for (MPF_Int j = 0; j < m_H; ++j)
    {
      H[m_H*i+j] = 0.0;
    }
  }

  for (MPF_Int i = 0; i < m_H; ++i)
  {
    br[i] = 0.0;
  }

  memcpy(r, b, (sizeof r)*n);
  mpf_sparse_s_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x, 1.0, r);
  b_norm = mpf_snrm2(n, b, 1);

  /* outer iterations (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    memcpy(vfirst, r, (sizeof *vfirst)*n);
    r_norm = mpf_snrm2(n, r, 1);
    br[0] = r_norm;
    mpf_sscal(n, 1/r_norm, vfirst, 1);

    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      w = &V[n*(j+1)];
      vprev = &V[n*j];
      mpf_sparse_s_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, vprev,
        0.0, w);

      for (MPF_Int i = 0; i < j + 1; ++i)
      {
        vprev = vfirst + n*i;
        H[m_H*j+i] = mpf_sdot(n, w, 1, vprev, 1);
        mpf_saxpy(n, -H[m_H*j+i], vprev, 1, w, 1);
      }
      H[m_H*j+j+1] = mpf_snrm2(n, w, 1);

      if (H[m_H*j+j+1] <= 1e-12)
      {
        inner_iterations = k;
        m_H = inner_iterations+1;
        n_H = inner_iterations;
        break;
      }
      mpf_sscal(n, 1/H[m_H*j+j+1], w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    mpf_qr_givens_sge(n_H, 1, H, m_H, br, m_H, tempf_matrix);
    mpf_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans , CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, br, m_H);
    //mpf_sgemv(CblasColMajor, CblasNoTrans, n, n_H, 1.0, V, n, br, 1, 1.0, x, 1);

    /* computes residual and residual norm */
    memcpy(r, b, (sizeof *r)*n);
    mpf_sparse_s_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x, 1.0, r);
    r_norm = mpf_snrm2(n, r, 1);

    #if DEBUG == 1
      printf("relative residual: %1.4E\n", r_norm/b_norm);
      printf(" outer_iterations: %d/%d\n", k, outer_iterations);
    #endif

    if (r_norm/b_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  V = NULL;
  H = NULL;
  tempf_matrix = NULL;
  br = NULL;
  vfirst = NULL;
  vlast = NULL;
  w = NULL;
  vprev = NULL;
  r = NULL;
}

void mpf_zge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  /* context */
  MPF_Int n = solver->ld;
  double b_norm = 0.0;
  double r_norm = 0.0;
  //double b_norm_abs = 0.0;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble normalization_coeff;
  double h_last = 0.0;

  MPF_ComplexDouble* b = (MPF_ComplexDouble*)b_dense->data;
  MPF_ComplexDouble* x = (MPF_ComplexDouble*)x_dense->data;

  /* meta */
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  //MPF_Int ld_H = m_H;

  /* cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;    /*holds Krylov orthogonal basis*/
  MPF_ComplexDouble *H = &V[n*m_H]; /*upper Hessenberg Krylov recurrence matrix*/
  MPF_ComplexDouble *tempf_matrix = &H[m_H*n_H];
  MPF_ComplexDouble *br = &tempf_matrix[2*n_H];

  /* handles on assigned memory */
  MPF_ComplexDouble *vfirst = V;
  MPF_ComplexDouble *vlast = &V[n*solver->iterations]; /* last vector of krylov basis */
  MPF_ComplexDouble *w = &vfirst[n];
  MPF_ComplexDouble *vprev = vfirst;
  MPF_ComplexDouble *r = vlast;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* outer iterations (restarts) */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(r, b, (sizeof *r)*n);
  mpf_sparse_z_mv(MPF_SPARSE_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, x, ONE_C, r);
  memcpy(vfirst, r, (sizeof *vfirst)*n);
  b_norm = mpf_dznrm2(n, b, 1);
  r_norm = mpf_dznrm2(n, r, 1);
  //#if DEBUG == 1
    printf("[init] r_norm/norm_b: %1.4E\n", r_norm/b_norm);
    printf("r_norm: %1.4E\n", r_norm);
    printf("norm_b: %1.4E\n", b_norm);
    printf("     n: %d\n", n);
  //#endif

  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    /* first iteration */
    mpf_zeros_z_set(MPF_BLAS_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = mpf_scalar_z_init(r_norm, 0.0);
    normalization_coeff = mpf_scalar_z_normalize(ONE_C, r_norm);
    mpf_zscal(n, &normalization_coeff, vfirst, 1);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      w = &V[n*(j+1)];
      vprev = &V[n*j];
      mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, vprev, ZERO_C, w);

      for (MPF_Int i = 0; i < j + 1; ++i)
      {
        vprev = &V[n*i];
        mpf_zgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_CONJ_TRANS, MPF_BLAS_NO_TRANS,
          1, 1, n, &ONE_C, w, n, vprev, n, &ZERO_C, &H[m_H*j+i], m_H);
        tempf_complex = mpf_scalar_z_invert_sign(H[m_H*j+i]);
        mpf_zaxpy(n, &tempf_complex, vprev, 1, w, 1);
      }

      h_last = mpf_dznrm2(n, w, 1);
      H[m_H*j+j+1] = mpf_scalar_z_init(h_last, 0.0);
      if (h_last <= 1e-12)
      {
          n_H = j;
          m_H = n_H+1;
          break;
      }
      normalization_coeff = mpf_scalar_z_divide(ONE_C, H[m_H*j+j+1]);
      mpf_zscal(n, &normalization_coeff, w, 1);
    }

    /* computes solution to linear system and constructs solution to the linear
       system of equations and checks termination criteria */
    mpf_qr_givens_zge_2(H, m_H, br, m_H, m_H, n_H, 1, tempf_matrix);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, m_H, br, m_H);
    mpf_zgemv(CblasColMajor, CblasNoTrans, n, n_H, &ONE_C, V, n, br, 1,
      &ONE_C, x, 1);
    memcpy(r, b, (sizeof *r)*n);
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, x, ONE_C, r);
    r_norm = mpf_dznrm2(n, r, 1);

    //#if DEBUG == 1
      printf("(out) r_norm/norm_b: %1.4E\n", r_norm/b_norm);
    //#endif

    k += 1;
    if (r_norm/b_norm <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, r, (sizeof *V)*n);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  V = NULL;
  H = NULL;
  tempf_matrix = NULL;
  br = NULL;
  vfirst = NULL;
  vlast = NULL;
  w = NULL;
  vprev = NULL;
  r = NULL;
}

void mpf_zsy_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* constants */
  MPF_Int n = solver->ld;
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  /* context */
  MPF_ComplexDouble b_norm;
  MPF_ComplexDouble r_norm;
  MPF_ComplexDouble temp_complex = ZERO_C;
  MPF_ComplexDouble normalization_coeff;
  double h_abs;
  double r_norm_abs = 0.0;
  double b_norm_abs = 0.0;

  MPF_ComplexDouble* b = (MPF_ComplexDouble*)b_dense->data;
  MPF_ComplexDouble* x = (MPF_ComplexDouble*)x_dense->data;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  //MPF_Int ld_H = m_H;

  /* cpu memory */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;    /* holds Krylov orthogonal basis */
  MPF_ComplexDouble *H = &V[n*m_H]; /* Hessenberg Krylov recurrence matrix */
  MPF_ComplexDouble *temp_matrix = &H[m_H*n_H];
  MPF_ComplexDouble *br = &temp_matrix[2*n_H];

  /* handles on assigned memory */
  MPF_ComplexDouble *vlast = &V[n*solver->iterations];
  MPF_ComplexDouble *w = &V[n];  /* added in Krylov basis each iteration */
  MPF_ComplexDouble *vprev = V;  /* previous vector in krylov basis */
  MPF_ComplexDouble *r = vlast;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* outer iterations (restarts) */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(r, b, (sizeof *r)*n);
  mpf_sparse_z_mv(MPF_SPARSE_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, x, ONE_C, r);
  mpf_zgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, 1, 1, n,
  &ONE_C, b, n, b, n, &ZERO_C, &temp_complex, 1);
  mpf_vectorized_z_sqrt(1, &temp_complex, &b_norm);
  mpf_zgemm(layout, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, 1, 1, n, &ONE_C, r,
    n, r, n, &ZERO_C, &temp_complex, 1);
  mpf_vectorized_z_sqrt(1, &temp_complex, &r_norm);
  memcpy(V, r, (sizeof *V)*n);
  mpf_vectorized_z_abs(1, &r_norm, &r_norm_abs);
  mpf_vectorized_z_abs(1, &b_norm, &b_norm_abs);

  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    /* first iteration */
    mpf_zeros_z_set(MPF_BLAS_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = r_norm;
    normalization_coeff = mpf_scalar_z_divide(ONE_C, r_norm);
    mpf_zscal(n, &normalization_coeff, V, 1);

    /* inner iterations */
    for (MPF_Int j = 0; j < inner_iterations; ++j)
    {
      w = &V[n*(j+1)];
      vprev = &V[n*j];
      mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, vprev, ZERO_C, w);   /* w <-- A*V(:, j) */

      for (MPF_Int i = 0; i < j + 1; ++i)
      {
        vprev = &V[n*i];
        mpf_zgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, 1, 1, n,
          &ONE_C, w, n, vprev, n, &ZERO_C, &H[m_H*j+i], m_H);
        temp_complex = mpf_scalar_z_invert_sign(H[m_H*j+i]);
        mpf_zaxpy(n, &temp_complex, vprev, 1, w, 1);
      }
      mpf_zgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, 1, 1, n,
        &ONE_C, w, n, w, n, &ZERO_C, &H[m_H*j+j+1], m_H);

      mpf_vectorized_z_sqrt(1, &H[m_H*j+j+1], &H[m_H*j+j+1]);
      mpf_vectorized_z_abs(1, &H[m_H*j+j+1], &h_abs);
      if (h_abs <= 1e-12)
      {
        n_H = j;
        m_H = n_H+1;
        break;
      }
      normalization_coeff = mpf_scalar_z_divide(ONE_C, H[m_H*j+j+1]);
      mpf_zscal(n, &normalization_coeff, w, 1);
    }

    /* computes solution to linear system and constructs solution to the
       linear system of equations and checks termination criteria */
    mpf_qr_zge_givens(H, br, m_H, n_H, 1, temp_matrix);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, m_H, br, m_H);
    mpf_zgemv(CblasColMajor, CblasNoTrans, n, n_H, &ONE_C, V, n, br, 1,
      &ONE_C, x, 1);

    memcpy(r, b, (sizeof *r)*n);
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, x, ONE_C, r);
    mpf_zgemm(MPF_BLAS_COL_MAJOR, MPF_BLAS_TRANS, MPF_BLAS_NO_TRANS, 1, 1, n,
      &ONE_C, r, n, r, n, &ZERO_C, &temp_complex, 1);

    mpf_vectorized_z_sqrt(1, &temp_complex, &r_norm);
    temp_complex = mpf_scalar_z_divide(r_norm, b_norm);
    mpf_vectorized_z_abs(1, &temp_complex, &r_norm_abs);

    #if DEBUG == 1
        printf("relative residual: %1.4E\n", r_norm_abs/b_norm_abs);
    #endif

    k += 1;
    if (r_norm_abs/b_norm_abs <= solver->tolerance)
    {
        outer_iterations = k;
        break;
    }
    else
    {
      memcpy(V, r, (sizeof *V)*n);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  V = NULL;
  H = NULL;
  temp_matrix = NULL;
  br = NULL;
  vlast = NULL;
  w = NULL;
  vprev = NULL;
  r = NULL;
}

void mpf_cpy_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_Complex ONE_C = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex MINUS_ONE_C = mpf_scalar_c_init(-1.0, 0.0);
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);

  /* context */
  MPF_Int n = solver->ld;
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Complex b_norm;
  MPF_Complex r_norm;
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Complex normalization_coeff;

  MPF_Complex* B = (MPF_Complex*)B_dense->data;
  MPF_Complex* X = (MPF_Complex*)X_dense->data;

  /* solver solver->*/
  float h_abs;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int m_H = solver->iterations+1;
  MPF_Int n_H = solver->iterations;
  MPF_Int size_V = n*m_H;
  MPF_Int size_H = m_H*n_H;
  MPF_Int size_tempf_matrix = 2*n_H;

  /* cpu memory */
  MPF_Complex *V = (MPF_Complex*)solver->inner_mem;     /* holds Krylov orthogonal basis */
  MPF_Complex *H = &V[size_V]; /* upper Hessenberg Krylov recurrence matrix */
  MPF_Complex *tempf_matrix = &H[size_H];
  MPF_Complex *br = &tempf_matrix[size_tempf_matrix];

  /* handles on assigned memory */
  MPF_Complex *vfirst = V;
  MPF_Complex *vlast = &V[n*solver->iterations];  /* last vector of krylov basis */
  MPF_Complex *w = &V[n];                 /* new vector in Krylov basis */
  MPF_Complex *vprev = vfirst;            /* previous vector in krylov basis */
  MPF_Complex *r = vlast;

  /* first iterations */
  mpf_zeros_c_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(vfirst, B, (sizeof *vfirst)*n);
  mpf_sparse_c_mv(MPF_SPARSE_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, X, ONE_C, vfirst);
  mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C, B, n,
    B, n, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_c_sqrt(1, &tempf_complex, &b_norm);
  mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C, r, n,
    r, n, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_c_sqrt(1, &tempf_complex, &r_norm);

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; k++)
  {
    /* first iteration */
    mpf_zeros_c_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = r_norm;
    normalization_coeff = mpf_scalar_c_divide(ONE_C, r_norm);
    mpf_cscal(n, &normalization_coeff, vfirst, 1);
    /* inner iterations */
    for (j = 0; j < inner_iterations; j++)
    {
      w = &V[n*(j + 1)];
      vprev = &V[n*j];
      mpf_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, vprev, ZERO_C, w);
      for (i = 0; i < j + 1; i++)
      {
        vprev = &V[n*i];
        mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
          w, n, vprev, n, &ZERO_C, &H[m_H*j+i], m_H);
        tempf_complex = mpf_scalar_c_invert_sign(H[m_H*j+i]);
        mpf_caxpy(n, &tempf_complex, vprev, 1, w, 1);
      }
      mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C,
        w, n, w, n, &ZERO_C, &H[m_H*j+j+1], m_H);
      mpf_vectorized_c_sqrt(1, &H[m_H*j+j+1], &H[m_H*j+j+1]);
      mpf_vectorized_c_abs(1, &H[m_H*j+j+1], &h_abs);
      if (h_abs <= 1e-12)
      {
          break;
      }
      normalization_coeff = mpf_scalar_c_divide(ONE_C, H[m_H*j+j+1]);
      mpf_cscal(n, &normalization_coeff, w, 1);
    }

    /* computes solution to linear system and constructs solution to the
       linear system of equations and checks termination criteria */
    //mpf_qr_givens_cge_factorize(H, br, m_H, n_H, 1, tempf_matrix);
    mpf_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, m_H, br, m_H);
    mpf_cgemv(CblasColMajor, CblasNoTrans, n, n_H, &ONE_C, V, n, br, 1,
      &ONE_C, X, 1);
    memcpy(r, B, (sizeof *r)*n);
    mpf_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, X, ONE_C, r);
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n, &ONE_C, r,
      n, r, n, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_c_sqrt(1, &tempf_complex, &r_norm);
    tempf_complex = mpf_scalar_c_divide(r_norm, b_norm);
    mpf_vectorized_c_abs(1, &tempf_complex, &h_abs);

    #if DEBUG == 1
      printf("relative residual: %1.4E\n", h_abs);
    #endif

    if (h_abs <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, r, (sizeof *V)*n);
      inner_iterations = solver->iterations;
      m_H = solver->iterations+1;
      n_H = solver->iterations;
    }
  }

  V = NULL;
  H = NULL;
  tempf_matrix = NULL;
  br = NULL;
  vfirst = NULL;
  vlast = NULL;
  w = NULL;
  vprev = NULL;
  r = NULL;
}

void mpf_gmres_init
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
  context->args.n_inner_solve = 3;

  if ((solver->precond_type == MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DGE_GMRES;
      solver->inner_function = &mpf_dge_gmres;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_GMRES;
      solver->inner_function = &mpf_zsy_gmres;
      solver->device = MPF_DEVICE_CPU;
  }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZGE_GMRES;
      solver->inner_function = &mpf_zge_gmres;
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
  solver->inner_get_mem_size_function = &mpf_gmres_get_mem_size;
}

void mpf_gmres_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int iterations = solver->iterations;
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
      n*(iterations+1)            /* size_V */
      +(iterations+1)*iterations  /* size_H */
      +2*iterations               /* size_temp_matrix */
      +(iterations+1);            /* size_br */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
      +(n*(iterations+1)         /* size_V */
      +(iterations+1)*iterations /* size_H */
      +2*iterations              /* size_temp_matrix */
      +(iterations+1));          /* size_br */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
      solver->inner_bytes = sizeof(MPF_ComplexDouble)*
          +(n*(iterations+1)         /* size_V */
          +(iterations+1)*iterations /* size_H */
          +2*iterations              /* size_temp_matrix */
          +(iterations+1));          /* size_br */
  }
}
