#include "mpf.h"

void mpf_dsy_lanczos
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

  /* solver->*/
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int m_B = solver->ld;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;

  double *b = (double*)b_dense->data;
  double *x = (double*)x_dense->data;

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
  mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x,
    1.0, r);
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
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, 1.0, H, n_H, br, m_H);
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

void mpf_zsy_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble COMPLEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  MPF_ComplexDouble *b = (MPF_ComplexDouble*)b_dense->data;
  MPF_ComplexDouble *x = (MPF_ComplexDouble*)x_dense->data;

  /* solver context */
  double b_norm = 0.0;
  double r_norm = 0.0;
  double h_abs = 0.0;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble normalization_coeff;

  /* solver->*/
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int blk = solver->batch;
  MPF_Int n = solver->ld;
  MPF_Int m_B = n;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;

  /* assigned memory to mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[(solver->iterations+1)*m_B];
  MPF_ComplexDouble *br = &H[m_H*n_H];
  MPF_ComplexDouble *r = &br[m_H];

  /* handles on allocated memory */
  MPF_ComplexDouble *w = NULL;
  MPF_ComplexDouble *vprev = NULL;
  MPF_ComplexDouble *vcurr = NULL;

  /* initializes krylov method */
  memset(H, 0, (sizeof *H) * m_H * n_H);
  memcpy(r, b, (sizeof *r) * m_B);
  mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle,
    A->descr, x, ONE_C, r);
  b_norm = mpf_dznrm2(m_B, b, 1);
  r_norm = mpf_dznrm2(m_B, r, 1);
  memcpy(V, r, (sizeof *V) * m_B);
  memset(br, 0, (sizeof *br) * m_H);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
    r, m_B, r, m_B, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
  mpf_vectorized_z_abs(1, &tempf_complex, &r_norm);
  br[0] = tempf_complex;
  tempf_complex = mpf_scalar_z_divide(ONE_C, tempf_complex);
  mpf_zscal(m_B, &tempf_complex, V, 1);
  #if DEBUG == 1
    printf("relative residual: %1.4E\n", r_norm / b_norm);
  #endif

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    w = &V[m_B];
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
      A->descr, V, ZERO_C, w);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      w, m_B, V, m_B, &ZERO_C, &H[0], m_B);
    tempf_complex = mpf_scalar_z_invert_sign(H[0]);
    mpf_zaxpy(m_B, &tempf_complex, V, 1, w, 1);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      w, m_B, w, m_B, &ZERO_C, &H[1], m_B);
    mpf_vectorized_z_sqrt(1, &H[1], &H[1]);
    mpf_vectorized_z_abs(1, &H[1], &h_abs);
    if ((r_norm/b_norm) < 1e-12)
    {
      inner_iterations = 1;
    }
    normalization_coeff = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m_B, &normalization_coeff, w, 1);

    /* inner iterations */
    for (MPF_Int j = 1; j < inner_iterations; ++j)
    {
      w = &V[m_B*(j + 1)];
      vcurr = &V[m_B*j];
      vprev = &V[m_B*(j - 1)];
      mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, vcurr, ZERO_C, w);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B,
         &ONE_C, w, m_B, vcurr, m_B, &ZERO_C, &H[m_H*j + j], m_B);
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*j+j]);
      mpf_zaxpy(m_B, &tempf_complex, vcurr, 1, w, 1);
      H[m_H*j+j-1] = H[m_H*(j-1)+j];
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*j+j-1]);
      mpf_zaxpy(m_B, &tempf_complex, vprev, 1, w, 1);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
        w, m_B, w, m_B, &ZERO_C, &tempf_complex, m_B);

      mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
      mpf_vectorized_z_abs(1, &tempf_complex, &h_abs);
      if ((h_abs <= 1e-12) || (j == inner_iterations - 1))
      {
        m_H = j+1;
        n_H = m_H;
        break;
      }
      H[m_H*j+j+1] = tempf_complex;
      normalization_coeff = mpf_scalar_z_divide(ONE_C, H[m_H*j+j+1]);
      mpf_zscal(m_B, &normalization_coeff, w, 1);
    }

    /* solves upper triangular linear system of equations and checks */
    /* termination condition */
    mpf_qr_zsy_givens_2(H, br, ld_H, n_H, blk);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, &ONE_C, H, m_H, br, m_H);
    mpf_zgemv(CblasColMajor, CblasNoTrans, m_B, m_H, &ONE_C, V, m_B, br, 1,
      &ONE_C, x, 1);
    memcpy(r, b, (sizeof *r)*m_B);
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, COMPLEX_MINUS_ONE, A->handle,
      A->descr, x, ONE_C, r);
    r_norm = mpf_dznrm2(m_B, r, 1);

    #if DEBUG == 1
      printf("relative residual: %1.4E\n", r_norm / b_norm); 
    #endif

    if ((r_norm/b_norm <= solver->tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      /* restart */
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
        r, m_B, r, m_B, &ZERO_C, &tempf_complex, 1);
      mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
      memcpy(V, r, (sizeof *V)*m_B);
      normalization_coeff = mpf_scalar_z_divide(ONE_C, tempf_complex);
      mpf_zscal(m_B, &normalization_coeff, V, 1);
      memset(br, 0, (sizeof *br) * m_H);
      br[0] = normalization_coeff;
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

void mpf_zhe_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble COMPF_LEX_MINUS_ONE = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  /* solver context */
  double b_norm = 0.0;
  double r_norm = 0.0;
  double h_abs = 0.0;
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble normalization_coeff;

  /* solver->*/
  MPF_Int outer_iterations = solver->restarts + 1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int blk = 1;
  MPF_Int m = solver->ld;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;
  MPF_Int ld_H = m_H;

  MPF_ComplexDouble *b = (MPF_ComplexDouble*)b_dense->data;
  MPF_ComplexDouble *x = (MPF_ComplexDouble*)x_dense->data;

  /* assigned memory to mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[(solver->iterations+1)*m];
  MPF_ComplexDouble *br = &H[m_H*n_H];
  MPF_ComplexDouble *r = &br[m_H];

  /* handles on allocated memory */
  MPF_ComplexDouble *w = NULL;
  MPF_ComplexDouble *vprev = NULL;
  MPF_ComplexDouble *vcurr = NULL;

  /* initializes krylov method */
  memset(H, 0, (sizeof *H)*m_H*n_H);
  memcpy(r, b, (sizeof *r)*m);
  mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, COMPF_LEX_MINUS_ONE, A->handle,
    A->descr, x, ONE_C, r);
  b_norm = mpf_dznrm2(m, b, 1);
  r_norm = mpf_dznrm2(m, r, 1);
  memcpy(V, r, (sizeof *V)*m);
  memset(br, 0, (sizeof *br)*m_H);
  br[0] = mpf_scalar_z_init(r_norm, 0.0);
  tempf_complex = mpf_scalar_z_divide(ONE_C, br[0]);
  mpf_zscal(m, &tempf_complex, V, 1);

  #if DEBUG == 1
    printf("relative residual: %1.4E\n", r_norm / b_norm);
  #endif

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    w = &V[m];
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
       A->descr, V, ZERO_C, w);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m, &ONE_C,
      w, m, V, m, &ZERO_C, &H[0], m_H);
    tempf_complex = mpf_scalar_z_invert_sign(H[0]);
    mpf_zaxpy(m, &tempf_complex, V, 1, w, 1);
    H[1] = mpf_scalar_z_init(mpf_dznrm2(m, w, 1), 0.0);
    if ((r_norm/b_norm) < 1e-12)
    {
      inner_iterations = 1;
    }
    normalization_coeff = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m, &normalization_coeff, w, 1);

    /* inner iterations */
    for (MPF_Int j = 1; j < inner_iterations; ++j)
    {
      w = &V[m*(j + 1)];
      vcurr = &V[m*j];
      vprev = &V[m*(j - 1)];
      mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
        A->descr, vcurr, ZERO_C, w);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, m,
        &ONE_C, w, m, vcurr, m, &ZERO_C, &H[m_H*j+j], m);
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*j+j]);
      mpf_zaxpy(m, &tempf_complex, vcurr, 1, w, 1);
      H[m_H*j+j-1] = H[m_H*(j-1)+j];
      tempf_complex = mpf_scalar_z_invert_sign(H[m_H*j+j-1]);
      mpf_zaxpy(m, &tempf_complex, vprev, 1, w, 1);
      h_abs = mpf_dznrm2(m, w, 1);
      if ((h_abs <= 1e-12) || (j == inner_iterations - 1))
      {
        m_H = j+1;
        n_H = m_H;
        break;
      }
      H[m_H*j+j+1] = mpf_scalar_z_init(h_abs, 0.0);
      normalization_coeff = mpf_scalar_z_normalize(ONE_C, h_abs);
      mpf_zscal(m, &normalization_coeff, w, 1);
    }

    /* solves upper triangular linear system of equations and checks */
    /* termination condition */
    mpf_qr_zsy_givens_2(H, br, ld_H, n_H, blk);  // note: uses this in order to avoid error when ld_H != m_H
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, &ONE_C, H, m_H, br, m_H);
    mpf_zgemv(CblasColMajor, CblasNoTrans, m, m_H, &ONE_C, V, m, br, 1,
             &ONE_C, x, 1);
    memcpy(r, b, (sizeof *r)*m);
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, COMPF_LEX_MINUS_ONE, A->handle,
      A->descr, x, ONE_C, r);
    r_norm = mpf_dznrm2(m, r, 1);

    #if DEBUG == 1
      printf("r_norm/b_norm: %1.4E\n", r_norm / b_norm); 
    #endif

    if ((r_norm/b_norm <= solver->tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      /* restart */
      memcpy(V, r, (sizeof *V)*m);
      normalization_coeff = mpf_scalar_z_normalize(ONE_C, r_norm);
      mpf_zscal(m, &normalization_coeff, V, 1);
      MPF_Int i = 0;
      for (i = 0; i < m_H; ++i)
      {
        br[i] = ZERO_C;
      }
      br[0] = mpf_scalar_z_init(r_norm, 0.0);
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

void mpf_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
)
{
  MPF_Solver* solver = &context->solver;
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;
  context->args.n_inner_solve = 4;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_LANCZOS;
    solver->inner_function = &mpf_dsy_lanczos;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_LANCZOS;
    solver->inner_function = &mpf_zsy_lanczos;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_LANCZOS;
    solver->inner_function = &mpf_zhe_lanczos;
  }

  solver->inner_alloc_function = &mpf_krylov_alloc;
  solver->inner_free_function = &mpf_krylov_free;
  mpf_lanczos_get_mem_size(solver);
}


void mpf_lanczos_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int iterations = solver->iterations;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)
      * (n*blk                 /* size_B */
        +n*blk                 /* size_X */
        +n*(iterations+1)      /* size_V */
        +iterations*iterations /* size_H */
        +iterations            /* size_br */
        +n);                   /* size_residual */
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
