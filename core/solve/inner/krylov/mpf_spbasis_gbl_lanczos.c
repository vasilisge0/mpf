#include "mpf.h"

void mpf_spbasis_gbl_lanczos_init
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
    solver->inner_function = &mpf_dsy_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC)
  )
  {
    solver->inner_type = MPF_SOLVER_ZSY_GBL_LANCZOS;
    solver->inner_function = &mpf_zsy_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN)
  )
  {
    solver->inner_type = MPF_SOLVER_ZHE_GBL_LANCZOS;
    solver->inner_function = &mpf_zhe_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  mpf_gbl_lanczos_get_mem_size(solver);
}

void mpf_dsy_spbasis_gbl_lanczos
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

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* solver->*/
  MPF_Int n = solver->ld;
  MPF_Int m_B = n;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int ld_H = inner_iterations;
  MPF_Int max_n_H = inner_iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = max_n_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* assign cpu memory to mathematical objects */
  double *V = (double*)solver->inner_mem;
  double *H = &V[m_B*blk*(m_H+1)];
  double *Hblk = &H[m_H*n_H];
  double *Br = &Hblk[blk*blk];

  /* handles to allocated cpu memory */
  double *W = NULL;
  double *Vprev = NULL;
  double *Vlast = &V[(m_B * blk) * inner_iterations];
  double *R = Vlast;

  /* required for sparsification */
  MPF_Int nz_new = 0;
  MPF_Int end_sparse = 0;
  MPF_Int blk_max = solver->max_blk_fA;
  MPF_Int current_rhs = solver->current_rhs;

  /* compute residual vectors block using initial approximation of solution */
  /* block vectors */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V)*m_B*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m_B, 1.0, V, m_B);
  B_norm = mpf_dlange (CblasColMajor, 'F', m_B, blk, B, m_B);
  R_norm = mpf_dlange(CblasColMajor, 'F', m_B, blk, V, m_B);

  #if DEBUG == 1
    printf("residual norm: %1.4E\n", R_norm/B_norm);
  #endif

  if (R_norm/B_norm <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_dscal(m_B*blk, 1/R_norm, V, 1);
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0] = B_norm;

    W = &V[(m_B*blk)];
    Vprev = V;
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      sparse_layout, Vprev, blk, m_B, 0.0, W, m_B);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
      Vprev, m_B, W, m_B, 0.0, Hblk, blk);
    trace = 0.0;

    for (MPF_Int t = 0; t < blk; ++t)
    {
      trace = trace + Hblk[blk*t + t];
    }

    H[0] = trace;
    mpf_daxpy(m_B*blk, -trace, Vprev, 1, W, 1);

    H[1] = LAPACKE_dlange (LAPACK_COL_MAJOR, 'F', m_B, blk, W, m_B);
    mpf_dscal(m_B*blk, 1/H[1], W, 1);

    MPF_Int j;
    for (j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];

      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, 0.0, W, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
        Vprev, m_B, W, m_B, 0.0, Hblk, blk);

      trace = 0.0;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = trace + Hblk[blk*t + t];
      }

      H[ld_H*j + j] = trace;
      mpf_daxpy(m_B*blk, -trace, Vprev , 1, W, 1);
      Vprev = &V[(m_B*blk)*(j-1)];
      mpf_daxpy(m_B*blk, -H[m_H*(j-1) + j], Vprev , 1, W, 1);
      h_temp = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m_B, blk, W, m_B);

      H[ld_H*j + j-1] = H[ld_H*(j-1) + j];
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }

      H[ld_H*j + j+1] = h_temp;
      mpf_dscal(m_B*blk, 1/H[m_H*j + j+1], W, 1);
      vecblk_d_block_sparsify(
        m_B,              /* length of input vector */
        blk,
        Vprev,            /* input vector */
        &V[end_sparse],   /* output vector (compresed) */
        blk_max,
        current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    vecblk_d_block_sparsify(
      m_B,              /* length of input vector */
      blk,
      Vprev,            /* input vector */
      &V[j*(m_B*blk)],  /* output vector (compresed) */
      blk_max,
      current_rhs,
      &solver->color_to_node_map,
      &nz_new);

    end_sparse += nz_new;    // better max error, this is the correct one

    /* solves upper triangular linear system and evaluates
       termination criteria */
    mpf_qr_dsy_givens(m_H, n_H, 1, H, ld_H, Br);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, Br, ld_H);

    global_krylov_dge_sparse_basis_block_combine
    (
      current_rhs,
      blk*n_H,
      blk,
      V,
      &solver->color_to_node_map,
      blk_max,
      ld_H,
      Br,
      X,
      m_B
    );

    memcpy(R, B, (sizeof *R) * m_B * blk);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, m_B, 1.0, R, m_B);
    R_norm = mpf_dlange(CblasColMajor, 'F', m_B, blk, R, m_B);

    #if DEBUG == 1
      printf("residual norm: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm/B_norm <= solver->tolerance)
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
  Hblk = NULL;
  Br = NULL;
  W = NULL;
  Vprev = NULL;
  V = NULL;
  Vlast = NULL;
  R = NULL;
}

void mpf_zhe_spbasis_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble temp1_c = ZERO_C;

  /* solver context */
  double B_norm = 0.0;
  double R_norm = 0.0;
  MPF_ComplexDouble trace = ZERO_C;
  double h_temp = 0.0;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X;

  /* meta */
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1 + solver->restarts;
  MPF_Int ld_H = inner_iterations;
  MPF_Int max_n_H = inner_iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = max_n_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* assign cpu memory to mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[m*blk*(m_H+1)];
  MPF_ComplexDouble *Hblk = &H[m_H*n_H];
  MPF_ComplexDouble *Br = &Hblk[blk*blk];

  /* handles to allocated cpu memory */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Vlast = &V[(m * blk) * inner_iterations];
  MPF_ComplexDouble *R = Vlast;

  /* required for sparsification */
  MPF_Int nz_new = 0;
  MPF_Int end_sparse = 0;

  //MPF_Int blk_max_fA = mpf_get_blk_max_fA();
  /* compute residual vectors block using initial approximation of solution */

  /* block vectors */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V)*m*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, ONE_C, V, m);
  B_norm = mpf_zlange (CblasColMajor, 'F', m, blk, B, m);
  R_norm = mpf_zlange(CblasColMajor, 'F', m, blk, V, m);

  #if DEBUG
    printf("residual norm: %1.4E\n", R_norm/B_norm);
  #endif

  if (R_norm/B_norm <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    temp1_c = mpf_scalar_z_normalize(ONE_C, R_norm);
    mpf_zscal(m*blk, &temp1_c, V, 1);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0].real = B_norm;
    Br[0].imag = 0.0;

    W = &V[(m*blk)];
    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Vprev, blk, m, ZERO_C, W, m);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, &ONE_C,
      Vprev, m, W, m, &ZERO_C, Hblk, blk);

    trace = ZERO_C;
    for (MPF_Int t = 0; t < blk; ++t)
    {
      trace = mpf_scalar_z_add(trace, Hblk[blk*t + t]);
    }
    H[0] = trace;
    temp1_c = mpf_scalar_z_subtract(ZERO_C, trace);
    mpf_zaxpy(m*blk, &temp1_c, Vprev, 1, W, 1);

    H[1].real = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', m, blk, W, m);
    H[1].imag = 0.0;
    temp1_c = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m*blk, &temp1_c, W, 1);

    MPF_Int j = 0;
    for (j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];

      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m, ZERO_C, W, m);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, &ONE_C,
        Vprev, m, W, m, &ZERO_C, Hblk, blk);

      trace = ZERO_C;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
      }

      H[ld_H*j + j] = trace;
      temp1_c = mpf_scalar_z_subtract(ZERO_C, trace);
      mpf_zaxpy(m*blk, &temp1_c, Vprev, 1, W, 1);

      Vprev = &V[(m*blk)*(j-1)];
      temp1_c = mpf_scalar_z_invert_sign(H[m_H*(j-1) + j]);
      mpf_zaxpy(m*blk, &temp1_c, Vprev , 1, W, 1);
      h_temp = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', m, blk, W, m);

      H[ld_H*j + j-1] = H[ld_H*(j-1) + j];
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }

      H[ld_H*j+j+1].real = h_temp;
      H[ld_H*j+j+1].imag = 0.0;
      temp1_c = mpf_scalar_z_divide(ONE_C, H[m_H*j + j+1]);
      mpf_zscal(m*blk, &temp1_c, W, 1);
      vecblk_z_block_sparsify(
        m,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    vecblk_z_block_sparsify(
      m,              /* length of input vector */
      blk,
      Vprev,            /* input vector */
      &V[j*(m*blk)],  /* output vector (compresed) */
      solver->max_blk_fA,
      solver->current_rhs,
      &solver->color_to_node_map,
      &nz_new);

    end_sparse += nz_new;    // better max error, this is the correct one

    /* solves upper triangular linear system and evaluates
       termination criteria */
    //mpf_qr_dsy_givens(m_H, n_H, 1, H, ld_H, Br);
    mpf_qr_zsy_givens_2(H, Br, m_H, n_H, 1);

    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);

    global_krylov_zge_sparse_basis_block_combine(
      solver->current_rhs,
      blk*n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m);

    /* computes residual */
    memcpy(R, B, (sizeof *R) * m * blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, sparse_layout, X, blk, m, ONE_C, R, m);
    R_norm = mpf_zlange(CblasColMajor, 'F', m, blk, R, m);

    #if DEBUG
      printf("residual norm: %1.4E\n", R_norm/B_norm);
    #endif

    if (R_norm/B_norm <= solver->tolerance)
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
  V = NULL;
  Vlast = NULL;
  R = NULL;
}

void mpf_zsy_spbasis_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  /* solver context */
  MPF_ComplexDouble B_norm = ZERO_C;
  double B_norm_abs = 0.0;
  MPF_ComplexDouble R_norm = ZERO_C;
  double R_norm_abs = 0.0;
  MPF_ComplexDouble trace = ZERO_C;
  MPF_ComplexDouble h_temp = ZERO_C;
  double h_tempf_abs = 0.0;
  MPF_ComplexDouble temp1_c = ZERO_C;

  MPF_ComplexDouble* B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble* X = (MPF_ComplexDouble*)X_dense->data;

  /* solver->*/
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int ld_H = inner_iterations;
  MPF_Int max_n_H = inner_iterations;
  MPF_Int m_H = ld_H;
  MPF_Int n_H = max_n_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* assign cpu memory to mathematical objects */
  MPF_ComplexDouble *V = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *H = &V[m_B*blk*(m_H+1)];
  MPF_ComplexDouble *Hblk = &H[m_H*n_H];
  MPF_ComplexDouble *Br = &Hblk[blk*blk];

  /* handles to allocated cpu memory */
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Vlast = &V[(m_B * blk) * inner_iterations];
  MPF_ComplexDouble *R = Vlast;
  /* required for sparsification */
  MPF_Int nz_new = 0;
  MPF_Int end_sparse = 0;
  //MPF_Int solver->max_blk_fA_fA = mpf_get_solver->max_blk_fA_fA();

  /* compute residual vectors block using initial approximation of solution
     block vectors */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  memcpy(V, B, (sizeof *V)*m_B*blk);

  //mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
  //  SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m_B, ONE_C, V, m_B);

  /* computes B_norm */
  //B_norm = mpf_zlange (CblasColMajor, 'F', m_B, blk, B, m_B);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, B,
    m_B*blk, B, m_B*blk, &ZERO_C, &temp1_c, 1);
  mpf_vectorized_z_sqrt(1, &temp1_c, &B_norm);
  mpf_vectorized_z_abs(1, &B_norm, &B_norm_abs);

  //R_norm = mpf_zlange(CblasColMajor, 'F', m_B, blk, V, m_B);
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, V,
    m_B*blk, V, m_B*blk, &ZERO_C, &temp1_c, 1);
  mpf_vectorized_z_sqrt(1, &temp1_c, &R_norm);
  mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);

  if (R_norm_abs/B_norm_abs <= solver->tolerance)
  {
    return;
  }

  /* outer-loop (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    temp1_c = mpf_scalar_z_divide(ONE_C, R_norm);
    mpf_zscal(m_B*blk, &temp1_c, V, 1);
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, 1, Br, m_H);
    Br[0] = R_norm; /* @BAD_STYLE: temp2_C is not evident what it is */

    /* computes W and H[0] */
    W = &V[(m_B*blk)];

    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
      Vprev, m_B, W, m_B, &ZERO_C, Hblk, blk);

    trace = ZERO_C;
    for (MPF_Int t = 0; t < blk; ++t)
    {
      trace = mpf_scalar_z_add(trace, Hblk[blk*t+t]);
    }
    H[0] = trace;
    temp1_c = mpf_scalar_z_subtract(ZERO_C, trace);
    mpf_zaxpy(m_B*blk, &temp1_c, Vprev, 1, W, 1);

    /* computes H[1] and normalizes W */
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, W,
      m_B*blk, W, m_B*blk, &ZERO_C, &temp1_c, 1);
    mpf_vectorized_z_sqrt(1, &temp1_c, &H[1]);
    temp1_c = mpf_scalar_z_divide(ONE_C, H[1]);
    mpf_zscal(m_B*blk, &temp1_c, W, 1);

    /* inner loop */
    MPF_Int j = 0;
    for (j = 1; j < inner_iterations; ++j)
    {
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];

      /* Computes W <-- A*V and trace <-- (Vprev^H)*W */
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
        Vprev, m_B, W, m_B, &ZERO_C, Hblk, blk);

      trace = ZERO_C;
      for (MPF_Int t = 0; t < blk; ++t)
      {
        trace = mpf_scalar_z_add(trace, Hblk[blk*t + t]);
      }

      /* Sets H[ld_H*j+j], Vprev and W */
      H[ld_H*j + j] = trace;
      temp1_c = mpf_scalar_z_subtract(ZERO_C, trace);
      mpf_zaxpy(m_B*blk, &temp1_c, Vprev , 1, W, 1);
      Vprev = &V[(m_B*blk)*(j-1)];
      temp1_c = mpf_scalar_z_subtract(ZERO_C, H[m_H*(j-1) + j]);
      mpf_zaxpy(m_B*blk, &temp1_c, Vprev , 1, W, 1);

      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
         W, m_B*blk, W, m_B*blk, &ZERO_C, &temp1_c, 1);

      mpf_vectorized_z_sqrt(1, &temp1_c, &h_temp);
      mpf_vectorized_z_abs(1, &h_temp, &h_tempf_abs);
      H[ld_H*j + j-1] = H[ld_H*(j-1) + j];
      if ((h_tempf_abs <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }

      /* sets H[ld_H*j + j+1] and applies sparsification to Vprev */
      H[ld_H*j + j+1] = h_temp;
      temp1_c = mpf_scalar_z_divide(ONE_C, H[m_H*j + j+1]);
      mpf_zscal(m_B*blk, &temp1_c, W, 1);
      vecblk_z_block_sparsify(
        m_B,            /* length of input vector */
        blk,
        Vprev,          /* input vector */
        &V[end_sparse], /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    if (j < inner_iterations-1)
    {
      vecblk_z_block_sparsify(
        m_B,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse], /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    /* solves upper triangular linear system and evaluates
       termination criteria */
    mpf_qr_zsy_givens(H, Br, ld_H, n_H, 1);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    global_krylov_zge_sparse_basis_block_combine(
      solver->current_rhs,
      blk*n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m_B);

    memcpy(R, B, (sizeof *R) * m_B * blk);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, sparse_layout, X, blk, m_B, ONE_C, R, m_B);

    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, R,
      m_B*blk, R, m_B*blk, &ZERO_C, &temp1_c, 1);

    mpf_vectorized_z_sqrt(1, &temp1_c, &R_norm);
    mpf_vectorized_z_abs(1, &R_norm, &R_norm_abs);
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
  Hblk = NULL;
  Br = NULL;
  W = NULL;
  Vprev = NULL;
  V = NULL;
  Vlast = NULL;
  R = NULL;
}
