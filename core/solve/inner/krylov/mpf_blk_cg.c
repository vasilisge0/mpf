#include "mpf.h"

void mpf_dsy_blk_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense)
{
  /* solver context */
  MPF_Int n = solver->ld;
  MPF_Int m_B = n;
  MPF_Int blk = solver->batch;
  double B_norm = 0.0;
  double R_norm = 0.0;
  double dvec_norm = 0;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* memory cpu */
  double *Rold = (double*)solver->inner_mem;
  double *Rnew = &Rold[m_B*blk];
  double *Dvec = &Rnew[m_B*blk];
  double *Dvec_new  = &Dvec[m_B*blk];
  double *Dvec_temp = &Dvec_new[m_B*blk];
  double *R = &Dvec_temp[m_B*blk];
  double *alpha_matrix = &R[m_B*blk];
  double *beta_matrix  = &alpha_matrix[blk*blk];
  double *zeta_matrix  = &beta_matrix[blk*blk];
  double *reflectors_array = &zeta_matrix[blk*blk];
  double *tempf_vecblk = NULL;
  MPF_Int *pivots_array = (MPF_Int*)mpf_malloc((sizeof *pivots_array)*m_B);

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold)*m_B*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    sparse_layout, X, blk, m_B, 1.0, Rold, m_B);
  B_norm = mpf_dlange(layout, 'F', m_B, blk, B, m_B);
  R_norm = mpf_dlange(layout, 'F', m_B, blk, Rold, m_B);
  memcpy(R, Rold, (sizeof *R) * m_B * blk);
  mpf_dgeqrf(layout, m_B, blk, R, m_B, reflectors_array);
  memcpy(Dvec, R, (sizeof *Dvec) * m_B * blk);
  mpf_dorgqr(layout, m_B, blk, blk, Dvec, m_B, reflectors_array);

  /* main loop (iterations ) */
  MPF_Int i = 0;
  while ((i < solver->iterations) && ((R_norm/B_norm > solver->tolerance)))
  {
    /* computes alpha_matrix */
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      sparse_layout, Dvec, blk, m_B, 0.0, Dvec_new, m_B);
    mpf_dgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0, Dvec, m_B,
      Dvec_new, m_B, 0.0, beta_matrix, blk);

    mpf_dgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0, Rold,
             m_B, Rold, m_B, 0.0, alpha_matrix, blk);
    memcpy(zeta_matrix, alpha_matrix, (sizeof *zeta_matrix)*blk*blk);     /* store R'R for later use */
    mpf_dtrsm(layout, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, blk, blk, 1.0,
             R, m_B, alpha_matrix, blk);
    mpf_dgesv(layout, blk, blk, beta_matrix, blk, pivots_array, alpha_matrix, blk);

    /* updates solution X and residual */
    mpf_dgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, 1.0, Dvec,
             m_B, alpha_matrix, blk, 1.0, X, m_B);
    memcpy(Rnew, Rold, (sizeof *Rnew)*m_B*blk);
    mpf_dgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, 1.0, Dvec,
             m_B, alpha_matrix, blk, 0.0, Dvec_temp, m_B);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr, sparse_layout,
      Dvec_temp, blk, m_B, 1.0, Rnew, m_B);

    /* computes beta_matrix parameters */
    mpf_dimatcopy('C', 'T', blk, blk, 1.0, R, m_B, blk);
    mpf_dgesv(layout, blk, blk, zeta_matrix, blk, pivots_array, R, blk);
    mpf_dimatcopy('C', 'T', blk, blk, 1.0, R, blk, m_B);
    mpf_dgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0, Rnew,
             m_B, Rnew, m_B, 0.0, beta_matrix, blk);
    R_norm = mpf_dlange(layout, 'F', m_B, blk, Rnew, m_B);
    mpf_dgemm(layout, CblasNoTrans, CblasNoTrans, blk, blk, blk, 1.0, R, m_B,
             beta_matrix, blk, 0.0, beta_matrix, blk);

    /* computes zeta_matrix (reorthogonalize) */
    memcpy(Rold, Rnew, (sizeof *Rold) * m_B * blk);
    mpf_dgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, 1.0, Dvec,
             m_B, beta_matrix, blk, 1.0, Rold, m_B);
    memcpy(Dvec, Rold, (sizeof *Dvec) * m_B * blk);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, Dvec, m_B, reflectors_array);
    dvec_norm = mpf_dlange(layout, 'F', blk, blk, Dvec, m_B);
    if (dvec_norm < 1e-10)  // *** NOTE: threshold should be able to be set dynamically
    {
      break;
    }
    memcpy(R, Dvec, (sizeof *R) * m_B * blk);
    mpf_dorgqr(LAPACK_COL_MAJOR, m_B,  blk, blk, Dvec,
              m_B, reflectors_array);

    /* swaps old and new residual vecblks and checks termination criteria */
    R_norm = mpf_dlange(layout, 'F', m_B, blk, Rnew, m_B);
    tempf_vecblk = Rold;
    Rold = Rnew;
    Rnew = tempf_vecblk;
    i = i + 1;
  }

  #if DEBUG == 1
    printf("r_norm: %1.4E\n", R_norm);
  #endif

  mpf_free(pivots_array);
  pivots_array = NULL;
  reflectors_array = NULL;
  tempf_vecblk = NULL;
  Rold = NULL;
  Rnew = NULL;
  Dvec = NULL;
  Dvec_new = NULL;
  Dvec_temp = NULL;
  alpha_matrix = NULL;
  beta_matrix = NULL;
  zeta_matrix = NULL;
}

void mpf_zsy_blk_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  /* solver context */
  MPF_Int i = 0;
  MPF_ComplexDouble B_norm = ZERO_C;
  MPF_ComplexDouble R_norm = ZERO_C;
  MPF_ComplexDouble tempf_complex;
  double R_norm_relative_abs;

  /* solver->*/
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* memory cpu */
  MPF_ComplexDouble *Rold = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *Rnew = &Rold[m_B*blk];
  MPF_ComplexDouble *Dvec = &Rnew[m_B*blk];
  MPF_ComplexDouble *Dvec_new = &Dvec[m_B*blk];
  MPF_ComplexDouble *Dvec_temp = &Dvec_new[m_B*blk];
  MPF_ComplexDouble *R = &Dvec_temp[m_B*blk];
  MPF_ComplexDouble *alpha_matrix = &R[m_B*blk];
  MPF_ComplexDouble *beta_matrix = &alpha_matrix[blk*blk];
  MPF_ComplexDouble *zeta_matrix = &beta_matrix[blk*blk];
  MPF_Int *pivots_array = (MPF_Int*)mpf_malloc((sizeof *pivots_array)*m_B);
  MPF_ComplexDouble *tempf_vecblk = NULL;

  /* first iteration (initialization) */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold)*m_B*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
    A->descr, sparse_layout, X, blk, m_B, ONE_C, Rold, m_B);
  mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, B,
    m_B*blk, B, m_B*blk, &ZERO_C, &B_norm, 1);
  mpf_vectorized_z_sqrt(1, &B_norm, &B_norm);
  mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Rold,
    m_B*blk, Rold, m_B*blk, &ZERO_C, &R_norm, 1);
  mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
  tempf_complex = mpf_scalar_z_divide(R_norm, B_norm);
  mpf_vectorized_z_abs(1, &tempf_complex, &R_norm_relative_abs);
  memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);
  mpf_gram_schmidt_zge(m_B, blk, Dvec, R, m_B);

  /* main loop */
  while ((i < solver->iterations) && (R_norm_relative_abs > solver->tolerance))
  {
    /* computes alpha_matrix */
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Dvec, blk, m_B, ZERO_C, Dvec_new, m_B);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Dvec,
      m_B, Dvec_new, m_B, &ZERO_C, beta_matrix, blk);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Rold,
      m_B, Rold, m_B, &ZERO_C, alpha_matrix, blk);
    memcpy(zeta_matrix, alpha_matrix, (sizeof *zeta_matrix)*blk*blk);     // store R'R for later use
    mpf_ztrsm(layout, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, blk, blk,
      &ONE_C, R, m_B, alpha_matrix, blk);
    mpf_zgesv(layout, blk, blk, beta_matrix, blk, pivots_array, alpha_matrix,
      blk);

    /* update solution X and residual */
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, alpha_matrix, blk, &ONE_C, X, m_B);
    memcpy(Rnew, Rold, (sizeof *Rnew) * m_B * blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, alpha_matrix, blk, &ZERO_C, Dvec_temp, m_B);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, sparse_layout, Dvec_temp, blk, m_B, ONE_C, Rnew, m_B);

    /* computes beta parameters */
    mpf_zimatcopy('C', 'T', blk, blk, ONE_C, R, m_B, blk);
    mpf_zgesv(layout, blk, blk, zeta_matrix, blk, pivots_array,
             R, blk);
    mpf_zimatcopy('C', 'T', blk, blk, ONE_C, R, blk, m_B);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Rnew,
             m_B, Rnew, m_B, &ZERO_C, beta_matrix, blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, blk, blk, blk, &ONE_C, R, m_B,
             beta_matrix, blk, &ZERO_C, beta_matrix, blk);

    /* compute zeta (reorthogonalize) */
    memcpy(Rold, Rnew, (sizeof *Rold)*m_B*blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, beta_matrix, blk, &ONE_C, Rold, m_B);
    memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);
    mpf_gram_schmidt_zge(m_B, blk, Dvec, R, m_B);
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C,
      Rnew, m_B*blk, Rnew, m_B*blk, &ZERO_C, &R_norm, 1);
    mpf_vectorized_z_sqrt(1, &R_norm, &R_norm);
    tempf_vecblk = Rold;
    Rold = Rnew;
    Rnew = tempf_vecblk;

    /* checks termination criteria */
    i = i + 1;
    tempf_complex = mpf_scalar_z_divide(R_norm, B_norm);
    mpf_vectorized_z_abs(1, &tempf_complex, &R_norm_relative_abs);
    #if DEBUG_DENSE == 1
      printf("matrix_R_norm: %1.4E\n", R_norm_relative_abs);
    #endif
  }

  mpf_free(pivots_array);
  pivots_array = NULL;
  tempf_vecblk = NULL;
  Rold = NULL;
  Rnew = NULL;
  Dvec = NULL;
  Dvec_new = NULL;
  Dvec_temp = NULL;
  alpha_matrix = NULL;
  beta_matrix = NULL;
  zeta_matrix = NULL;
}

void mpf_zhe_blk_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* constants */
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  /* solver context */
  MPF_Int i = 0;
  double B_norm = 0.0;
  double R_norm = 0.0;

  /* solver->*/
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* memory cpu */
  MPF_ComplexDouble *Rold = (MPF_ComplexDouble *) solver->inner_mem;
  MPF_ComplexDouble *Rnew = &Rold[m_B*blk];
  MPF_ComplexDouble *Dvec = &Rnew[m_B*blk];
  MPF_ComplexDouble *Dvec_new = &Dvec[m_B*blk];
  MPF_ComplexDouble *Dvec_temp = &Dvec_new[m_B*blk];
  MPF_ComplexDouble *R = &Dvec_temp[m_B*blk];
  MPF_ComplexDouble *alpha_matrix = &R[m_B*blk];
  MPF_ComplexDouble *beta_matrix = &alpha_matrix[blk*blk];
  MPF_ComplexDouble *zeta_matrix = &beta_matrix[blk*blk];
  MPF_Int *pivots_array = (MPF_Int*)mpf_malloc((sizeof *pivots_array)*m_B);
  MPF_ComplexDouble *tempf_vecblk = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold)*m_B*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
    sparse_layout, X, blk, m_B, ONE_C, Rold, m_B);
  B_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, B, m_B);
  R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rold, m_B);
  memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);
  mpf_gram_schmidt_zhe(m_B, blk, Dvec, R, m_B);

  /* main loop */
  while ((i < solver->iterations) && (R_norm/B_norm > solver->tolerance))
  {
    /* computes alpha_matrix */
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      sparse_layout, Dvec, blk, m_B, ZERO_C, Dvec_new, m_B);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Dvec,
      m_B, Dvec_new, m_B, &ZERO_C, beta_matrix, blk);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Rold,
      m_B, Rold, m_B, &ZERO_C, alpha_matrix, blk);
    memcpy(zeta_matrix, alpha_matrix, (sizeof *zeta_matrix)*blk*blk);     // store R'R for later use
    mpf_ztrsm(layout, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, blk,
      blk, &ONE_C, R, m_B, alpha_matrix, blk);
    mpf_zgesv(layout, blk, blk, beta_matrix, blk, pivots_array, alpha_matrix,
    blk);

    /* update solution X and residual */
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, alpha_matrix, blk, &ONE_C, X, m_B);
    memcpy(Rnew, Rold, (sizeof *Rnew) * m_B * blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, alpha_matrix, blk, &ZERO_C, Dvec_temp, m_B);
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle,
      A->descr, sparse_layout, Dvec_temp, blk, m_B, ONE_C, Rnew, m_B);

    /* computes beta parameters */
    mpf_zimatcopy('C', 'C', blk, blk, ONE_C, R, m_B, blk);
    mpf_zgesv(layout, blk, blk, zeta_matrix, blk, pivots_array,
             R, blk);
    mpf_zimatcopy('C', 'C', blk, blk, ONE_C, R, blk, m_B);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, blk, blk, m_B, &ONE_C, Rnew,
             m_B, Rnew, m_B, &ZERO_C, beta_matrix, blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, blk, blk, blk, &ONE_C, R, m_B,
             beta_matrix, blk, &ZERO_C, beta_matrix, blk);

    /* compute zeta (reorthogonalize) */
    memcpy(Rold, Rnew, (sizeof *Rold)*m_B*blk);
    mpf_zgemm(layout, CblasNoTrans, CblasNoTrans, m_B, blk, blk, &ONE_C, Dvec,
             m_B, beta_matrix, blk, &ONE_C, Rold, m_B);
    memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);
    mpf_gram_schmidt_zhe(m_B, blk, Dvec, R, m_B);
    R_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rnew, m_B);

    tempf_vecblk = Rold;
    Rold = Rnew;
    Rnew = tempf_vecblk;

    /* checks termination criteria */
    i = i + 1;
    #if DEBUG_DENSE == 1
      printf("matrix_R_norm: %1.4E\n", R_norm/B_norm);
    #endif
  }

  mpf_free(pivots_array);
  pivots_array = NULL;
  tempf_vecblk = NULL;
  Rold = NULL;
  Rnew = NULL;
  Dvec = NULL;
  Dvec_new = NULL;
  Dvec_temp = NULL;
  alpha_matrix = NULL;
  beta_matrix = NULL;
  zeta_matrix = NULL;
}

void mpf_blk_cg_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations
)
{
  MPF_Solver* solver = &context->solver;
  MPF_Int ld = context->A.m;
  solver->ld = ld;
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;
  solver->framework = MPF_SOLVER_FRAME_MPF;
  context->args.n_inner_solve = 3;

  if ((solver->precond_type == MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DSY_CG;
      solver->device = MPF_DEVICE_CPU;
        solver->inner_function = &mpf_dsy_blk_cg;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_CG;
      solver->inner_function = &mpf_zsy_blk_cg;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZHE_CG;
      solver->inner_function = &mpf_zhe_blk_cg;
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
  solver->inner_get_mem_size_function = &mpf_blk_cg_get_mem_size;
}

void mpf_blk_cg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double) *
      (m_B*blk     /* size_residual_old */
      +m_B*blk     /* size_residual_new */
      +m_B*blk     /* size_direction */
      +m_B*blk     /* size_direction_new */
      +m_B*blk     /* size_direction_temp */
      +m_B*blk     /* size_R */
      +blk*blk     /* size_alpha */
      +blk*blk     /* size_beta */
      +blk*blk     /* size_zeta */
      +blk         /* size_reflectors */
      +blk);       /* size_pivots */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
     solver->inner_bytes = sizeof(MPF_ComplexDouble) *
      (m_B*blk   /* size_residual_old */
      +m_B*blk   /* size_residual_new */
      +m_B*blk   /* size_direction */
      +m_B*blk   /* size_direction_new */
      +m_B*blk   /* size_direction_temp */
      +m_B*blk   /* size_R */
      +blk*blk   /* size_alpha */
      +blk*blk   /* size_beta */
      +blk*blk   /* size_zeta */
      +blk       /* size_reflectors */
      +blk);     /* size_pivots */
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble) *
      (m_B*blk   /* size_residual_old */
      +m_B*blk   /* size_residual_new */
      +m_B*blk   /* size_direction */
      +m_B*blk   /* size_direction_new */
      +m_B*blk   /* size_direction_temp */
      +m_B*blk   /* size_R */
      +blk*blk   /* size_alpha */
      +blk*blk   /* size_beta */
      +blk*blk   /* size_zeta */
      +blk       /* size_reflectors */
      +blk);     /* size_pivots */
  }
}
