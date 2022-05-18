#include "mpf.h"

void mpf_dsy_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  if (solver->precond_type == MPF_PRECOND_NONE) /* do not use preconditioner */
  {
    /* solver solver */
    double norm_b = 0.0;
    double norm_r = 0.0;
    double alpha = 0.0;
    double beta  = 0.0;
    MPF_Int m_B = solver->ld;
    double *tempf_vector = NULL;
    MPF_Layout layout = MPF_COL_MAJOR;
    MPF_LayoutSparse sparse_layout;

    double *b = (double*)b_dense->data;
    double *x = (double*)x_dense->data;

    /* memory cpu */
    double *r_new = (double*)solver->inner_mem;
    double *r_old = &r_new[m_B];
    double *p = &r_old[m_B];

    /* first iteration */
    mpf_convert_layout_to_sparse(layout, &sparse_layout);
    memcpy(r_old, b, (sizeof *r_old)*m_B);
    norm_b = cblas_dnrm2(m_B, b, 1);

    MPF_Int status = mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0,
      A->handle, A->descr, x, 1.0, r_old);

    norm_r = mpf_dnrm2(m_B, r_old, 1);
    memcpy(p, r_old, (sizeof *p)*m_B);
    norm_r = mpf_dnrm2(m_B, b, 1);

    MPF_Int i = 0;
    while ((i < solver->iterations) && (norm_r/norm_b > solver->tolerance))
    {
      status = mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle,
        A->descr, p, 0.0, r_new);

      alpha = mpf_ddot(m_B, r_new, 1, p, 1);
      alpha = mpf_ddot(m_B, r_old, 1, r_old, 1)/alpha;
      mpf_daxpy(m_B, alpha, p, 1, x, 1);

      mpf_dscal(m_B, -alpha, r_new, 1);
      mpf_daxpy(m_B, 1.0, r_old, 1, r_new, 1);
      beta = mpf_ddot(m_B, r_new, 1, r_new, 1);
      beta = beta/mpf_ddot(m_B, r_old, 1, r_old, 1);
      mpf_dscal(m_B, beta, p, 1);
      mpf_daxpy(m_B, 1.0, r_new, 1, p, 1);

      norm_r = mpf_dnrm2(m_B, r_new, 1);

      tempf_vector = r_old;
      r_old = r_new;
      r_new = tempf_vector;
      i = i + 1;

      #if MPF_PRINTOUT_SOLVER
        printf("relative residual: %1.4E\n", norm_r/norm_b);
      #endif
    }

    r_new = NULL;
    r_old = NULL;
    tempf_vector = NULL;
  }
  else /* use solver->M as preconditioner */
  {
    /* solver context */
    double norm_b = 0.0;
    double norm_r = 0.0;
    double alpha = 0.0;
    double beta  = 0.0;
    double trace_r_old = 0.0;
    double trace_r_new = 0.0;
    double gamma = 0.0;
    double t = 0.0;
    MPF_Int m_B = solver->ld;
    double *swap = NULL;
    MPF_Layout layout = MPF_COL_MAJOR;
    MPF_LayoutSparse sparse_layout;

    double *b = (double*)b_dense->data;
    double *x = (double*)x_dense->data;

    /* memory cpu */
    double *r_new = (double*)solver->inner_mem;
    double *r_old = &r_new[m_B];
    double *p = &r_old[m_B];

    /* added temporary vectors */
    double *z = &p[m_B];
    double *z_new = &z[m_B];

    /* first iteration */
    mpf_convert_layout_to_sparse(layout, &sparse_layout);
    memcpy(r_old, b, (sizeof *r_old)*m_B);

    mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle,
      A->descr, x, 1.0, r_old);

    solver->precond_apply_function(solver, r_old, z);
    norm_r = mpf_dnrm2(m_B, z, 1);
    norm_b = norm_r;
    memcpy((void*)p, (void*)z, sizeof(*z)*m_B);

    MPF_Int i = 0;
    while ((i < solver->iterations) && (norm_r/norm_b > solver->tolerance))
    {
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        p, 0.0, r_new);

      trace_r_old = mpf_ddot(m_B, r_old, 1, z, 1);
      gamma = mpf_ddot(m_B, r_new, 1, p, 1);
      alpha = trace_r_old/gamma;

      /* updates x and residual */
      mpf_daxpy(m_B, alpha, p, 1, x, 1);
      mpf_dscal(m_B, -alpha, r_new, 1);
      mpf_daxpy(m_B, 1.0, r_old, 1, r_new, 1);

      /* applies preconditioning step */
      solver->precond_apply_function(solver, r_new, z_new);
      norm_r = mpf_dnrm2(m_B, r_new, 1);
      t = mpf_ddot(m_B, r_old, 1, z, 1);
      if (norm_r/norm_b < solver->tolerance)
      {
        break;
      }

      /* computes beta */
      trace_r_new = mpf_ddot(m_B, r_new, 1, z_new, 1);
      beta = trace_r_new/t;

      /* update direction vector */
      mpf_dscal(m_B, beta, p, 1);
      mpf_daxpy(m_B, 1.0, z_new, 1, p, 1);

      swap = r_old;
      r_old = r_new;
      r_new = swap;
      swap = z;
      z = z_new;
      z_new = swap;

      i += 1;

      #if MPF_PRINTOUT_SOLVER
        printf("relative residual: %1.4E\n", norm_r / norm_b);
      #endif
    }

    r_new = NULL;
    r_old = NULL;
    swap = NULL;
  }
}

void mpf_zsy_cg
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

  /* solver context */
  MPF_ComplexDouble alpha;
  MPF_ComplexDouble beta;
  MPF_ComplexDouble tempf_complex;
  MPF_ComplexDouble norm_b;
  MPF_ComplexDouble r_norm;
  double norm_abs = 0.0;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *b = (MPF_ComplexDouble*) b_dense->data;
  MPF_ComplexDouble *x = (MPF_ComplexDouble*) x_dense->data;

  /* meta */
  MPF_Int m_B = A->m;
  MPF_Int ld = m_B;

  /* memory cpu */
  MPF_ComplexDouble *r_new = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *r_old = &r_new[m_B];
  MPF_ComplexDouble *dvec = &r_old[m_B];

  /* handles */
  MPF_ComplexDouble *swap_vector = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(r_old, b, (sizeof *r_old)*m_B);
  mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, b,
    ld, b, ld, &ZERO_C, &norm_b, 1);
  mpf_vectorized_z_sqrt(1, &norm_b, &norm_b);
  mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
    x, ONE_C, r_old);
  mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_old, ld,
           r_old, ld, &ZERO_C, &r_norm, 1);
  memcpy(dvec, r_old, (sizeof *dvec) * m_B);
  mpf_vectorized_z_sqrt(1, &r_norm, &r_norm);
  tempf_complex = mpf_scalar_z_divide(r_norm, norm_b);
  mpf_vectorized_z_abs(1, &tempf_complex, &norm_abs);

  #if DEBUG == 1
    printf("residual_norm: %1.4E\n", norm_abs);
  #endif

  /* main loop */
  MPF_Int i = 0;
  while ((i < solver->iterations) && (norm_abs > solver->tolerance))
  {
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      dvec, ZERO_C, r_new);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_new, ld,
             dvec, ld, &ZERO_C, &tempf_complex, 1);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_old, ld,
             r_old, ld, &ZERO_C, &alpha, 1);
    alpha = mpf_scalar_z_divide(alpha, tempf_complex);
    mpf_zaxpy(m_B, &alpha, dvec, 1, x, 1);
    tempf_complex = mpf_scalar_z_invert_sign(alpha);//-alpha in dsy_orihinal
    mpf_zscal(m_B, &tempf_complex, r_new, 1);
    mpf_zaxpy(m_B, &ONE_C, r_old, 1, r_new, 1);

    mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_new, ld,
             r_new, ld, &ZERO_C, &beta, 1);
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_old, ld,
             r_old, ld, &ZERO_C, &tempf_complex, 1);
    beta = mpf_scalar_z_divide(beta, tempf_complex);
    mpf_zscal(m_B, &beta, dvec, 1);
    mpf_zaxpy(m_B, &ONE_C, r_new, 1, dvec, 1);

    swap_vector = r_old;
    r_old = r_new;
    r_new = swap_vector;
    mpf_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C, r_old, ld,
             r_old, ld, &ZERO_C, &r_norm, 1);
    mpf_vectorized_z_sqrt(1, &r_norm, &r_norm);
    r_norm = mpf_scalar_z_divide(r_norm, norm_b);
    mpf_vectorized_z_abs(1, &r_norm, &norm_abs);
    i = i + 1;
  }
  printf("residual_norm: %1.4E\n", norm_abs);

  r_new = NULL;
  r_old = NULL;
  dvec = NULL;
}

void mpf_zhe_cg
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

  /* solver context */
  MPF_ComplexDouble alpha;
  MPF_ComplexDouble beta;
  MPF_ComplexDouble tempf_complex;
  double norm_b;
  double r_norm = 0.0;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* meta */
  MPF_Int n = solver->ld;
  MPF_Int m_B = n;
  MPF_Int ld = n;

  MPF_ComplexDouble *b = (MPF_ComplexDouble*)b_dense->data;
  MPF_ComplexDouble *x = (MPF_ComplexDouble*)x_dense->data;

  /* memory cpu */
  MPF_ComplexDouble *r_new = (MPF_ComplexDouble*)solver->inner_mem;
  MPF_ComplexDouble *r_old = &r_new[m_B];
  MPF_ComplexDouble *dvec  = &r_old[m_B];

  /* handles */
  MPF_ComplexDouble *swap_vector = NULL;

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(r_old, b, (sizeof *r_old)*m_B);
  MPF_Int status = mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C,
    A->handle, A->descr, x, ONE_C, r_old);
  norm_b = mpf_dznrm2(m_B, b, 1);
  r_norm = mpf_dznrm2(m_B, r_old, 1);
  memcpy(dvec, r_old, (sizeof *dvec)*m_B);

  /* main loop */
  MPF_Int i = 0;
  while ((i < solver->iterations) && (r_norm/norm_b > solver->tolerance))
  {
    mpf_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle,
      A->descr, dvec, ZERO_C, r_new);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      r_new, ld, dvec, ld, &ZERO_C, &tempf_complex, 1);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      r_old, ld, r_old, ld, &ZERO_C, &alpha, 1);
    alpha = mpf_scalar_z_divide(alpha, tempf_complex);
    mpf_zaxpy(m_B, &alpha, dvec, 1, x, 1);
    tempf_complex = mpf_scalar_z_invert_sign(alpha);//-alpha in dsy_orihinal
    mpf_zscal(m_B, &tempf_complex, r_new, 1);
    mpf_zaxpy(m_B, &ONE_C, r_old, 1, r_new, 1);

    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      r_new, ld, r_new, ld, &ZERO_C, &beta, 1);
    mpf_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      r_old, ld, r_old, ld, &ZERO_C, &tempf_complex, 1);
    beta = mpf_scalar_z_divide(beta, tempf_complex);
    mpf_zscal(m_B, &beta, dvec, 1);
    mpf_zaxpy(m_B, &ONE_C, r_new, 1, dvec, 1);
    r_norm = mpf_dznrm2(m_B, r_new, 1);

    swap_vector = r_old;
    r_old = r_new;
    r_new = swap_vector;
    i = i + 1;
  }
  printf("norm_abs: %1.4E\n", r_norm/norm_b);

  r_new = NULL;
  r_old = NULL;
  dvec = NULL;
}

void mpf_cg_init
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
  solver->precond_type = MPF_PRECOND_NONE;

printf("tolerance: %1.4E, iterations: %d\n", solver->tolerance, solver->iterations);

  if ((solver->precond_type == MPF_PRECOND_NONE) &&
      (solver->defl_type == MPF_DEFL_NONE))
  {
    if (solver->data_type == MPF_REAL)
    {
      solver->inner_type = MPF_SOLVER_DSY_CG;
      solver->device = MPF_DEVICE_CPU;
        solver->inner_function = &mpf_dsy_cg;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
    {
      solver->inner_type = MPF_SOLVER_ZSY_CG;
      solver->inner_function = &mpf_zsy_cg;
      solver->device = MPF_DEVICE_CPU;
    }
    else if ((solver->data_type == MPF_COMPLEX)
            && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
    {
      solver->inner_type = MPF_SOLVER_ZHE_CG;
      solver->inner_function = &mpf_zhe_cg;
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
  solver->inner_get_mem_size_function = &mpf_cg_get_mem_size;
}

void mpf_cg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int precond_size = 0;
  if (solver->precond_type != MPF_PRECOND_NONE)
  {
    precond_size = n*2;
  }

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
        (n*2   /* size_residuals */
        +n*2   /* size_direction */
        +precond_size); 
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble) *
        (n*2   /* size_residuals */
        +n     /* size_direction */
        +precond_size); 
  }
  else if (solver->data_type == MPF_COMPLEX_32)
  {
    solver->inner_bytes = sizeof(MPF_Complex) *
        (n*2   /* size_residuals */
        +n     /* size_direction */
        +precond_size); 
  }
}

void mpf_cg_alloc
(
  MPF_Solver *solver
)
{
  solver->inner_mem = mpf_malloc(solver->inner_bytes);
}

