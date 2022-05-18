#include "mpf.h"

void mpf_defl_ev_cg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int n_ev_max
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;
  solver->defl_n_ev_max = n_ev_max;
  solver->n_defl = n_ev_max;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_CG;
    solver->inner_function = &mpf_dsy_defl_ev_cg;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_LANCZOS;
    solver->inner_function = &mpf_zsy_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_LANCZOS;
    solver->inner_function = &mpf_zhe_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  mpf_ev_defl_cg_get_mem_size(solver);
}

void mpf_ev_defl_cg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int iterations = solver->iterations;
  MPF_Int n_ev_max = solver->n_defl;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)
      *(
         n                /* size_residual */
        +n                /* size_residual */
        +n*(iterations+1) /* size_P */
      );
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)
      * (n*blk                     /* size_B */
        +n*blk                     /* size_X */
        +n*(iterations+1)*n_ev_max /* size_V */
        +iterations*iterations     /* size_H */
        +iterations                /* size_br */
        +n);                       /* size_residual */
  }
}

void mpf_dsy_defl_ev_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* solver context */
  double norm_b = 0.0;
  double r_norm = 0.0;
  double alpha = 0.0;
  double beta  = 0.0;
  MPF_Int i = 0;
  MPF_Int n = solver->ld;
  MPF_Int m_B = n;
  double *tempf_vector = NULL;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_Int m_H = solver->iterations;
  MPF_Int m_G = solver->n_defl+m_H;
  MPF_Int n_B = 1;

  double *b = (double*)b_dense->data;
  double *x = (double*)x_dense->data;

  /* unpacks memory cpu */
  double *r_new = (double*)solver->inner_mem;
  double *r_old = &r_new[m_B];
  double *P = &r_old[m_B];

  /* unpacks memory_defl (deflation memory) */
  MPF_Int n_defl = solver->n_defl;
  double *Vdefl = (double*)solver->inner_mem;
  double *Hdefl = &Vdefl[m_B*solver->iterations*solver->batch];
  double *refs_defl_array = &Hdefl[4*m_H*m_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m_B*solver->batch];
  double *e = &Mdefl[m_B*solver->batch*solver->iterations];
  double *d = &e[solver->iterations+solver->n_defl];
  double *refl_ev = &d[solver->iterations+solver->n_defl];
  double *evals = &refl_ev[solver->iterations];
  double *evecs = &evals[solver->iterations];
  double *G = &evecs[(solver->n_defl+m_H)*solver->n_defl];  /* to be assigned */
  double *F = &G[(solver->iterations+n_defl)*(solver->iterations+n_defl)];  /* to be assigned */
  double *Z = &F[(solver->iterations+n_defl)*(solver->iterations+n_defl)];

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(r_old, b, (sizeof *r_old)*m_B);
  norm_b = cblas_dnrm2(m_B, b, 1);
  mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle,
    A->descr, x, 1.0, r_old);
  r_norm = mpf_dnrm2(m_B, r_old, 1);
  memcpy(P, r_old, (sizeof *P)*m_B);

  #if STATUS
    printf("relative residual: %1.4E\n", r_norm / norm_b);
  #endif

  r_norm = mpf_dnrm2(m_B, b, 1);

  if (solver->use_defl)
  {
    mpf_matrix_d_announce(Vdefl, 10, n_defl, m_B, "Vdefl");

    /* computes Vdefl^T*(A*Vdefl) */
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vdefl, n_defl, m_B, 0.0, Z, m_B);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_defl, m_B, 1.0,
      Vdefl, m_B, Z, m_B, 0.0, Hdefl, n_defl);

    /* initializes x */
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_B, m_B, 1.0,
      Vdefl, m_B, r_old, m_B, 0.0, Tdefl, n_defl);
    mpf_qr_dsy_ref_givens(n_defl, n_defl, n_B, Hdefl, n_defl, Tdefl,
      refs_defl_array);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_defl, n_B, 1.0, Hdefl, n_defl, Tdefl, n_defl);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_defl, 1.0,
      Vdefl, m_B, Tdefl, n_defl, 1.0, x, m_B);

    /* computes residual */
    mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle,
      A->descr, x, 1.0, r_old);

    /* initializes p */
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_defl, -1.0,
      Vdefl, m_B, Tdefl, n_defl, 0.0, P, m_B);
    mpf_daxpy(m_B, 1.0, r_old, 1, P, 1);

    /* main loop (iterations) */
    while ((i < solver->iterations) && (r_norm/norm_b > solver->tolerance))
    {
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        &P[m_B*i], 0.0, r_new);
      alpha = mpf_ddot(m_B, r_new, 1, &P[m_B*i], 1);
      alpha = mpf_ddot(m_B, r_old, 1, r_old, 1)/alpha;
      mpf_daxpy(m_B, alpha, &P[m_B*i], 1, x, 1);

      mpf_dscal(m_B, -alpha, r_new, 1);
      mpf_daxpy(m_B, 1.0, r_old, 1, r_new, 1);
      beta = mpf_ddot(m_B, r_new, 1, r_new, 1);
      beta = beta/mpf_ddot(m_B, r_old, 1, r_old, 1);
      memcpy(&P[m_B*(i+1)], &P[m_B*i], (sizeof *P)*m_B);
      mpf_dscal(m_B, beta, &P[m_B*(i+1)], 1);

      /* update p (dvec) */
      mpf_daxpy(m_B, 1.0, r_new, 1, &P[m_B*(i+1)], 1);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_B, m_B, 1.0,
        Vdefl, m_B, r_new, m_B, 0.0, Tdefl, n_defl);
      mpf_qr_dsy_rhs_givens(n_defl, n_defl, n_B, Hdefl, n_defl, Tdefl,
        refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        n_defl, n_B, 1.0, Hdefl, n_defl, Tdefl, n_defl);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, 1, n_defl, -1.0,
        Vdefl, m_B, Tdefl, n_defl, 1.0, &P[m_B*(i+1)], m_B);

      r_norm = mpf_dnrm2(m_B, r_new, 1);
      printf("i: %d, r_norm/norm_b: %1.4E\n", i, r_norm / norm_b);
      tempf_vector = r_old;
      r_old = r_new;
      r_new = tempf_vector;
      i = i + 1;
    }

    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vdefl, n_defl, m_B, 0.0, Z, m_B);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, P, m_H, m_B, 0.0, &Z[m_B*n_defl], m_B);

    /* computes G */
    mpf_zeros_d_set(MPF_COL_MAJOR, m_G, m_G, G, m_G);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_G, m_G, m_B,
      1.0, Z, m_B, Z, m_B, 0.0, G, m_G);
    mpf_matrix_d_announce(G, m_G, m_G, m_G, "G");

    /* computes F */
    mpf_zeros_d_set(MPF_COL_MAJOR, m_G, m_G, F, m_G);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, m_G, m_B,
      1.0, Vdefl, m_B, Z, m_B, 0.0, F, m_G);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_G, m_B,
      1.0, P, m_B, Z, m_B, 0.0, &F[n_defl], m_G);

    mpf_matrix_d_announce(F, m_G, m_G, m_G, "F");

    printf("m_G: %d, m_s: %d\n", m_G, m_H);

    MPF_Int n_ev_found = 0;
    //MPF_Int n_blocks = 0;
    MPF_Int *iblock = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
    //MPF_Int *isplit = mpf_malloc((sizeof *iblock)*m_H);
    MPF_Int *issup = (MPF_Int*)mpf_malloc((sizeof *issup)*m_H*2);
    lapack_logical tryrac;

    /* computes cholesky factorization of matrix F */
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', m_G, F, m_G);

    /* reduces generalized eigenvalue problem to standard form */
    MPF_Int info = 0;
    info = LAPACKE_dsygst(LAPACK_COL_MAJOR, 1, 'U', m_G, G, m_G, F, m_G);
    printf("info: %d\n", info);

    /* converts to tridiagonal matrix */
    LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', m_G, G, m_G, d, e, refl_ev);

    /* computes eigenvalues and eigenvectors */
    printf("n_defl: %d\n", n_defl);
    LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', m_G, d, e, 0.0, 0.0, 1, n_defl,
      &n_ev_found, evals, evecs, m_G, n_defl, issup, &tryrac);
    mpf_matrix_d_announce(evals, n_defl, 1, m_G, "evals");

    memcpy(Z, Vdefl, (sizeof *Z)*m_B*n_defl);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, n_defl, n_defl,
      1.0, Z, m_B, evecs, m_G, 0.0, Vdefl, m_B);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, n_defl, m_H,
      1.0, P, m_B, &evecs[n_defl], m_G, 1.0, Vdefl, m_B);

    r_new = NULL;
    r_old = NULL;
    tempf_vector = NULL;
  }
  else
  {
    /* main loop (iterations) */
    while ((i < solver->iterations) && (r_norm/norm_b > solver->tolerance))
    {
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        &P[m_B*i], 0.0, r_new);
      alpha = mpf_ddot(m_B, r_new, 1, &P[m_B*i], 1);
      alpha = mpf_ddot(m_B, r_old, 1, r_old, 1)/alpha;
      mpf_daxpy(m_B, alpha, &P[m_B*i], 1, x, 1);

      mpf_dscal(m_B, -alpha, r_new, 1);
      mpf_daxpy(m_B, 1.0, r_old, 1, r_new, 1);
      beta = mpf_ddot(m_B, r_new, 1, r_new, 1);
      beta = beta/mpf_ddot(m_B, r_old, 1, r_old, 1);
      memcpy(&P[m_B*(i+1)], &P[m_B*i], (sizeof *P)*m_B);
      mpf_dscal(m_B, beta, &P[m_B*(i+1)], 1);
      mpf_daxpy(m_B, 1.0, r_new, 1, &P[m_B*(i+1)], 1);

      r_norm = mpf_dnrm2(m_B, r_new, 1);
      printf("i: %d, r_norm/norm_b: i, %1.4E\n", i, r_norm/norm_b);
      tempf_vector = r_old;
      r_old = r_new;
      r_new = tempf_vector;
      i = i + 1;
    }

    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, P, m_H, m_B, 0.0, Z, m_B);

    /* computes G */
    mpf_zeros_d_set(MPF_COL_MAJOR, m_G, m_G, G, m_G);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_H, m_B,
      1.0, Z, m_B, Z, m_B, 0.0, G, m_G);

    mpf_matrix_d_announce(G, m_H, m_H, m_G, "G");

    /* computes F */
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, m_H, F, m_G);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_H, m_B,
      1.0, P, m_B, Z, m_B, 0.0, F, m_G);

    mpf_matrix_d_announce(F, m_H, m_H, m_G, "F");

    printf("m_G: %d, m_s: %d\n", m_G, m_H);

    MPF_Int n_ev_found = 0;
    //MPF_Int n_blocks = 0;
    MPF_Int *iblock = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
    //MPF_Int *isplit = mpf_malloc((sizeof *iblock)*m_H);
    MPF_Int *issup = (MPF_Int*)mpf_malloc((sizeof *issup)*m_H*2);
    lapack_logical tryrac;

    /* computes cholesky factorization of matrix F */
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', m_H, F, m_G);
    mpf_matrix_d_announce(F, m_H, m_H, m_G, "F (chol)");

    /* reduces generalized eigenvalue problem to standard form */
    MPF_Int info = 0;
    info = LAPACKE_dsygst(LAPACK_COL_MAJOR, 1, 'U', m_H, G, m_G, F, m_G);
    printf("info: %d\n", info);

    /* converts to tridiagonal matrix */
    LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', m_H, G, m_G, d, e, refl_ev);

    /* computes eigenvalues and eigenvectors */
    printf("n_defl: %d\n", n_defl);
    LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', m_H, d, e, 0.0, 0.0, 1, n_defl,
      &n_ev_found, evals, evecs, m_H, n_defl, issup, &tryrac);

    mpf_matrix_d_announce(evals, n_defl, 1, n_defl, "evals");
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, n_defl, m_H,
      1.0, P, m_B, evecs, m_H, 0.0, Vdefl, m_B);
  }
  r_new = NULL;
  r_old = NULL;
  tempf_vector = NULL;
}
