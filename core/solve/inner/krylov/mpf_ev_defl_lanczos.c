#include "mpf.h"

void mpf_dsy_defl_ev_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  printf("\n<-- IN DEFL_EV lanczos --> \n");
  /* context */
  MPF_Int k = 0;
  MPF_Int j = 0;
  double b_norm;
  double r_norm;
  double h_temp;

  /* solver->*/
  MPF_Int n_defl = solver->n_defl;
  MPF_Int outer_iterations = solver->restarts+1;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int m = solver->ld;
  MPF_Int n_B = solver->batch;
  MPF_Int ld_H = solver->iterations;
  MPF_Int m_H = solver->iterations;
  MPF_Int n_H = solver->iterations;
  //MPF_Int n_V = solver->iterations;
  MPF_Int m_G = n_defl+m_H;

  /* unpacks memory (used for current rhs) */
  double *V = (double*)solver->inner_mem;
  double *H = &V[m*(m_H+1)];
  double *br = &H[m_H*n_H];
  double *r = &br[m_H];
  double *xt = &r[m];
  double *refs_array = &xt[m];

  /* map handles to allocated memory */
  double *w = NULL;
  double *vprev = NULL;
  double *vcurr = NULL;

  double *b = (double*)b_dense->data;
  double *x = (double*)x_dense->data;

  /* unpacks memory_defl (deflation memory) */
  double *Vdefl = (double*)solver->defl_mem;
  double *Hdefl = &Vdefl[m*solver->iterations*solver->batch];
  double *refs_defl_array = &Hdefl[4*m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m*solver->batch];
  double *e_vecs = &Mdefl[m*solver->batch*solver->iterations];
  double *e_vals = &e_vecs[m*solver->iterations];

  /* @FIX: extra objects that require allocation */
  double *e = &e_vals[solver->iterations];
  double *d = &e[solver->iterations+n_defl];
  double *refl_ev = &d[solver->iterations+n_defl];
  double *evals = &refl_ev[solver->iterations];
  double *evecs = &evals[solver->iterations];
  double *G = &evecs[(n_defl+m_H)*n_defl];  /* to be assigned */
  double *F = &G[(solver->iterations+n_defl)*(solver->iterations+n_defl)];  /* to be assigned */
  double *Z = &F[(solver->iterations+n_defl)*(solver->iterations+n_defl)];

  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  if (solver->use_defl)
  {
    /*-----------------------------------------------------*/
    /* projects residual to approximate invariant subspace */
    /*-----------------------------------------------------*/
    //memcpy(Vdefl, V, (sizeof *Vdefl)*m*n_defl);
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vdefl, n_defl, m, 0.0, Z, m);
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_defl, m, 1.0,
      Vdefl, m, Z, m, 0.0, Hdefl, n_defl);

    mpf_matrix_d_announce(Hdefl, n_defl, n_defl, n_defl, "Hdefl (in 1)");

    /* computes r0 */
    memcpy(r, b, (sizeof *r)*m);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, x, 1.0, r);

    /* adds to x */
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_B, m, 1.0,
      Vdefl, m, r, m, 0.0, Tdefl, n_defl);
    mpf_qr_dsy_ref_givens(n_defl, n_defl, n_B, Hdefl, n_defl, Tdefl,
      refs_defl_array);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_defl, n_B, 1.0, Hdefl, n_defl, Tdefl, n_defl);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, 1, n_defl, 1.0,
      Vdefl, m, Tdefl, n_defl, 1.0, x, m);

    /* computes residual */
    memcpy(r, b, (sizeof *r)*m);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, x, 1.0, r);
    memcpy(V, r, (sizeof *V)*m);
    b_norm = mpf_dnrm2(m, b, 1); //@FIX: fix this to consider projector as well
    r_norm = mpf_dnrm2(m, r, 1);
    printf("[start] relative residual: %1.4E\n", r_norm/b_norm);

    /*------------------------*/
    /* low rank decomposition */
    /*------------------------*/
    /* outer-loop (restarts) */
    for (k = 0; k < outer_iterations; ++k)
    {
      mpf_dscal(m, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;
      /* update w */
      w = &V[m];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, V, n_B, m, 0.0, w, m);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_B, m, 1.0,
        Z, m, w, m, 0.0, Tdefl, n_defl);
      /* solves linear system */
      mpf_qr_dsy_rhs_givens(n_defl, n_defl, 1, Hdefl, n_defl, Tdefl,
        refs_defl_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        n_defl, n_B, 1.0, Hdefl, n_defl, Tdefl, n_defl);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n_B, n_defl, 1.0,
        Z, m, Tdefl, n_defl, 0.0, Mdefl, m);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m, 1.0, w, m);
      /* computes column of H */
      H[0] = mpf_ddot(m, w, 1, V, 1);
      mpf_daxpy(m, -H[0], V, 1, w, 1);
      h_temp = mpf_dnrm2(m, w, 1);
      if (h_temp < 1e-12)
      {
        inner_iterations = 1;
        break;
      }
      H[1] = h_temp;
      mpf_dscal(m, 1/H[1], w, 1);

      /* inner iterations */
      for (j = 1; j < inner_iterations; ++j)
      {
        w = &V[m*(j+1)];
        vcurr = &V[m*j];
        vprev = &V[m*(j-1)];
        /* update w */
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, vcurr, n_B, m, 0.0, w, m);
        mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, 1, m, 1.0,
          Z, m, w, m, 0.0, Tdefl, n_defl);
        mpf_qr_dsy_rhs_givens(n_defl, n_defl, 1, Hdefl, n_defl, Tdefl,
          refs_defl_array);
        mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
          CblasNonUnit, n_defl, 1, 1.0, Hdefl, n_defl, Tdefl, n_defl);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, 1, n_defl, 1.0,
          Z, m, Tdefl, n_defl, 0.0, Mdefl, m);
        mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
          SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m, 1.0, w, m);
        /* computes new column of H */
        H[m_H*j+j] = mpf_ddot(m, w, 1, vcurr, 1);
        mpf_daxpy(m, -H[m_H*j+j], vcurr, 1, w, 1);
        H[m_H*j+j-1] = H[m_H*(j-1)+j];
        mpf_daxpy(m, -H[m_H*j+j-1], vprev, 1, w, 1);
        h_temp = mpf_dnrm2(m, w, 1);
        if ((h_temp <= 1e-12) || (j == inner_iterations-1))
        {
          inner_iterations = j+1;
          m_H = inner_iterations;
          n_H = inner_iterations;
          break;
        }
        H[m_H*j+j+1] = h_temp;
        mpf_dscal(m, 1/H[m_H*j+j+1], w, 1);
      }

      /*--------------------------------------------------------------------*/
      /* solves linear system of equations and checks termination condition */
      /*--------------------------------------------------------------------*/
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);
      mpf_dgemv(CblasColMajor, CblasNoTrans, m, m_H, 1.0, V, m, br, 1, 1.0,
        x, 1);

      /*------------------------*/
      /* eigenvalue computation */
      /*------------------------*/
      /* @FIX: have to use A-UU' instead of A */
      //mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      //  SPARSE_LAYOUT_COLUMN_MAJOR, Vdefl, n_defl, m, 0.0, Z, m);
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, V, m_H, m, 0.0, &Z[m*n_defl], m);

//      mpf_matrix_d_announce(Z, 10, m_G, m, "Z (first)");
//
//      /* */
//      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, m_G, m,
//        1.0, Vdefl, m, Z, m, 0.0, Mdefl, n_defl);
//
//      mpf_matrix_d_announce(Hdefl, n_defl, n_defl, n_defl, "Hdefl");
//      mpf_matrix_d_announce(Mdefl, n_defl, m_G, n_defl, "Mdefl (0)");
//      /* solves */
//      mpf_qr_dsy_mrhs_givens(n_defl, n_defl, m_G, Hdefl, n_defl, Mdefl,
//        refs_defl_array);
//
//      mpf_matrix_d_announce(Hdefl, n_defl, n_defl, n_defl, "Hdefl");
//      mpf_matrix_d_announce(Mdefl, n_defl, m_G, n_defl, "Mdefl (first)");
//      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//        n_defl, m_G, 1.0, Hdefl, n_defl, Mdefl, n_defl);
//      mpf_matrix_d_announce(Mdefl, n_defl, m_G, n_defl, "Mdefl (inter)");
////mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n_defl, n_defl,
////  -1.0, Vdefl, m, Mdefl, n_defl, 1.0, Z, m);
//      double *Zdefl = mpf_malloc((sizeof *Zdefl)*m*n_defl);
//
//      memcpy(Zdefl, Z, (sizeof *Zdefl)*m*n_defl);
//      //this is the correct one
//      //mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m_G, n_defl,
//      //  -1.0, Zdefl, m, Mdefl, n_defl, 1.0, Z, m);
//      mpf_matrix_d_announce(Z, 10, n_defl, m, "Zdefl");
//      mpf_matrix_d_announce(Mdefl, n_defl, m_G, n_defl, "Mdefl");
//      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m_G, n_defl,
//        -1.0, Zdefl, m, Mdefl, n_defl, 0.0, Z, m);
//      mpf_matrix_d_announce(Z, 10, n_defl, m, "Ztest");
//      mpf_free(Zdefl);
//
//      //mpf_matrix_d_announce(Mdefl, n_defl, m_G, n_defl, "Mdefl (second)");
//      mpf_matrix_d_announce(Zdefl, 10, n_defl, m, "Zdefl");

      /* computes G */
      mpf_zeros_d_set(MPF_COL_MAJOR, m_G, m_G, G, m_G);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_G, m_G, m,
        1.0, Z, m, Z, m, 0.0, G, m_G);

      mpf_matrix_d_announce(G, m_G, m_G, m_G, "G");

      /* computes F */
      mpf_zeros_d_set(MPF_COL_MAJOR, m_G, m_G, F, m_G);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, m_G, m,
        1.0, Vdefl, m, Z, m, 0.0, F, m_G);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_G, m,
        1.0, V, m, Z, m, 0.0, &F[n_defl], m_G);

      printf("m_G: %d, m_s: %d\n", m_G, m_H);

      MPF_Int n_ev_found = 0;
      //MPF_Int n_blocks = 0;
      MPF_Int *iblock = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
      MPF_Int *isplit = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
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
      LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', m_G, d, e, 0.0, 0.0, 1, n_defl,
        &n_ev_found, evals, evecs, m_G, n_defl, issup, &tryrac);

      /* reconstruct Vdefl */
      memcpy(Mdefl, Vdefl, (sizeof *Mdefl)*m*n_defl);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n_defl, n_defl,
        1.0, Mdefl, m, evecs, m_G, 0.0, Vdefl, m);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n_defl, m_H,
        1.0, V, m, &evecs[n_defl], m_G, 1.0, Vdefl, m);

      printf("n_ev_found: %d\n", n_ev_found);
      mpf_matrix_d_announce(evals, n_defl, 1, n_defl, "evals");

      mpf_free(iblock);
      mpf_free(isplit);
      mpf_free(issup);

      /*-------------------*/
      /* computes residual */
      /*-------------------*/
      memcpy(r, b, (sizeof *r)*m);
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr, x,
        1.0, r);
      r_norm = mpf_dnrm2(m, r, 1);
      #if DEBUG == 1
        printf("[end] relative residual: %1.4E\n", r_norm/b_norm);
      #endif
      if ((r_norm / b_norm <= solver->tolerance) && (k == outer_iterations-1))
      {
        outer_iterations = k;
        break;
      }
      else
      {
        /* restart */
        memcpy(V, r, (sizeof *V)*m);
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }
  else
  {
    memcpy(r, b, (sizeof *b)*m);
    mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1, A->handle, A->descr, x, 1.0, r);
    b_norm = mpf_dnrm2(m, b, 1);
    r_norm = mpf_dnrm2(m, r, 1);
    memcpy(V, r, (sizeof *V)*m);

    /*-------------------------------------------*/
    /* low-rank decomposition (outer iterations) */
    /*-------------------------------------------*/
    for (k = 0; k < outer_iterations; ++k)
    {
      /* first iteration */
      mpf_dscal(m, 1/r_norm, V, 1);
      mpf_zeros_d_set(MPF_COL_MAJOR, m_H, 1, br, m_H);
      br[0] = r_norm;
      w = &V[m];
      mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle, A->descr, V,
        0.0, w);
      H[0] = mpf_ddot(m, w, 1, V, 1);
      mpf_daxpy(m, -H[0], V, 1, w, 1);
      h_temp = mpf_dnrm2(m, w, 1);
      if (h_temp < 1e-12)
      {
        inner_iterations = 1;
        break;
      }
      H[1] = h_temp;
      mpf_dscal(m, 1/H[1], w, 1);
      /* inner iterations */
      for (j = 1; j < inner_iterations; ++j)
      {
        w = &V[m*(j+1)];
        vcurr = &V[m*j];
        vprev = &V[m*(j-1)];
        mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, 1.0, A->handle,
          A->descr, vcurr, 0, w);
        H[m_H*j+j] = mpf_ddot(m, w, 1, vcurr, 1);
        mpf_daxpy(m, -H[m_H*j+j], vcurr, 1, w, 1);
        H[m_H*j+j-1] = H[m_H*(j-1)+j];
        mpf_daxpy(m, -H[m_H*j+j-1], vprev, 1, w, 1);
        h_temp = mpf_dnrm2(m, w, 1);
        if ((h_temp <= 1e-12) || (j == inner_iterations-1))
        {
          inner_iterations = j+1;
          m_H = inner_iterations;
          n_H = inner_iterations;
          break;
        }
        H[m_H*j+j+1] = h_temp;
        mpf_dscal(m, 1/H[m_H*j+j+1], w, 1);
      }

      /*------------------------*/
      /* eigenvalue computation */
      /*------------------------*/
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, V, m_H, m, 0.0, Z, m);
      /* computes G */
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_H, m,
        1.0, Z, m, Z, m, 0.0, G, m_H);
      /* computes F */
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_H, m,
        1.0, V, m, Z, m, 0.0, F, m_H);

      MPF_Int n_ev_found = 0;
      //MPF_Int n_blocks = 0;
      MPF_Int *iblock = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
      MPF_Int *isplit = (MPF_Int*)mpf_malloc((sizeof *iblock)*m_H);
      MPF_Int *issup = (MPF_Int*)mpf_malloc((sizeof *issup)*m_H*2);
      lapack_logical tryrac;

      /* computes cholesky factorization of matrix F */
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', m_H, F, m_H);

      /* reduces generalized eigenvalue problem to standard form */
      MPF_Int info = 0;
      info = LAPACKE_dsygst(LAPACK_COL_MAJOR, 1, 'U', m_H, G, m_H, F, m_H);
      printf("info: %d\n", info);

      /* converts to tridiagonal matrix */
      info = LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', m_H, G, m_H, d, e, refl_ev);
      printf("info: %d\n", info);
      mpf_matrix_d_announce(d, m_H, 1, m_H, "d");
      mpf_matrix_d_announce(e, m_H, 1, m_H, "e");

      /* compute eigenvalues and eigenvectors */
      //LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', m_H, d, e, 0.0, 0.0, 1, n_defl,  // this is used originally
      //  &n_ev_found, evals, Vdefl, m_H, n_defl, issup, &tryrac);
      LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', m_H, d, e, 0.0, 0.0, 1, n_defl,
        &n_ev_found, evals, evecs, m_H, n_defl, issup, &tryrac);

      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_defl, m_H,
        1.0, evecs, m_H, evecs, m_H, 0.0, Hdefl, n_defl);
      mpf_matrix_d_announce(Hdefl, n_defl, n_defl, n_defl, "(evecs^T*evecs) -> testing..");

      /* reconstruct Vdefl */
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n_defl, m_H,
        1.0, V, m, evecs, m_H, 0.0, Vdefl, m);

      double test = mpf_ddot(m, Vdefl, 1, Vdefl, 1);
      printf("test: %1.4E\n", test);

      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_defl, n_defl, m,
        1.0, Vdefl, m, Vdefl, m, 0.0, Hdefl, n_defl);

      mpf_matrix_d_announce(Hdefl, n_defl, n_defl, n_defl, "(Vdefl^T*Vdefl) -> testing 2..");

      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m_H, m_H, m,
        1.0, V, m, V, m, 0.0, Mdefl, m_H);
      mpf_matrix_d_announce(Mdefl, m_H, m_H, m_H, "(V^T*V) -> testing 3..");

      printf("n_ev_found: %d\n", n_ev_found);
      mpf_matrix_d_announce(evals, n_defl, 1, n_defl, "evals");
      mpf_matrix_d_announce(evecs, m_H, n_defl, m_H, "evecs (iter 0)");
      mpf_matrix_d_announce(Vdefl, 10, n_defl, m, "Vdefl (iter 0)");

      mpf_free(iblock);
      mpf_free(isplit);
      mpf_free(issup);

      /*--------------------------------------------------------------------*/
      /* solves linear system of equations and checks termination condition */
      /*--------------------------------------------------------------------*/
      mpf_qr_dsy_ref_givens(m_H, n_H, 1, H, ld_H, br, refs_array);
      mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m_H, 1, 1.0, H, n_H, br, m_H);
      mpf_dgemv(CblasColMajor, CblasNoTrans, m, m_H, 1.0, V, m, br, 1,
        1.0, x, 1);

      /*-------------------*/
      /* computes residual */
      /*-------------------*/
      memcpy(r, b, (sizeof *r)*m);
      mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr, x,
        1.0, r);
      r_norm = mpf_dnrm2(m, r, 1);
      #if DEBUG == 1
        printf("relative residual: %1.4E\n", r_norm/b_norm);
      #endif
      /* copy krylov subspace to memory_defl */
      if ((r_norm / b_norm <= solver->tolerance) || (k == outer_iterations-1))
      {
        outer_iterations = k;
        break;
      }
      else
      {
        /* restart */
        memcpy(V, r, (sizeof *V)*m);
        inner_iterations = solver->iterations;
        m_H = inner_iterations;
        n_H = inner_iterations;
      }
    }
  }

  //memcpy(Hdefl, H, (sizeof *Hdefl)*m_H*n_H);
  //memcpy(Vdefl, V, (sizeof *V)*m*inner_iterations);
  //memcpy(refs_defl_array, refs_array, (sizeof *refs_defl_array)
  //  *(inner_iterations-1));

  w = NULL;
  vprev = NULL;
  vcurr = NULL;
  V = NULL;
  H = NULL;
  br = NULL;
  r = NULL;

  Vdefl = NULL;
  Hdefl = NULL;
  Tdefl = NULL;
  Mdefl = NULL;

  e = NULL;
}

void mpf_defl_ev_lanczos_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts,
  MPF_Int n_ev_max
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;
  solver->defl_n_ev_max = n_ev_max;
  solver->n_defl = n_ev_max;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_LANCZOS;
    solver->inner_function = &mpf_dsy_defl_ev_lanczos;
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

  mpf_ev_defl_lanczos_get_mem_size(solver);

  //mpf_defl_memory_get(solver->data_type, solver->meta_solver.krylov,
  //  solver->m_A, solver->m_A, &solver->bytes_inner);
}

void mpf_ev_defl_lanczos_get_mem_size
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
         n*blk                     /* size_B */
        +n*blk                     /* size_X */
        +n*(iterations+1)*n_ev_max /* size_V */
        +iterations*iterations     /* size_H */
        +iterations                /* size_br */
        +n                         /* size_residual */
        +iterations                /* refs_array */
        +n_ev_max                  /* evals */
        +(n_ev_max+iterations)*n_ev_max/* evecs */
        +n-1                       /* off diagonal band of tridiagonal matrix */
        +(n_ev_max+iterations)*(n_ev_max+iterations)     /* matrix G */
        +(n_ev_max+iterations)*(n_ev_max+iterations)     /* F */
        +n*(n_ev_max+iterations)                         /* Z */
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
