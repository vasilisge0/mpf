#include "mpf.h"

void mpf_spbasis_blk_lanczos_init
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
    solver->inner_type = MPF_SOLVER_DSY_BLK_LANCZOS;
    solver->inner_function = &mpf_dsy_spbasis_blk_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_BLK_LANCZOS;
    solver->inner_function = &mpf_zsy_spbasis_blk_lanczos;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_BLK_LANCZOS;
    solver->inner_function = &mpf_zhe_spbasis_blk_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  mpf_blk_lanczos_get_mem_size(solver);
}

// @NOTE: 1 of the 2 methods, this tries the sparsified X between restarts,
// (nonlinear), try the non sparsified output also to compare.
void mpf_dsy_diag_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* solver->*/

  MPF_Int blk = solver->batch;
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1 + solver->restarts;

  MPF_Int m_B = B_dense->m;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;

  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* assign cpu memory to mathematical objects */
  double *V = (double*)solver->inner_mem;
  double *H = &V[m_B*(m_H+blk)];
  double *Br = &H[m_H*n_H];
  double *reflectors_array = &Br[m_H*blk];
  double *B_norms_array = &reflectors_array[blk];

  /* assign handles to cpu memory */
  double *V_first_vecblk = V;
  double *Vlast = &V[(m_B*blk)*inner_iterations];
  double *R = Vlast;
  double *W = NULL;
  double *Vprev = NULL;
  double *Hblk = NULL;
  double *Hblk_dest = NULL;

  /* required for sparsification */
  MPF_Int end_sparse = 0;
  MPF_Int nz_new = 0;

  /* */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dnrm2(m_B, &((double*)B)[m_B*i], 1);
  }
  memcpy(V_first_vecblk, B, (sizeof *V)*m_B*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m_B, 1.0, V, m_B);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    r_norm = mpf_dnrm2(m_B, &V[m_B*i], 1)/B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }
  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif
  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer iterations (restarts) */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, V, m_B, reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m_B, blk, V, m_B, Br, ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, V, m_B, reflectors_array);
    W = &V[(m_B * blk)];
    Vprev = V;
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, 0.0, W, m_B);

    Hblk = H;
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0, Vprev,
      m_B, W, m_B, 0.0, Hblk, ld_H);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
      Vprev, m_B, Hblk , ld_H, 1.0, W, m_B);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W , m_B, reflectors_array);

    Hblk = H + blk;
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
    H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);

    /* inner iterations */
    MPF_Int j = 0;
    for (j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk * j)];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, V, blk, m_B, 0.0, W, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
        Vprev, m_B, W, m_B, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V_first_vecblk[(m_B * blk)*(j-1)];
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W, m_B, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
      H_IJ_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk * (inner_iterations+1);
        n_H = blk * inner_iterations;
        break;
      }
      mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
      vecblk_d_sparsify(m_B,              /* length of input vector */
                        blk,
                        Vprev,            /* input vector */
                        &V[end_sparse],   /* output vector (compresed) */
                        solver->max_blk_fA,
                        solver->current_rhs,
                        &solver->color_to_node_map,
                        &nz_new);
      end_sparse += nz_new;    // better max error, this is the correct one
    }

    if (H_IJ_norm > 1e-12)
    {
      j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, 0.0, W, m_B);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, 1.0,
        Vprev, m_B, W, m_B, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = V_first_vecblk + (m_B * blk)*(j-1);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk, -1.0,
        Vprev, m_B, Hblk, ld_H, 1.0, W, m_B);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);

      vecblk_d_sparsify(
        m_B,              /* length of input vector */
        blk,
        Vprev,            /* input vector */
        &V[end_sparse],   /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_dsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, 1.0, H, ld_H, Br, ld_H);

    //@NOTE: Beware as it produces an entry-wise compressed vector block,
    //       and as thus not the same dense X that is produced from regular
    //       block_lanczos. 
    //       Since B is also sparsified probably result will be better
    //       have to sparsify X first.
    block_krylov_dge_sparse_basis_combine(
      solver->current_rhs,
      n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m_B);

    memcpy(R, B, (sizeof *R)*m_B*blk);
    mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, m_B, 1.0, R, m_B);

    r_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dnrm2(m_B, &R[m_B*i], 1) / B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG
      printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_max, k);
    #endif

    if (r_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m_B*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

// @NOTE: 1 of the 2 methods, this tries the sparsified X between restarts,
// (nonlinear), try the non sparsified output also to compare.
void mpf_dsy_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  /* context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* assign cpu memory to mathematical objects */
  double *V = (double *) solver->inner_mem;
  double *H = &V[m*(m_H+blk)];
  double *Br = &H[m_H*n_H];
  double *reflectors_array = &Br[m_H*blk];
  double *B_norms_array = &reflectors_array[blk];

  /* assign handles to cpu memory */
  double *V_first_vecblk = V;
  double *Vlast = &V[(m*blk)*solver->iterations];
  double *R = Vlast;
  double *W = NULL;
  double *Vprev = NULL;
  double *Hblk = NULL;
  double *Hblk_dest = NULL;

  /* required for sparsification */
  MPF_Int end_sparse = 0;
  MPF_Int nz_new = 0;

  /* */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_d_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dnrm2(m, &((double*)B)[m*i], 1);
  }
  memcpy(V, B, (sizeof *V)*m*blk);
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, 1.0, V, m);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    r_norm = mpf_dnrm2(m, &V[m*i], 1) / B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }

  #if DEBUG
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif

  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_d_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, V_first_vecblk, m,
      reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m, blk, V_first_vecblk, m, Br,
      ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, V_first_vecblk, m,
      reflectors_array);
    W = &V[(m * blk)];
    Vprev = V;
    mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, 0.0, W, m);

    Hblk = H;
    mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, 1.0,
      Vprev, m, W, m, 0.0, Hblk, ld_H);
    mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk, -1.0,
      Vprev, m, Hblk , ld_H, 1.0, W, m);
    mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, W , m, reflectors_array);
    Hblk = &H[blk];
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m, Hblk, ld_H);
    mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, W, m, reflectors_array);
    H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);

    /* inner iterations */
    MPF_Int j = 0;
    for (j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m*blk)*(j+1)];
      Vprev = &V[(m*blk*j)];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        sparse_layout, Vprev, blk, m, 0.0, W, m);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, 1.0,
        Vprev, m, W, m, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk, -1.0,
        Vprev, m, Hblk, ld_H, 1.0, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, blk, blk, -1.0,
        Vprev, m, Hblk, ld_H, 1.0, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, W, m, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m, Hblk, ld_H);

      H_IJ_norm = mpf_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk * (inner_iterations+1);
        n_H = blk * inner_iterations;
        break;
      }
      mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, W, m, reflectors_array);
      vecblk_d_block_sparsify(
        m,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new
      );

      end_sparse += nz_new; // better max error, this is the correct one
    }

    if (H_IJ_norm > 1e-12)
    {
      j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, 0.0, W, m);
      mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m, 1.0,
        Vprev, m, W, m, 0.0, Hblk, ld_H);
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk, -1.0,
        Vprev, m, Hblk, ld_H, 1.0, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, blk, blk, -1.0,
        Vprev, m, Hblk, ld_H, 1.0, W, m);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);
      vecblk_d_block_sparsify(
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

    //vecblk_d_sparsify(m,             /* length of input vector */
    //                  blk,
    //                  W,            /* input vector */
    //                  &V[end_sparse],    /* output vector (compresed) */
    //                  blk_max_fA,
    //                  current_rhs,
    //                  &context->probing_color_to_node_map,
    //                  &num_new_nonzeros);

    //end_sparse += num_new_nonzeros;    // better max error, this is the correct one


    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_dsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, 1.0, H, ld_H, Br, ld_H);

    //@NOTE: Beware as it produces an entry-wise compressed vector block, and as thus not the same dense X
    //that is produced from regular block_lanczos.
    // Since B is also sparsified probably result will be better have to sparsify X first.
    block_krylov_dge_sparse_basis_block_combine(
      solver->current_rhs,
      n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m);

    memcpy(R, B, (sizeof *R) * m * blk);
    mpf_sparse_d_mm(MPF_SPARSE_NON_TRANSPOSE, -1.0, A->handle, A->descr,
      sparse_layout, X, blk, m, 1.0, R, m);

    r_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dnrm2(m, &R[m*i], 1)/B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG
      printf("max_relative residual: %1.4E -- (restart %d)\n",
        r_norm_max, k);
    #endif

    if (r_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

// @NOTE: 1 of the 2 methods, this tries the sparsified X between restarts,
// (nonlinear), try the non sparsified output also to compare.
void mpf_zhe_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  /* context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  MPF_ComplexDouble* B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble* X = (MPF_ComplexDouble*)X_dense->data;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  /* assign cpu memory to mathematical objects */
  MPF_ComplexDouble *V          = (MPF_ComplexDouble *) solver->inner_mem;
  MPF_ComplexDouble *H          = &V[m*(m_H+blk)];
  MPF_ComplexDouble *Br         = &H[m_H*n_H];
  MPF_ComplexDouble *reflectors_array = &Br[m_H*blk];
  double *B_norms_array    = (double*)&reflectors_array[blk];

  /* assign handles to cpu memory */
  MPF_ComplexDouble *V_first_vecblk = V;
  MPF_ComplexDouble *Vlast = &V[(m*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Hblk_dest = NULL;

  /* required for sparsification */
  MPF_Int end_sparse = 0;
  MPF_Int nz_new = 0;

  /* */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dznrm2(m, &B[m*i], 1);
  }
  memcpy(V, B, (sizeof *V)*m*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m, ONE_C, V, m);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    r_norm = mpf_dznrm2(m, &V[m*i], 1) / B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif

  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    //mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, V_first_vecblk, m, reflectors_array);
    mpf_gram_schmidt_zge(m, blk, V_first_vecblk, Br, m_H);
    //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m, blk, V_first_vecblk, m, Br,
      //ld_H);
    //mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, V_first_vecblk, m, reflectors_array);

    W = &V[(m * blk)];
    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W, m);

    Hblk = H;
    mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m, &ONE_C,
      Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
      &MINUS_ONE_C, Vprev, m, Hblk , ld_H, &ONE_C, W, m);

    Hblk = &H[blk];
    mpf_gram_schmidt_zge(m, blk, W, Hblk, ld_H);
    //mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, W , m, reflectors_array);
    //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m, Hblk, ld_H);
    //mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, W, m, reflectors_array);
    //H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
    H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);

    /* inner iterations */
    for (MPF_Int j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m*blk)*(j+1)];
      Vprev = &V[(m*blk*j)];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m, ZERO_C, W, m);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &MINUS_ONE_C, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H*blk)*(j-1) + blk*j];
      Vprev = &V[(m*blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, blk, blk,
        &MINUS_ONE_C, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_gram_schmidt_zge(m, blk, W, Hblk, ld_H);
      //mpf_dgeqrf(LAPACK_COL_MAJOR, m, blk, W, m, reflectors_array);
      //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m, Hblk, ld_H);

      H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk * (inner_iterations+1);
        n_H = blk * inner_iterations;
        break;
      }

      //mpf_dorgqr(LAPACK_COL_MAJOR, m, blk, blk, W, m, reflectors_array);
      vecblk_z_block_sparsify
      (
        m,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new
      );

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    if (H_IJ_norm > 1e-12)
    {
      MPF_Int j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m * blk)*(j+1)];
      Vprev = &V[(m * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m, ZERO_C, W, m);
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, blk, blk, m,
        &ONE_C, Vprev, m, W, m, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blk, blk,
        &MINUS_ONE_C, Vprev, m, Hblk, ld_H, &ONE_C, W, m);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, blk, blk,
        &MINUS_ONE_C, Vprev, m, Hblk, ld_H, &ONE_C, W, m);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy ('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest, ld_H);
      vecblk_z_block_sparsify
      (
        m,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new
      );

      end_sparse += nz_new; // better max error, this is the correct one
    }

    //vecblk_d_sparsify(m,               /* length of input vector */
    //                  blk,
    //                  W,                 /* input vector */
    //                  &V[end_sparse],    /* output vector (compresed) */
    //                  solver->max_blk_fA_fA,
    //                  current_rhs,
    //                  &context->probing_color_to_node_map,
    //                  &num_new_nonzeros);

    //end_sparse += num_new_nonzeros;    // better max error, this is the correct one


    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_zsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, &ONE_C, H, ld_H, Br, ld_H);

    //@NOTE: Beware as it produces an entry-wise compressed vector block, and
    //as thus not the same dense X that is produced from regular block_lanczos.
    // Since B is also sparsified probably result will be better have to
    //sparsify X first.
    block_krylov_zge_sparse_basis_block_combine
    (
      solver->current_rhs,
      n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m
    );

    memcpy(R, B, (sizeof *R) * m * blk);
    mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
      sparse_layout, X, blk, m, ONE_C, R, m);

    r_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dznrm2(m, &R[m*i], 1)/B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n",
        r_norm_max, k);
    #endif

    if (r_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

// @NOTE: 1 of the 2 methods, this tries the sparsified X between restarts,
// (nonlinear), try the non sparsified output also to compare.
void mpf_zsy_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble MINUS_ONE_C = mpf_scalar_z_init(-1.0, 0.0);

  /* context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* solver->*/
  MPF_Int inner_iterations = solver->iterations;
  MPF_Int outer_iterations = 1+solver->restarts;
  MPF_Int m_B = solver->ld;
  MPF_Int blk = solver->batch;
  MPF_Int m_H = solver->iterations*blk;
  MPF_Int n_H = solver->iterations*blk;
  MPF_Int ld_H = m_H;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  MPF_ComplexDouble *B = (MPF_ComplexDouble*)B_dense->data;
  MPF_ComplexDouble *X = (MPF_ComplexDouble*)X_dense->data;

  /* assign cpu memory to mathematical objects */
  MPF_ComplexDouble *V  = (MPF_ComplexDouble *) solver->inner_mem;
  MPF_ComplexDouble *H  = &V[m_B*(m_H+blk)];
  MPF_ComplexDouble *Br = &H[m_H*n_H];
  MPF_ComplexDouble *reflectors_array = &Br[m_H*blk];
  double *B_norms_array = (double*)&reflectors_array[blk];
  /* assign handles to cpu memory */
  MPF_ComplexDouble *V_first_vecblk = V;
  MPF_ComplexDouble *Vlast = &V[(m_B*blk)*solver->iterations];
  MPF_ComplexDouble *R = Vlast;
  MPF_ComplexDouble *W = NULL;
  MPF_ComplexDouble *Vprev = NULL;
  MPF_ComplexDouble *Hblk = NULL;
  MPF_ComplexDouble *Hblk_dest = NULL;

  /* required for sparsification */
  MPF_Int end_sparse = 0;
  MPF_Int nz_new = 0;

  /* */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  mpf_zeros_z_set(MPF_COL_MAJOR, m_H, n_H, H, m_H);
  for (MPF_Int i = 0; i < blk; ++i)
  {
    B_norms_array[i] = mpf_dznrm2(m_B, &B[m_B*i], 1);
  }
  memcpy(V, B, (sizeof *V)*m_B*blk);
  mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, m_B, ONE_C, V, m_B);

  for (MPF_Int i = 0; i < blk; ++i)
  {
    r_norm = mpf_dznrm2(m_B, &V[m_B*i], 1) / B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }

  #if DEBUG == 1
    printf("max relative residual: %1.4E \n", r_norm_max);
  #endif

  if (r_norm_max <= solver->tolerance)
  {
    return;
  }

  /* outer iterations */
  for (MPF_Int k = 0; k < outer_iterations; ++k)
  {
    mpf_zeros_z_set(MPF_COL_MAJOR, m_H, blk, Br, m_H);
    //mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, V_first_vecblk, m_B, reflectors_array);
    mpf_gram_schmidt_zge(m_B, blk, V_first_vecblk, Br, m_H);
    //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m_B, blk, V_first_vecblk, m_B, Br, ld_H);
    //mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, V_first_vecblk, m_B, reflectors_array);

    W = &V[(m_B*blk)];
    Vprev = V;
    mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
      SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, ZERO_C, W, m_B);

    Hblk = H;
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
      Vprev, m_B, W, m_B, &ZERO_C, Hblk, ld_H);
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk,
      &MINUS_ONE_C, Vprev, m_B, Hblk , ld_H, &ONE_C, W, m_B);

    Hblk = &H[blk];
    mpf_gram_schmidt_zge(m_B, blk, W, Hblk, ld_H);
    //mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W , m_B, reflectors_array);
    //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
    //mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
    //H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
    H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);

    /* inner iterations */
    for (MPF_Int j = 1; j < inner_iterations-1; ++j)
    {
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m_B*blk)*(j+1)];
      Vprev = &V[(m_B*blk*j)];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        sparse_layout, Vprev, blk, m_B, ZERO_C, W, m_B);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
        Vprev, m_B, W, m_B, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk,
        &MINUS_ONE_C, Vprev, m_B, Hblk, ld_H, &ONE_C, W, m_B);

      Hblk = &H[(ld_H*blk)*(j-1) + blk*j];
      Vprev = &V[(m_B*blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk,
        &MINUS_ONE_C, Vprev, m_B, Hblk, ld_H, &ONE_C, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[(ld_H * blk)*j + blk*(j+1)];
      mpf_gram_schmidt_zge(m_B, blk, W, Hblk, ld_H);
      //mpf_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, W, m_B, reflectors_array);
      //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, W, m_B, Hblk, ld_H);
      H_IJ_norm = mpf_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk * (inner_iterations+1);
        n_H = blk * inner_iterations;
        break;
      }
      //mpf_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, W, m_B, reflectors_array);
      vecblk_z_block_sparsify(
        m_B,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    if (H_IJ_norm > 1e-12)
    {
      MPF_Int j = inner_iterations-1;
      Hblk = &H[(ld_H * blk)*j + blk*j];
      W = &V[(m_B * blk)*(j+1)];
      Vprev = &V[(m_B * blk)*j];
      mpf_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A->handle, A->descr,
        SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, m_B, ZERO_C, W, m_B);
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, m_B, &ONE_C,
        Vprev, m_B, W, m_B, &ZERO_C, Hblk, ld_H);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_B, blk, blk,
        &MINUS_ONE_C, Vprev, m_B, Hblk, ld_H, &ONE_C, W, m_B);

      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Vprev = &V[(m_B * blk)*(j-1)];
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m_B, blk, blk,
        &MINUS_ONE_C, Vprev, m_B, Hblk, ld_H, &ONE_C, W, m_B);
      Hblk = &H[(ld_H * blk)*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mpf_zomatcopy ('C', 'T', blk, blk, ONE_C, Hblk, ld_H, Hblk_dest, ld_H);
      vecblk_z_block_sparsify(
        m_B,             /* length of input vector */
        blk,
        Vprev,           /* input vector */
        &V[end_sparse],  /* output vector (compresed) */
        solver->max_blk_fA,
        solver->current_rhs,
        &solver->color_to_node_map,
        &nz_new);

      end_sparse += nz_new;    // better max error, this is the correct one
    }

    //vecblk_d_sparsify(m_B,             /* length of input vector */
    //                  blk,
    //                  W,            /* input vector */
    //                  &V[end_sparse],    /* output vector (compresed) */
    //                  solver->max_blk_fA_fA,
    //                  current_rhs,
    //                  &context->probing_color_to_node_map,
    //                  &num_new_nonzeros);

    //end_sparse += num_new_nonzeros;    // better max error, this is the correct one


    /* solves system of equations and evaluates termination criteria */
    mpf_block_qr_zsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mpf_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, &ONE_C, H, ld_H, Br, ld_H);

    //@NOTE: Beware as it produces an entry-wise compressed vector block, and
    // as thus not the same dense X that is produced from regular block_lanczos.
    // Since B is also sparsified probably result will be better have to
    //sparsify X first.

    block_krylov_zge_sparse_basis_block_combine
    (
      solver->current_rhs,
      n_H,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      m_B
    );

    memcpy(R, B, (sizeof *R) * m_B * blk);
    mpf_sparse_z_mm(MPF_SPARSE_NON_TRANSPOSE, MINUS_ONE_C, A->handle, A->descr,
      sparse_layout, X, blk, m_B, ONE_C, R, m_B);

    r_norm_max = 0.0;
    for (MPF_Int i = 0; i < blk; ++i)
    {
      r_norm = mpf_dznrm2(m_B, &R[m_B*i], 1)/B_norms_array[i];
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }

    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n",
        r_norm_max, k);
    #endif

    if (r_norm_max <= solver->tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *V)*m_B*blk);
    }
  }

  Hblk = NULL;
  Hblk_dest = NULL;
  Vprev = NULL;
  W = NULL;
  R = NULL;
}
