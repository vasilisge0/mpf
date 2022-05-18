#include "mpf.h"

void mpf_spbasis_defl_cg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int n_defl
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;
  solver->n_defl = n_defl;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_CG;
    solver->inner_function = &mpf_dsy_cg;
    solver->device = MPF_DEVICE_CPU;
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

  mpf_dsy_spbasis_defl_cg_get_mem_size(solver);
}

void mpf_dsy_spbasis_defl_cg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int blk = solver->batch;
  MPF_Int n = solver->ld;
  MPF_Int nz_max = 1;
  MPF_Int num_deflation_vecblks = 1;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double) *
        (n*blk*2  /* size_B and size_X */
        +n*2      /* size_residuals */
        +n*2      /* size_direction */
        +nz_max*num_deflation_vecblks);
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble) *
        (n*blk*2 /* size_B and size_X */
        +n*2     /* size_residuals */
        +n*2     /* size_direction */
        +nz_max*num_deflation_vecblks);

  }
  else if (solver->data_type == MPF_COMPLEX_32)
  {
    solver->inner_bytes = sizeof(MPF_Complex) *
        (n*blk*2 /* size_B and size_X */
        +n*2     /* size_residuals */
        +n*2     /* size_direction */
        +nz_max*num_deflation_vecblks);
  }
}

//void mp_dsy_deflated_sparsified_cg
//(
//  /* solver parameters */
//  const KrylovMeta meta,
//
//  /* data */
//  const MPSparseDescr A_descr,
//  const MPMatrixSparseHandle A_handle,
//  const MPInt n,
//  const double *b,
//  double *x,
//  double *memory,
//  const MPInt current_rhs,
//  const MPInt blk_max,
//  MPBucketArray *color_to_node_map,
//
//  /* collected metadata */
//  MPSolverInfo *info
//)
//{
//  /* solver context */
//  double norm_b = 0.0;
//  double r_norm = 0.0;
//  double alpha = 0.0;
//  double beta  = 0.0;
//  MPInt i = 0;
//  MPInt m_B = n;
//  double *temp_vector = NULL;
//  MPLayout layout = MP_COL_MAJOR;
//  MPLayoutSparse sparse_layout;
//  /* memory cpu */
//  double *r_new = memory;
//  double *r_old = &r_new[m_B];
//  double *dvec = &r_old[m_B];
//
//  /* first iteration */
//  mp_convert_layout_to_sparse(layout, &sparse_layout);
//  memcpy(r_old, b, (sizeof *r_old)*m_B);
//  norm_b = cblas_dnrm2(m_B, b, 1);
//  mp_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A_handle, A_descr, x,
//    1.0, r_old);
//  r_norm = mp_dnrm2(m_B, r_old, 1);
//  memcpy(dvec, r_old, (sizeof *dvec)*m_B);
//  #if STATUS == MP_DEBUG
//    printf("relative residual: %1.4E\n", r_norm/norm_b);
//  #endif
//  r_norm = mp_dnrm2(m_B, b, 1);
//
//  /* main loop (iterations) */
//  while ((i < meta.iterations) && (r_norm/norm_b > meta.tolerance))
//  {
//    mp_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_handle, A_descr, dvec,
//      0.0, r_new);
//
//    alpha = mp_ddot(m_B, r_new, 1, dvec, 1);
//    alpha = mp_ddot(m_B, r_old, 1, r_old, 1)/alpha;
//    mp_daxpy(m_B, alpha, dvec, 1, x, 1);
//    mp_dscal(m_B, -alpha, r_new, 1);
//    mp_daxpy(m_B, 1.0, r_old, 1, r_new, 1);
//
//    beta = mp_ddot(m_B, r_new, 1, r_new, 1);
//    beta = beta/mp_ddot(m_B, r_old, 1, r_old, 1);
//    mp_dscal(m_B, beta, dvec, 1);
//    mp_daxpy(m_B, 1.0, r_new, 1, dvec, 1);
//    r_norm = mp_dnrm2(m_B, r_new, 1);
//
//    temp_vector = r_old;
//    r_old = r_new;
//    r_new = temp_vector;
//    i = i + 1;
//  }
//  #if STATUS == MP_DEBUG
//    printf("relative residual: %1.4E\n", r_norm / norm_b);
//    printf("iterations completed: %d\n", i);
//  #endif
//
//  r_new = NULL;
//  r_old = NULL;
//  temp_vector = NULL;
//}
//
//void mp_dsy_sparsified_cg
//(
//  /* solver parameters */
//  const KrylovMeta meta,
//
//  /* data */
//  const MPSparseDescr A_descr,
//  const MPSparseHandle A_handle,
//  const MPInt n,
//  const double *b,
//  double *x,
//  double *memory,
//  const MPInt current_rhs,
//  const MPInt blk_max,
//  MPBucketArray *color_to_node_map,
//
//  /* collected metadata */
//  MPSolverInfo *info
//)
//{
//  /* solver context */
//  double norm_b = 0.0;
//  double r_norm = 0.0;
//  double alpha = 0.0;
//  double beta  = 0.0;
//  MPInt i = 0;
//  MPInt m_B = n;
//  double *temp_vector = NULL;
//  MPLayout layout = MP_COL_MAJOR;
//  MPLayoutSparse sparse_layout;
//  /* memory cpu */
//  double *r_new = memory;
//  double *r_old = &r_new[m_B];
//  double *dvec = &r_old[m_B];
//  /* first iteration */
//  mp_convert_layout_to_sparse(layout, &sparse_layout);
//  memcpy(r_old, b, (sizeof *r_old)*m_B);
//  norm_b = cblas_dnrm2(m_B, b, 1);
//  mp_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A_handle,
//    A_descr, x, 1.0, r_old);
//  r_norm = mp_dnrm2(m_B, r_old, 1);
//  memcpy(dvec, r_old, (sizeof *dvec)*m_B);
//  #if STATUS == MP_DEBUG
//    printf("relative residual: %1.4E\n", r_norm / norm_b);
//  #endif
//  r_norm = mp_dnrm2(m_B, b, 1);
//
//  /* main loop (iterations) */
//  while ((i < meta.iterations) && (r_norm/norm_b > meta.tolerance))
//  {
//    mp_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_handle, A_descr, dvec,
//      0.0, r_new);
//    alpha = mp_ddot(m_B, r_new, 1, dvec, 1);
//    alpha = mp_ddot(m_B, r_old, 1, r_old, 1)/alpha;
//    mp_daxpy(m_B, alpha, dvec, 1, x, 1);
//
//    mp_dscal(m_B, -alpha, r_new, 1);
//    mp_daxpy(m_B, 1.0, r_old, 1, r_new, 1);
//    beta = mp_ddot(m_B, r_new, 1, r_new, 1);
//    beta = beta/mp_ddot(m_B, r_old, 1, r_old, 1);
//    mp_dscal(m_B, beta, dvec, 1);
//    mp_daxpy(m_B, 1.0, r_new, 1, dvec, 1);
//
//    r_norm = mp_dnrm2(m_B, r_new, 1);
//    temp_vector = r_old;
//    r_old = r_new;
//    r_new = temp_vector;
//    i = i + 1;
//  }
//
//  r_new = NULL;
//  r_old = NULL;
//  temp_vector = NULL;
//}
