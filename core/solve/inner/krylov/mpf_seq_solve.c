#include "mpf.h"

void mpf_seq_gbl_lanczos_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = restarts;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_SPBASIS_GBL_LANCZOS;
    solver->inner_function = &mpf_dsy_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_SPBASIS_GBL_LANCZOS;
    solver->inner_function = &mpf_zsy_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX)
          && (solver->matrix_type == MPF_MATRIX_HERMITIAN)
  )
  {
    solver->inner_type = MPF_SOLVER_ZHE_SPBASIS_GBL_LANCZOS;
    solver->inner_function = &mpf_zhe_spbasis_gbl_lanczos;
    solver->device = MPF_DEVICE_CPU;
  }

  mpf_gbl_lanczos_get_mem_size(solver);
}

void mpf_seq_solve
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B_dense,
  MPF_Dense *X_dense
)
{
  MPF_Int n = solver->ld;
  MPF_Int m = A->m;
  MPF_Int m_B = A->m;
  MPF_Int ld_H = 0;
  MPF_Int m_H = 0;
  MPF_Int n_H = 0;
  MPF_Int blk = solver->batch;

  double *B = (double*)B_dense->data;
  double *X = (double*)X_dense->data;

  /* assign cpu memory to mathematical objects */
  double *V = (double*)solver->inner_mem;
  double *H = &V[m_B*blk*(m_H+1)];
  double *Hblk = &H[m_H*n_H];
  double *Br = &Hblk[blk*blk];

  /* computes sparse basis */
  mpf_dsy_spbasis_gbl_lanczos(solver, A, B_dense, X_dense);

  /* */
  m = solver->ld;
  ld_H = m;
  m_H = m;
  n_H = m;

  for (MPF_Int i = 1; i < solver->n_shifts; ++i)
  {
    /* update diagonal of H */
    for (MPF_Int j = 0; j < m; ++j)
    {
      H[m*j+j] += solver->shifts_array[i];
    }

    /* triangular solve */
    mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, Br, ld_H);

    /* combine basis */
    global_krylov_dge_sparse_basis_block_combine
    (
      solver->current_rhs,
      blk*m,
      blk,
      V,
      &solver->color_to_node_map,
      solver->max_blk_fA,
      ld_H,
      Br,
      X,
      n
    );
  }

}

