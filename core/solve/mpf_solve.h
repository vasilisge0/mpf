#ifndef MPF_SOLVE_H
#define MPF_SOLVE_H

#define MPF_PRINTOUT_SOLVER 0

#include "mpf_types.h"
#include "mpf_blas_mkl_internal.h"

/* ------------------------------ gmres ------------------------------------- */

void mpf_dge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_sge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B, /* (input) rhs blk of vectors */
  MPF_Dense *X  /* (output) solution of AX = B */
);

void mpf_sge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dge_gbl_gmres
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_sge_gbl_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

/* ------------------------- complexsym functions --------------------------- */

void mpf_zsy_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zge_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_cpy_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zsy_blk_gmres
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zge_blk_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_csy_blk_gmres
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zsy_gbl_gmres
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zge_gbl_gmres
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_csy_gbl_gmres
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

/* --------------------------- Lanczos solvers ------------------------------ */

void mpf_dsy_lanczos
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dsy_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dsy_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zsy_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zhe_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zsy_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zhe_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zsy_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

/* --------------------------- spbasis lanczos ------------------------------ */

void mpf_dsy_spbasis_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dsy_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_dsy_diag_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zsy_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zhe_spbasis_blk_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_dsy_spbasis_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zsy_spbasis_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);


void mpf_zhe_spbasis_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zsy_spbasis_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zhe_spbasis_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_gmres_get_mem_size
(
  MPF_Solver *context
);

void mpf_cheb_get_mem
(
  MPF_Solver *solver
);

void mpf_blk_gmres_get_mem_size
(
  MPF_Solver *context
);

void mpf_gbl_gmres_get_mem_size
(
  MPF_Solver *context
);

/* -------------------  memory allocation functions ------------------------- */

void mpf_lanczos_get_mem_size
(
  MPF_Solver *context
);

void mpf_blk_lanczos_get_mem_size
(
  MPF_Solver *context
);

void mpf_gbl_lanczos_get_mem_size
(
  MPF_Solver *context
);


/* ----------------------------- CG contexts -------------------------------- */

void mpf_dsy_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_dsy_blk_cg
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zsy_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_zsy_blk_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_zhe_blk_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

//void mpf_zsy_gbl_cg
//(
//  /* context parameters */
//  const KrylovMeta meta,
//
//  /* data */
//  const MPF_SparseDescr A_descr,
//  const MPF_SparseHandle A_handle,
//  const MPF_Int n,
//  const MPF_ComplexDouble *b,
//  MPF_ComplexDouble *x,
//  MPF_ComplexDouble *memory,
//
//  /* collected metadata */
//  MPF_SolverInfo *info
//);


//void mpf_zhe_gbl_cg
//(
//  /* context parameters */
//  const KrylovMeta meta,
//
//  /* data */
//  const MPF_SparseDescr A_descr,
//  const MPF_SparseHandle A_handle,
//  const MPF_Int n,
//  const MPF_ComplexDouble *B,
//  MPF_ComplexDouble *X,
//  void *memory,
//
//  /* collected metadata */
//  MPF_SolverInfo *info
//);

/* --------------------- CG memory allocation functions ----------------------*/

void mpf_cg_get_mem_size
(
  MPF_Solver *solver
);

void mpf_dsy_defl_spbasis_cg_get_mem_size
(
  MPF_Solver *context
);

void mpf_blk_cg_get_mem_size
(
  MPF_Solver *context
);

void mpf_gbl_cg_get_mem_size
(
  MPF_Solver *context
);

/* ----------------------------- spbasis CG ----------------------------------*/

void mpf_dsy_defl_spbasis_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);


/* ------------------ CG contexts for hermitian matrices -------------------- */

void mpf_zhe_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

/*===================*/
/*== outer contexts ==*/
/*===================*/

void mpf_batch_solve
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Sparse *A,
  void *fA_out
);

void mpf_batch_gko_solve
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Args *args,
  void *fA_out
);

  //void mpf_batch_2pass_solve
  //(
  //  MPF_Solver *solver,
  //  MPF_Sparse *A,
  //  MPF_Dense *b,
  //  MPF_Dense *x
  //);

/* ------------------------ Least squares contexts -------------------------- */

void mpf_ls_horner
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_ls_horner_diag_blks
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_blk_ls_horner
(
  MPF_Solver *context
);

void mpf_gbl_ls_horner
(
  MPF_Solver *context
);

void mpf_batch_d_dynamic_spbasis
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *diag_fA
);

void mpf_dsy_spbasis_cg
(
  MPF_Solver *context,

  /* data */
  const MPF_Sparse *A,
  const MPF_Int n,
  const double *b,
  double *x
);

void mpf_dsy_defl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);


void mpf_batch_defl_solve
(
  MPF_Solver *context
);

void mpf_defl_lanczos_get_mem_size
(
  MPF_Solver *context
);

void mpf_dsy_defl_lanczos_2
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_inverse_context_partial_reconstruct
(
  MPF_Solver *context,
  MPF_Int current_rhs,
  MPF_Int offset,
  MPF_BucketArray *H
);

void mpf_inverse_partial_reconstruct
(
  MPF_Solver *context,
  MPF_Int current_rhs,
  MPF_Int offset,
  MPF_BucketArray *H
);

void mpf_batch_matrix_solve
(
  MPF_Solver *context
);

void mpf_inverse_partial_init
(
  MPF_Solver *context
);

void mpf_inverse_xy_partial_reconstruct
(
  MPF_Solver *context,
  MPF_Int current_rhs,
  MPF_Int offset,
  MPF_BucketArray *H
);

void mpf_inverse_xy_partial_reconstruct_2
(
  MPF_Solver *context,
  MPF_Int current_rhs,
  MPF_Int offset,
  MPF_BucketArray *H
);

void mpf_inverse_xy_partial_init
(
  MPF_Solver *context
);

void mpf_inverse_xy_partial_init_2
(
  MPF_Solver *context
);

void mpf_inverse_xy_partial_reconstruct_3
(
  MPF_Solver *context
);

void mpf_batch_cheb
(
  MPF_Solver *context
);

void mpf_dsy_defl_ev_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_defl_ev_lanczos_memory_get
(
  MPF_Solver *context
);

void mpf_dsy_defl_ev_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_ev_defl_cg_get_mem_size
(
  MPF_Solver *solver
);

void mpf_inverse_xy_rel_partial_init
(
  MPF_Solver *context
);

void mpf_inverse_xy_partial_rel_reconstruct
(
  MPF_Solver *context,
  MPF_Int curr_rhs,
  MPF_Int offset,
  MPF_BucketArray *H
);

void mpf_d_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  double *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
);

void mpf_zhe_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  MPF_ComplexDouble *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
);

void mpf_zsy_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  MPF_ComplexDouble *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
);

void mpf_inverse_partial_reconstruct_adapt
(
  MPF_Solver *context,
  MPF_Int curr_rhs,
  MPF_Int offset,
  MPF_BucketArray *H,
  MPF_Sparse *P,
  MPF_Int *nnz,
  MPF_Int *buffer,
  MPF_Int *buffer_rev
);

void mpf_inverse_xy_adapt_partial_init
(
  MPF_Solver *context,
  MPF_Int *buffer,
  MPF_Int *buffer_rev
);

void mpf_solve_gbl_pcg_dsy
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_gbl_pcg_get_mem_size
(
  MPF_Solver *context
);

void mpf_pthread_batch_solve
(
  MPF_Solver *context
);

void *mpf_pthread_kernel_d_solve
(
  void *input_packed
);

void mpf_dsy_sparse_cg
(
  /* context parameters */
  MPF_Solver *context,

  /* input matrix A */
  const MPF_Sparse *A,

  /* input matrix B */
  const MPF_Sparse *B,
  MPF_SparseCsr *X
);

void mpf_d_sparse_solve
(
  MPF_Solver *context
);

void mpf_pcg_get_mem_size
(
  MPF_Solver *context
);

void mpf_dsy_pcg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b,
  MPF_Dense *x
);

void mpf_pcg_memory_get
(
  MPF_DataType data_type,
  MPF_MatrixType struct_type,
  KrylovMeta meta,
  MPF_Int n,
  MPF_Int *memory_bytes
);

void mpf_seq_solve
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

//pthread_mutex_t mutex;
//pthread_mutex_t mutex_merge;

void mpf_blk_cg_memory_get
(
  MPF_DataType data_type,
  MPF_MatrixType struct_type,
  KrylovMeta meta,
  MPF_Int n,
  MPF_Int *memory_bytes
);

void mpf_solve_gbl_pcg_dsy_constrained
(
  /* context parameters */
  MPF_Solver context,

  /* data */
  const MPF_Sparse *A,
  const double *B,
  double *X
);

void mpf_memory_outer_get
(
  MPF_Solver *context
);


void mpf_batch_d_defl_solve
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *diag_fA
);

void mpf_batch_z_defl_solve
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *diag_fA
);

void mpf_batch_d_spai
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Sparse *fA
);

void mpf_batch_d_solve_pthread
(
  MPF_Solver *context,
  MPF_Sparse *A,
  MPF_Dense *diag_fA
);

void mpf_zhe_gbl_lanczos
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_sparse_dsy_cheb
(
  MPF_Solver *solver,

  const MPF_Int iters,
  const double lmin,
  const double lmax,
  const double *c,

  /* data */
  const MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_sparse_dsy_ev_min_iterative
(
  MPF_Solver *solver,

  /* solver parameters */
  VSLStreamStatePtr stream,

  /* data */
  MPF_Sparse *A,
  double *ev_min
);

void mpf_ev_max_get_mem
(
  MPF_Solver *solver
);

void mpf_sparse_dsy_ev_max_iterative
(
  MPF_Solver *solver,
  VSLStreamStatePtr stream,

  /* data */
  MPF_Sparse *A,

  MPF_Int iterations_ev,
  double *ev_max
);

void mpf_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_defl_lanczos_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_spbasis_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_spbasis_defl_cg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int n_defl
);

void mpf_pcg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  char *precond,
  char *filename
);

void mpf_blk_gmres_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_gmres_init
(
  MPF_ContextHandle solver,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_blk_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_gbl_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_spbasis_blk_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_blk_cg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations
);

void mpf_cg_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations
);

void mpf_cheb_get_cheb
(
  MPF_DataType data_type,
  MPF_Int cheb_M,
  MPF_Int iterations,
  MPF_Int n,
  MPF_Int *bytes
);

void mpf_dsy_spbasis_defl_cg_get_mem_size
(
  MPF_Solver *solver
);

void mpf_ev_defl_cg_get_mem_size
(
  MPF_Solver *solver
);

void mpf_ev_defl_lanczos_get_mem_size
(
  MPF_Solver *solver
);

void mpf_batch_solve_alloc
(
  MPF_Solver *solver
);

void mpf_solver_free
(
  MPF_Solver *solver
);

void mpf_set_B
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_diag_d_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B
);

void mpf_diag_zhe_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B
);

void mpf_diag_zsy_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B
);

void mpf_batch_init
(
  MPF_ContextHandle context,
  MPF_Int blk,
  MPF_Int nthreads_outer,
  MPF_Int nthreads_inner
);

void mpf_dense_alloc
(
  void *A_in
);

void mpf_dense_free
(
  MPF_Dense *A
);

void mpf_d_select
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_blk_cg_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations
);

void mpf_gbl_cg_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations
);

void mpf_gbl_pcg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  char *precond,
  char *filename
);

void mpf_gbl_gmres_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_spbasis_gbl_lanczos_init
(
  MPF_ContextHandle context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_blk_cg_get_mem_size
(
  MPF_Solver *solver
);

void mpf_dsy_gbl_cg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *B,
  MPF_Dense *X
);

void mpf_krylov_free
(
  MPF_Solver *solver
);

void mpf_get_max_nrhs
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_cg_alloc
(
  MPF_Solver *solver
);

void mpf_preprocess_diag
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

/* --------------------- sparse reconstruct functions ----------------------- */

void mpf_spai_d_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_spai_zhe_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_spai_zsy_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_d_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_z_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_fA_alloc
(
  MPF_Context *context
);

void mpf_spai_init
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_sparse_csr_order
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Sparse *A
);

void mpf_krylov_alloc
(
  MPF_Solver *solver
);

void mpf_d_generate_B
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Dense *B
);

void mpf_z_generate_B
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Dense *B
);

void mpf_sparse_csr_d_order
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *A
);

void mpf_sparse_csr_z_order
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *A
);

template <typename T>
void mpf_inner_solve_wrapper
(
  MPF_Solver *solver,
  T A
);

template <typename T>
void mpf_inner_solve_gko_wrapper
(
  MPF_Solver *solver,
  T A
  //gko::matrix::Csr<double> *A
);

void mpf_batch_gko_solve
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Args *args,
  void *fA_out
);

//void mpf_ls_dsy_horner
//(
//  MPF_Solver *solver,
//  MPF_Sparse *A,
//  MPF_Dense *B,
//  MPF_Dense *X
//);
//
//void mpf_ls_init
//(
//  MPF_Solver *solver,
//  MPF_Int ld
//);


MPF_Int mpf_spai_get_nz
(
  MPF_Sparse *A,
  MPF_Int blk_rec,
  MPF_Int *temp_array,
  MPF_Int *temp_i_array
);

void mpf_precond_dsy_blkdiag_alloc
(
  MPF_Solver *solver
);

void mpf_sparse_csr_d_diag_alloc
(
  MPF_Solver *solver
);

void mpf_sparse_precond_free
(
  MPF_Solver* solver
);

void mpf_d_precond_init
(
  MPF_Solver* solver
);

void mpf_jacobi_precond_init
(
  MPF_ContextHandle context
);

void mpf_jacobi_precond_generate
(
  MPF_Solver* solver,
  MPF_Sparse* A
);

#endif /* MPF_SOLVE_H -- end */
