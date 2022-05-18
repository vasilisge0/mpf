#ifndef MPF_AUX_H
#define MPF_AUX_H

#include "mpf_types.h"

typedef char MM_typecode[4];

/* -------------------------- bucket_array functions ------------------------ */

void mpf_hashatable_init
(
  MPF_BucketArray *H,
  MPF_Int num_values,
  MPF_Int num_bins
);

void mpf_bucket_array_values_init
(
  MPF_BucketArray *H
);

void mpf_bucket_array_alloc
(
  MPF_BucketArray *H
);

void mpf_bucket_array_insert
(
  MPF_BucketArray *H,
  MPF_Int bin,
  MPF_Int value
);

void mpf_bucket_array_find_max_bin_size
(
  MPF_BucketArray *H
);

void mpf_bucket_array_free
(
  MPF_BucketArray *H
);

/* -----------------------  sparse auxiliary functions ---------------------- */

//void mpf_sparse_csr_export
//(
//  MPF_Context *context,
//  MPF_Target export_target
//);


/* ------------------- allocators for sparse matrix objects ----------------- */


void mpf_sparse_array_free
(
  MPF_Sparse *A_array,
  MPF_Int length
);

MPF_SparseCoo* mpf_sparse_coo_array_create
(
  MPF_Int array_length
);

MPF_SparseCoo* mpf_sparse_coo_array_allocate
(
  MPF_Int array_length
);

MPF_SparseCsr* mpf_sparse_csr_array_create
(
  MPF_Int array_length
);

MPF_SparseCsr* mpf_sparse_csr_array_allocate
(
  MPF_Int array_length
);

MPF_SparseHandle* mpf_sparse_handle_create
(
  MPF_Int array_length
);

void mpf_sparse_input_coo_allocate
(
  MPF_Context *context
);

void mpf_sparse_coo_alloc
(
  MPF_Sparse *A
);


/* ------------------ deallocators for sparse matrix objects ---------------- */


//void mpf_sparse_coo_free
//(
//  MPF_Context *context,
//  MPF_Target A_select
//);
//
//void mpf_sparse_csr_free
//(
//  MPF_Context *context,
//  MPF_Target A_select
//);

void mpf_sparse_csr_internal_free
(
  MPF_Context *context,
  MPF_SparseCsr *A
);

MPF_SparseCoo* mpf_sparse_coo_create
(
  MPF_Context *context
);

MPF_SparseCoo* mpf_sparse_coo_d_create
(
  const MPF_Int nz
);

MPF_SparseCoo* mpf_sparse_coo_z_create
(
  const MPF_Int nz
);

void mpf_sparse_coo_destroy
(
  MPF_SparseCoo *object_coo
);

MPF_SparseCsr* mpf_sparse_csr_d_create
(
  MPF_Context *context
);

MPF_SparseCsr* mpf_sparse_csr_s_create
(
  MPF_Context *context
);

MPF_SparseCsr* mpf_sparse_csr_z_create
(
  MPF_Context *context
);

void mpf_sparse_csr_destroy
(
  MPF_SparseCsr *object_csr
);

void mpf_sparse_coo_d_allocate
(
  MPF_SparseCoo *A
);

void mpf_csr_d_allocate
(
  MPF_Context *context,
  MPF_SparseCsr *A
);

/* ------------------------------  QR functions ----------------------------- */

void mpf_qr_givens_dge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
);

void mpf_qr_givens_dge_2
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
);

void mpf_qr_givens_mrhs_dge_2
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
);

void mpf_qr_givens_mrhs_dge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
);

void mpf_qr_givens_sge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *b,
  MPF_Int ld_br,
  float *tempf_matrix
);

void mpf_block_qr_dge_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B,
  const MPF_Int blk
);

void mpf_block_qr_sge_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *B,
  const MPF_Int ld_B,
  const MPF_Int blk
);

void mpf_qr_dsy_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b
);

void mpf_qr_ssy_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *b
);

void mpf_qr_block_givens_dsy
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B
);

void mpf_block_qr_ssy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  float *H,
  const MPF_Int ld_H,
  float *B,
  const MPF_Int ld_B
);

void mpf_qr_zge_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_qr_givens_zge_2
(
  MPF_ComplexDouble *H,
  MPF_Int ld_H,
  MPF_ComplexDouble *b,
  MPF_Int ld_b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_qr_zge_givens_3
(
  MPF_ComplexDouble *H,
  MPF_Int ld_H,
  MPF_ComplexDouble *b,
  MPF_Int ld_b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_zge_qr_givens_mrhs
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_qr_zge_givens_mrhs_2
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_qr_givens_cge
(
  MPF_Complex *H,
  MPF_Complex *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_Complex *tempf_matrix
);

void mpf_qr_zsy_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B
);

void mpf_qr_zsy_givens_2
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int ld_H,
  const MPF_Int n_H,
  const MPF_Int n_B
);

void mpf_qr_csy_givens
(
  MPF_Complex *H,
  MPF_Complex *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B
);

void mpf_qr_zsy_hessen_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  MPF_Int m_H,
  MPF_Int n_H,
  MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
);

void mpf_block_qr_zge_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
);

void mpf_block_qr_zhe_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
);

void mpf_block_qr_cge_givens
(
  MPF_Complex *H,
  MPF_Complex *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
);

void mpf_block_qr_dsy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B
);

void mpf_block_qr_zsy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  MPF_ComplexDouble *H,
  const MPF_Int ld_H,
  MPF_ComplexDouble *B,
  const MPF_Int ld_B
);

void mpf_block_qr_csy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  MPF_Complex *H,
  const MPF_Int ld_H,
  MPF_Complex *B,
  const MPF_Int ld_B
);

void mpf_gram_schmidt_zge  //@BUG: ld_B is missing
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_ComplexDouble *B,
  MPF_ComplexDouble *H,
  const MPF_Int m_H
);

void mpf_gram_schmidt_zhe
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_ComplexDouble *B,
  MPF_ComplexDouble *H,
  const MPF_Int m_H
);

void mpf_gram_schmidt_cge
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_Complex *B,
  MPF_Complex *H,
  const MPF_Int m_H
);


/* ---------------------  dense matrix manipulation functions ----------------*/


void mpf_zeros_d_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  double *A,
  const MPF_Int ld_A
);

void mpf_zeros_s_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  float *A,
  const MPF_Int ld_A
);

void mpf_zeros_z_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_ComplexDouble *A,
  const MPF_Int ld_A
);

void mpf_zeros_c_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Complex *A,
  const MPF_Int ld_A
);

void mpf_zeros_i_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Int *A,
  const MPF_Int ld_A
);

void mpf_matrix_i_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Int *A,
  const MPF_Int ld_A,
  const MPF_Int val
);

void mpf_matrix_d_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  double *A,
  const MPF_Int ld_A,
  const double val
);

void mpf_matrix_z_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_ComplexDouble *A,
  const MPF_Int ld_A,
  const MPF_ComplexDouble *val
);


/* ------------------  complex scalar manipulation functions ---------------- */


MPF_ComplexDouble mpf_scalar_z_set
(
  const double variable_real
);

MPF_ComplexDouble mpf_scalar_z_init
(
  const double real_value,
  const double imag_value
);

MPF_ComplexDouble mp_scalar_z_init
(
  const double real_value,
  const double imag_value
);


MPF_ComplexDouble mpf_scalar_z_add
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
);

MPF_ComplexDouble mpf_scalar_z_divide
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
);

MPF_ComplexDouble mpf_scalar_z_multiply
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
);

MPF_ComplexDouble mpf_scalar_z_normalize
(
  MPF_ComplexDouble alpha,
  const double beta
);

MPF_ComplexDouble mpf_scalar_z_subtract
(
  MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
);

MPF_ComplexDouble mpf_scalar_z_invert_sign
(
  MPF_ComplexDouble alpha
);

MPF_Complex mpf_scalar_c_init
(
  const float real_value,
  const float imag_value
);

MPF_Complex mpf_scalar_c_add
(
  const MPF_Complex alpha,
  const MPF_Complex beta
);

MPF_Complex mpf_scalar_c_divide
(
  const MPF_Complex alpha,
  const MPF_Complex beta
);

MPF_Complex mpf_scalar_c_multiply
(
  const MPF_Complex alpha,
  const MPF_Complex beta
);

MPF_Complex mpf_scalar_c_normalize
(
  MPF_Complex alpha,
  const float beta
);

MPF_Complex mpf_scalar_c_subtract
(
  MPF_Complex alpha,
  const MPF_Complex beta
);

MPF_Complex mpf_scalar_c_invert_sign
(
  MPF_Complex alpha
);

/* --------------------------- I/O convert functions ------------------------ */

void mpf_convert_layout_to_sparse
(
  MPF_Layout layout,
  MPF_LayoutSparse *sparse_layout
);


MPF_ApproxType mpf_approx_type_get
(
  MPF_Context *context
);

/* --------------------------------- I/O ------------------------------------ */

void mpf_context_read
(
  MPF_Context *context
);

void mpf_context_write
(
  MPF_Context *context
);

void mpf_diag_write
(
  MPF_Context *context,
  MPF_Dense *fA
);

int mpf_matrix_size_read
(
  MPF_Int *m,
  MPF_Int *n,
  char *filename
);

int mpf_matrix_meta_read
(
  FILE *file_handler,
  MM_typecode *matcode,
  MPF_Int *m,
  MPF_Int *n
);

int mpf_matrix_d_write
(
  FILE *file_handle,
  MM_typecode matcode,
  const double *handle,
  const MPF_Int m,
  const MPF_Int n
);

int mpf_matrix_i_write
(
  FILE *handle_file,
  MM_typecode matcode,
  const int *handle,
  const MPF_Int m,
  const MPF_Int n
);

int mpf_sparse_size_read_ext /* external version takes input file_handle */
(
  FILE *handle_file,
  MPF_Int *m,
  MPF_Int *n,
  MPF_Int *nz
);

int mpf_sparse_size_read
(
  MPF_Sparse *A,
  char *filename_A
);

int mpf_sparse_meta_read
(
  MPF_Sparse *A,
  char *filename_A,
  MM_typecode* typecode_A
);

int mpf_sparse_coo_read
(
  MPF_Sparse *A,
  char *filename_A,
  char *typecode_A
);

void mpf_sparse_coo_write
(
  MPF_Sparse *A,
  char *filename,
  MM_typecode matrix_code
);

void mpf_output_matrix_write
(
  MPF_Context *context,
  MPF_Target source
);

/* ----------------------------- printout functions ------------------------- */

void mpf_matrix_d_print
(
  const double *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
);

void mpf_matrix_d_announce
(
  const double *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
);

void mpf_matrix_z_announce
(
  const MPF_ComplexDouble *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
);


void mpf_complex_matrix_print
(
  MPF_ComplexDouble *A,
  MPF_Int m_A,
  MPF_Int n_A,
  MPF_Int ld_A
);

void mpf_matrix_z_print
(
  const MPF_ComplexDouble *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
);

void mpf_matrix_c_print
(
  const MPF_Complex *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
);

void mpf_sparse_coo_z_print
(
  const MPF_Sparse *storage_A
);

void mpf_printout
(
  MPF_Context *context
);


/* --------------------------- error evaluation ----------------------------- */


void mpf_diag_error_evaluate
(
  MPF_Context *context,
  void *diag_true_vector
);


/* --------------------------------- debug -----------------------------------*/


void mpf_debug
(
  char *filename,
  double *A,
  MPF_Int m_A,
  MPF_Int n_A,
  MPF_Int ld_A
);


/* --------------------------- context functions ---------------------------- */


void mpf_context_destroy
(
  MPF_ContextHandle context
);

MPF_Error mpf_context_create
(
  MPF_ContextHandle *context_handle,
  MPF_Int argc,
  char *argv[]
);


/* -------------------------- hashtable functions --------------------------- */

void mpf_bucket_array_init
(
  MPF_BucketArray *H,
  MPF_Int num_values,
  MPF_Int num_bins
);

void mpf_bucket_array_values_init
(
  MPF_BucketArray *H
);


void mpf_bucket_array_insert
(
  MPF_BucketArray *H,
  MPF_Int bin,
  MPF_Int value
);

void mpf_bucket_array_find_max_bin_size
(
  MPF_BucketArray *H
);

void mpf_bucket_array_free
(
  MPF_BucketArray *H
);


/* */

MPF_Int mpf_i_max
(
  const MPF_Int alpha,
  const MPF_Int beta
);

MPF_Int mpf_i_min
(
  const MPF_Int alpha,
  const MPF_Int beta
);

/* ---------------------- pattern manipulation functions -------------------- */

void mpf_sparse_csr_export
(
  MPF_Context *context,
  MPF_Target export_target
);

/* ------------------------------- threading -------------------------------- */

void mpf_probing_rhs_meta_pthreads_initialize
(
  MPF_Context *context
);

MPF_ContextPthreads *mpf_context_pthreads_array_create
(
  MPF_Context *context
);

void mpf_context_pthreads_array_destroy
(
  MPF_ContextPthreads *context_pthreads_array
);

MPF_SolverOuter_Pthreads *mpf_solver_outer_pthreads_array_create
(
  MPF_Solver *solver
);

//MPF_ContextOpenmp *mpf_context_openmpf_array_create
//(
//  MPF_Context *context
//);

//void mpf_context_openmpf_array_destroy
//(
//  MPF_ContextOpenmp *context_openmpf_array
//);


/* ------------------------ probing utility functions ----------------------- */


void mpf_colorings_allocate
(
  MPF_Context *context
);

void mpf_probing_allocate
(
  MPF_Context *context
);

void mpf_memory_probing_get
(
  MPF_Context* context
);

void mpf_probe_free
(
  MPF_Probe *Probe
);

/* ----------------------- sparsified krylov functions ---------------------- */

void krylov_dge_sparse_basis_combine
(
  MPF_Int m_V,
  MPF_Int n_V,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  double *c_vector,
  double *X,
  MPF_Int m_X
);

MPF_Int mpf_get_blk_max_fA(MPF_Context *context);

void mpf_set_B_2(MPF_Context *context);

void mpf_B_meta_export
(
  MPF_Context *context,
  MPF_Int *n_max_B,
  MPF_Int *ld_B,
  MPF_Int *n_blk
);

extern void mpf_set_B_2
(
  MPF_Context *context
);

extern MPF_Int mpf_get_blk_max_fA(MPF_Context *context);

extern MPF_Int mpf_n_diags_get(MPF_Int n_levels, MPF_Int degree);

extern void mpf_probing_sampling_offsets_get
(
  MPF_Context *context,
  MPF_Int *offset_rows,
  MPF_Int *offset_cols
);

extern void mpf_probing_stride_get
(
  MPF_Context *context,
  MPF_Int *stride
);

extern void mpf_probing_blocking_init
(
  MPF_Context *context,
  MPF_Int blk,
  MPF_Int n_levels
);

extern void mpf_probing_multilevel_sampling_init
(
  MPF_Context *context,
  MPF_Int stride,
  MPF_Int n_levels
);

extern void mpf_probing_multipath_sampling_init
(
  MPF_Context *context,
  MPF_Int stride,
  MPF_Int n_levels
);


/* -------------------- cpu solver initialization functions ----------------- */


void mpf_block_gmres_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_global_gmres_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

extern void mpf_block_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

extern void mpf_sparsified_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_sparsified_block_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_global_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_global_cg_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations
);

void mpf_block_cg_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations
);

void mpf_sparsified_global_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

/* ---------------- solver_batch_outer initialization functions ------------- */

void mpf_batch_2pass_init
(
  MPF_Solver *context,
  MPF_Int blk
);

void mpf_dynamic_batch_init
(
  MPF_Context *context
);

void mpf_batch_sparsified_init
(
  MPF_Context *context
);

void mpf_batch_sparsified_dynamic_init
(
  MPF_Context *context
);

/* ------------------------- solver utility functions ----------------------- */

void mpf_solver_allocate
(
  MPF_Context *context
);

void mpf_memory_inner_get
(
  MPF_Context *context
);

void mpf_solver_memory_get
(
  MPF_Context *context
);

void mpf_solver_info_allocate
(
  MPF_Context *context
);

void mpf_solver_info_free
(
  MPF_Context *context
);

/* ------------------- linked list data structure functions ----------------- */

MPF_LinkedList *mpf_linked_list_create
(
  MPF_Int num_entries
);

void mpf_linked_list_destroy
(
  MPF_LinkedList *list
);

/* -------------  assignment based diagonal extraction functions ------------ */

void mpf_matrix_d_diag_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A,
  MPF_Int ld_A,
  double value
);

void mpf_matrix_z_diag_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  MPF_ComplexDouble *A,
  MPF_Int ld_A,
  MPF_ComplexDouble value
);

/*============================================================================*/
/*== Set of sparsified_ utiliy functions (used in sparsified, combressed    ==*/
/*== Krylov solvers)                                                        ==*/
/*============================================================================*/

void vector_d_sparsify
(
  MPF_Int m_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_BucketArray *H
);

void vecblk_d_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
);

void vecblk_d_block_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
);

void vecblk_z_block_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  MPF_ComplexDouble *v_in_vector,
  MPF_ComplexDouble *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
);

void vector_z_sparsify
(
  MPF_Int m_v,
  MPF_ComplexDouble *v_in_vector,
  MPF_ComplexDouble *v_out_vector,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_BucketArray *H
);

void block_krylov_dge_sparse_basis_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
);

void block_krylov_dge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
);

void block_krylov_zge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
);

void global_krylov_dge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
);

void global_krylov_zge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
);

void krylov_zge_sparse_basis_combine
(
  MPF_Int m_V,
  MPF_Int n_V,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
);

void mpf_sparse_csr_to_pattern_export(
  MPF_Context *context
);

void mpf_memory_pattern_allocate
(
  MPF_Context *context
);

void mpf_probing_memory_get
(
  MPF_Context* context
);

MPF_Int mpf_blk_max_fA_get
(
  MPF_Context *context
);

int mpf_matrix_z_write
(
  FILE *handle_file,
  MM_typecode matcode,
  const MPF_ComplexDouble *handle,
  const MPF_Int m,
  const MPF_Int n
);

void mpf_byte_buffer_write
(
  FILE *file_h,
  MPF_Int n_bytes,
  void *buffer
);

void mpf_int_buffer_write
(
  FILE *file_h,
  MPF_Int n_entries,
  void *buffer
);

void mpf_double_buffer_write
(
  FILE *file_h,
  MPF_Int n_entries,
  void *buffer
);

/* ---------------------  context read input functions ---------------------- */

void mpf_context_filenames_read
(
  MPF_Context *context,
  char *argv
);

void mpf_solver_init
(
  MPF_Context *context
);

void mpf_d_plot
(
  char filename[MPF_MAX_STRING_SIZE],
  MPF_Int n_points,
  double *v,
  char header[MPF_MAX_STRING_SIZE]
);

MPF_Error mpf_matrix_d_read
(
  FILE *file_handle,
  MPF_Layout layout_A,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A
);

MPF_Error mpf_matrix_i_read
(
  FILE *file_handle,
  MPF_Layout layout_A,
  MPF_Int m_A,
  MPF_Int n_A,
  MKL_INT* A
);

MPF_Int mpf_validate
(
  MPF_Context *context
);

void mpf_bucket_array_write
(
  FILE *file_handle,
  MPF_BucketArray *H
);

void mpf_d_sparse_toep_create
(
  MPF_Int N,  /* length of diags */
  double *diags,
  MPF_SparseCoo *A
);

void mpf_covariance_samples_generate
(
  VSLStreamStatePtr stream,
  MPF_Int n_params,
  MPF_Int n_samples,
  double *values
);

void mpf_quic
(
  MPF_Int argc,
  char *argv[]
);

void mpf_quic_init
(
  MPF_Context *context,
  MPF_Int argc,
  char *argv[],
  double *lambda
);

void mpf_covariance_update
(
  MPF_Int m_V,
  MPF_Int n_V,
  double *V,  /* each column is a sample vector */
  MPF_Int ld_V,
  double lambda_tol,  /* sparsification threshold */
  MPF_Int blk,
  MPF_SparseCoo *A,  /* sparsified sample-covariance matrix */
  double *tempf_matrix,
  MPF_Int memory_inc
);

void mpf_matrix_d_threshold_apply
(
  MPF_Int blk_r,
  MPF_Int blk_c,
  MPF_Int I_r,
  MPF_Int I_c,
  MPF_Int m_V,
  MPF_Int n_V,
  double *M,
  MPF_Int ld_M,
  double threshold,
  MPF_SparseCoo *A,
  MPF_Int memory_inc
);

/* -------------------------- eigenvalue computation ------------------------ */

void mpf_dsy_X_defl
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,          /* (1) rows V */
  MPF_Int n_V,          /* (3) num_cols_V */
  MPF_Int m_H,          /* (4) num_rows_H */
  MPF_Int n_H,          /* (5) num_cols_H */
  MPF_Int n_B,          /* (2) num_cols_rhs */

  double *U,          /* (6) input */
  double *W,          /* (6) output */
  double *memory_defl /* (8) */
);

void mpf_dsy_X_defl_rec
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,          /* (1) rows V */
  MPF_Int n_V,          /* (3) num_cols_V */
  MPF_Int m_H,          /* (4) num_rows_H */
  MPF_Int n_H,          /* (5) num_cols_H */
  MPF_Int n_B,          /* (2) num_cols_rhs */

  double *U,          /* (6) input */
  double *W,          /* (6) output */
  double *memory_defl /* (8) */
);

void mpf_dsy_B_defl
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,          /* (1) rows V */
  MPF_Int n_V,          /* (3) num_cols_V */
  MPF_Int m_H,          /* (4) num_rows_H */
  MPF_Int n_H,          /* (5) num_cols_H */
  MPF_Int n_B,          /* (2) num_cols_rhs */

  double *U,          /* (6) input */
  double *memory_defl /* (8) */
);

void mpf_dsy_seed
(
  const MPF_SparseHandle A_handle,  /* (1) */
  const MPF_SparseDescr A_descr,    /* (2) */

  MPF_Int m_V,                      /* (3) rows V */
  MPF_Int n_V,                      /* (4) num_cols_V */
  MPF_Int m_H,                      /* (5) num_rows_H */
  MPF_Int n_H,                      /* (6) num_cols_H */
  MPF_Int n_B,                      /* (7) num_cols_rhs */

  double *U,                      /* (8) input */
  double *memory_defl             /* (10) */
);

void mpf_defl_memory_get
(
  MPF_DataType data_type,
  KrylovMeta meta,
  MPF_Int n,
  MPF_Int n_blocks,
  MPF_Int *memory_bytes
);

void mpf_context_allocate
(
  MPF_Context *context
);

/* new qr_dsy_givens to that saves reflectors */
void mpf_qr_dsy_ref_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
);

void mpf_qr_dsy_rhs_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
);

void mpf_diag_d_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A,
  MPF_Int ld_A,
  double *values
);

void mpf_cheb_init
(
  MPF_Context *context,
  char target_function[MPF_MAX_STRING_SIZE],
  MPF_Int cheb_M,
  MPF_Int iterations
);

void mpf_dsy_cheb_poly
(
  MPF_Int M,    /* number of points = polynomial_degree-1 */
  double lmin, /* minimum eigenvalue */
  double lmax, /* maximum eigenvalue */
  MPF_FunctionPtr target_func,
  double *memory,
  double *cheb_coeffs
);

void mpf_dsy_inv_1D
(
  MPF_Int n,
  double *x,
  double *fx
);

void mpf_qr_ev
(
  const MPF_Layout layout,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *R,
  double *Z,
  MPF_Int max_iters,
  double tol
);

void mpf_matrix_d_sy2b
(
  char mode,  /* indicates upper of lower triangular storage */
  MPF_Int m_H,  /* number of rows of matrix H */
  MPF_Int n_H,  /* number of columns of matrix H */
  double *H,  /* input banded matrix in standard dense storage scheme */
  MPF_Int ld_H, /* leading dimension of matrix H */
  MPF_Int k,    /* number of bands */
  double *h   /* output vector containing entries in banded storage scheme */
);

void mpf_matrix_d_sy_diag_extract
(
  char mode,  /* indicates upper of lower triangular storage */
  MPF_Int m_H,  /* number of rows of matrix H */
  MPF_Int n_H,  /* number of columns of matrix H */
  double *H,  /* input banded matrix in standard dense storage scheme */
  MPF_Int ld_H, /* leading dimension of matrix H */
  MPF_Int k,    /* current band */
  double *h   /* output vector containing entries in banded storage scheme */
);

void mpf_qr_dsy_mrhs_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
);


void mpf_global_pcg_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  char *precond,
  char *filename
);

void mpf_sparse_d_mm_wrapper
(
  MPF_Solver *context,
  double *B,
  double *X
);

void mpf_matrix_i_print
(
  const MPF_Int *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
);

void mpf_matrix_i_announce
(
  const MPF_Int *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
);

void mpf_probing_pthread_blocking_init
(
  MPF_Context *context,
  MPF_Int n_threads_probing,
  MPF_Int blk_threads_probing,
  MPF_Int blk,
  MPF_Int n_levels
);

void mpf_linked_list_init
(
  MPF_Int n_entries,
  MPF_LinkedList *list
);

void mpf_linked_list_free
(
  MPF_LinkedList *list
);


void mpf_zsy_cheb_poly
(
  MPF_Int M,    /* number of points = polynomial_degree-1 */
  MPF_ComplexDouble lmin, /* minimum eigenvalue */
  MPF_ComplexDouble lmax, /* maximum eigenvalue */
  MPF_FunctionPtr target_func,
  MPF_ComplexDouble *memory,
  MPF_ComplexDouble *cheb_coeffs
);

double mpf_sparse_d_dot
(
  MPF_Int m_A,
  MPF_Int m_B,
  MPF_Int *A_ind,
  double *A_data,
  MPF_Int *B_ind,
  double *B_data
);

void mpf_jacobi_precond_csr_init
(
  MPF_Solver *solver,
  MPF_Sparse *A
);

void mpf_precond_krylov_csr_create
(
  MPF_Int m,
  MPF_SparseCsr *A,
  KrylovMeta *meta
);

void mpf_krylov_precond_csr_destroy
(
  KrylovMeta *meta
);

void mpf_probing_blocking_mkl_init
(
  MPF_Context *context,
  MPF_Int blk,
  MPF_Int n_levels
);

void mpf_precond_csr_diag_blocks_init
(

  MPF_Int m,
  MPF_Int blk,
  double *d,
  MPF_SparseCsr *M
);

double mpf_time
(
  struct timespec start,
  struct timespec finish
);

void mpf_time_reset
(
  struct timespec *start,
  struct timespec *finish
);

void mpf_probing_blocking_mkl_low_mem_init
(
  MPF_Context *context,
  MPF_Int blk,
  MPF_Int n_levels,
  MPF_Int blk_probing_low_mem
);

void mpf_context_mem_sort_unpack
(
  MPF_Context *context,
  MPF_Int **tempf_array,
  MPF_Int **tempf_i_array
);

void mpf_sparse_debug
(
  MPF_Int start,
  MPF_Int end,
  MPF_SparseCsr *A,
  char msg[],
  MPF_Int *colorings_array
);

void mpf_probing_blocking_mkl_low_mem_init_Acoarse
(
  MPF_Context *context,
  MPF_Int blk,
  MPF_Int n_levels,
  MPF_Int blk_probing_low_mem
);

void mpf_meta_write
(
  MPF_Context *context
);

void mpf_seq_global_lanczos_init
(
  MPF_Context *context,
  double tolerance,
  MPF_Int iterations,
  MPF_Int restarts
);

void mpf_probing_blocking_mkl_hybrid_init
(
  MPF_Context *context,
  MPF_Int blk,
  MPF_Int n_levels
);


int mpf_matrix_read
(
  MPF_Context *context
);

void mpf_pattern_array_free
(
  MPF_Context *context
);

void mpf_context_write_fA
(
  MPF_ContextHandle mpf_handle
);

void mpf_write_fA
(
  MPF_ContextHandle mpf_handle,
  char filename_fA[]
);

void mpf_write_log
(
  MPF_Context* context,
  char filename_meta[],
  char filename_caller[]
);

#endif /* end -- AUXILIARY_H */
