#ifndef MPF_PROBE_H /* MPF_H.h -- start */
#define MPF_PROBE_H

#include "mpf_types.h"


/* ---------------  inverse pattern_fA approximation functions -------------- */

void mpf_blocking_xy_psy_fA
(
  MPF_Context *context
);

/* -------- auxiliary functions for approximating the inverse pattern --------*/

void mpf_multipath_endpoints_generate
(
  MPF_Context *context
);

void mpf_multipath_mappings_init
(
  MPF_Context *context
);

void mpf_multipath_offsets_get
(
  MPF_Context *context,
  MPF_Int edge_id,
  MPF_Int *row_offset,
  MPF_Int *col_offset
);

void mpf_average_multipath_offsets_get
(
  MPF_Context *context,
  MPF_Int edge_id,
  MPF_Int *row_offset,
  MPF_Int *col_offset
);

void mpf_multipath_path_unpack
(
  MPF_Int starting_node,
  MPF_Int stride,
  MPF_Int *operator_path,
  MPF_Int path_length
);

int mpf_max_i_get
(
  MPF_Int *v_array,
  MPF_Int v_length
);

/* ----------------- probing memory allocation functions -------------------- */

void mpf_probe_init
(
  MPF_Probe *context,
  MPF_Int m
);

void mpf_probing_finish
(
  MPF_Context *context
);

/* ----------------------- contraction functions ---------------------------- */


void mpf_sampling_contract
(
  MPF_Context *context,
  MPF_SparseCsr *A_csr,
  MPF_SparseCsr *Asample_csr
);


void mpf_psy_multilevel_2
(
  MPF_Context *context
);

void mpf_color
(
  MPF_Probe *context
);

void mpf_color_decoupled
(
  MPF_Sparse* P,
  MPF_Int* colorings_array,
  MPF_Int* ncolors
);

void mpf_color_partial
(
  MPF_Probe *context,
  MPF_Int current_row,
  MPF_Int current_blk,
  MPF_LinkedList *list
);

void mpf_probing_meta_init
(
  MPF_Context *context
);

void mpf_color_to_node_map_alloc
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

void mpf_color_to_node_map_set
(
  MPF_Probe *probe,
  MPF_Solver *solver
);

/* --------------------  right hand side operations ------------------------- */

void mpf_d_blk_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_d_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_d_xy_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int n_levels,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_z_blk_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  MPF_ComplexDouble *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_z_select_X
(
  MPF_Int blk_max_fA,
  MPF_Int m,
  MPF_Int n,
  MPF_ComplexDouble *B,
  MPF_ComplexDouble *X,
  MPF_ComplexDouble *buffer
);

void mpf_d_select_X
(
  MPF_Int blk_max_fA,
  MPF_Int m,
  MPF_Int n,
  double *B,
  double *X,
  double *buffer
);

void mpf_z_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  MPF_ComplexDouble *X,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

//void mpf_z_generate_B
//(
//  MPF_Int blk_max_fA,
//  MPF_Int *colorings_array,
//  MPF_Int cols_start,
//  MPF_Int m_B,
//  MPF_Int n_B,
//  MPF_ComplexDouble *B
//);
//
//void mpf_d_generate_B
//(
//  MPF_Int blk_max_fA,
//  MPF_Int *colorings_array,
//  MPF_Int cols_start,
//  MPF_Dense *B
//);


/*=======================*/
/*== probing functions ==*/
/*=======================*/

void mpf_probing_init
(
  MPF_Context *context
);

void mpf_pattern_multisample_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels,
  MPF_Int n_endpoints
);

void mpf_pattern_sample_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
);

void mpf_d_xy_generate_B
(
  MPF_Int blk_max_fA,
  MPF_Int n_levels,
  MPF_Int *colorings_array,
  MPF_Int cols_start,
  MPF_Int m_B,
  MPF_Int n_B,
  double *B
);

void mpf_d_blk_xy_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_row_contract
(
  MPF_Int curr_rhs,
  MPF_Int stride,
  MPF_Int *n_blocks,
  MPF_Sparse *P_csr,
  MPF_Int *row,
  MPF_Int *row_rev
);

void *mpf_pthread_blocking_psy_fA_kernel
(
  void *t_context_packed
);

void mpf_color_unordered
(
  MPF_Probe *context,
  MPF_Int max_coloring,
  MPF_Int n_V,
  MPF_Int *V
);

void mpf_pthread_color
(
  MPF_Probe *context
);


void *mpf_pthread_kernel_color
(
  void *input_packed
);

void mpf_pthread_blocking_xy_contract
(
  //void *t_context_packed
  //MPF_Context *context,
  MPF_Int m_P,
  MPF_Int blk,
  MPF_Int *nz_new,
  MPF_Int row_start,
  MPF_Int row_end,
  MPF_SparseCsr *A_csr,
  MPF_SparseCsr *Ablk_csr,
  MPF_Int *tempf_array,
  MPF_Int *tempf_inverted_array,
  MPF_Int *tempf_cols
);

void mpf_pthread_blocking_xy_psy_fA
(
  MPF_Context *context
);

void *mpf_pthread_blocking_xy_psy_fA_kernel
(
  void *t_context_packed
);

void mpf_z_blk_xy_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  MPF_ComplexDouble *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
);

void mpf_block_probe
(
  MPF_Context *context
);

void mpf_blocking
(
  MPF_Probe *context,
  MPF_Sparse *A
);

void mpf_batch_blocking
(
  MPF_Probe *context,
  MPF_Sparse *A
);

void mpf_batch_compact_blocking
(
  MPF_Probe *context,
  MPF_Sparse *A
);

void mpf_blocking_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
);

void mpf_blocking_batch_coarse_init
(
  MPF_ContextHandle context,
  MPF_Int blk,
  MPF_Int n_levels,
  MPF_Int blk_probing_low_mem,
  MPF_Int expansion_degree
);

void mpf_blocking_batch_init
(
  MPF_ContextHandle context,
  MPF_Int blk,
  MPF_Int n_levels,
  MPF_Int batch,
  MPF_Int expansion_degree
);

void mpf_blocking_contract_mkl
(
  MPF_Context *context,         /* mp context, to be removed later and exact inputs be specified */
  MPF_SparseHandle A_handle,    /* input pattern in csr format */
  MPF_SparseHandle Ablk_handle, /* outpiut */
  MPF_Int *tempf_array,         /* temporary storage array */
  MPF_Int *tempf_inverted_array /* provides inverted indexing to tempf_array */
);

void mpf_contract_blocking
(
  MPF_Int blk,
  MPF_SparseDescr *descr,
  MPF_SparseHandle *A_handle, /* input pattern in csr format */
  MPF_Int *m_P,
  MPF_Int *nz_P,
  MPF_Int *tempf_array,       /* temporary storage array */
  MPF_Int *tempf_i_array,     /* provides inverted indexing to tempf_array */
  MPF_SparseCsr *Ablk
);

void mpf_blocking_mkl_fA_low_mem
(
  MPF_Context *contextj
);

void mpf_block_row_contract
(
  MPF_Int blk,
  MPF_Sparse *A,    /* input pattern in csr format */
  MPF_Int current_row,
  MPF_Int *tempf_array,    /* temporary storage array */
  MPF_Int *tempf_i_array,  /* provides inverted indexing to tempf_array */
  MPF_Sparse *C            /* output */
);

void mpf_batch_blocking_coarse
(
  MPF_Context *context  // main context
);

void mpf_pattern_sample
(
  MPF_Probe *context,
  MPF_Sparse *A
);

void mpf_pattern_multisample
(
  MPF_Probe *context,
  MPF_Sparse *A
);

void mpf_contract_sampling
(
  MPF_Probe *context,
  MPF_Sparse *A,
  MPF_Sparse *Asample
);

void mpf_multipath_node_unpack
(
  MPF_Int node_id,
  MPF_Int depth,
  MPF_Int *parent,
  MPF_Int *expand_op
);

void mpf_probe_alloc
(
  MPF_Probe *context
);

void mpf_avg_probe_alloc
(
  MPF_Probe *context
);

void mpf_probe_unpack_sort_mem
(
  MPF_Probe *probe,
  MPF_Int *tempf_array,
  MPF_Int *tempf_i_array
);

void mpf_block_contract
(
  MPF_Int blk,
  MPF_Int *temp_array,        /* temporary storage array */
  MPF_Int *temp_i_array,      /* provides inverted indexing to temp_array */
  MPF_Sparse *A, /* input pattern in csr format */
  MPF_Sparse *C
);

void mpf_contract_dynamic_sample
(
  MPF_Probe *context,
  MPF_Sparse *A,
  MPF_Sparse *B,
  MPF_Int coarse_op
);

void mpf_generate_sampling_endpoints
(
  MPF_Probe *probe
);

void mpf_compact_hierarchy
(
  MPF_Probe *probe,
  MPF_Sparse *A,
  MPF_Sparse *Ac_array
);

void mpf_contract_block_hybrid
(
  MPF_Int blk,
  MPF_Int *temp_array,      /* temporary storage array */
  MPF_Int *temp_i_array,    /* provides inverted indexing to temp_array */
  MPF_BucketArray *T,
  MPF_Sparse *A,            /* input pattern in csr format */
  MPF_BucketArray *M,
  MPF_Sparse *C             /* coarsened pattern */
);

void mpf_blocking_hybrid_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
);

void mpf_blocking_hybrid
(
  MPF_Probe *probe,
  MPF_Sparse *A
);

void mpf_probe_init
(
  MPF_Context* context
);

#endif /* MPF_H.h -- start */
