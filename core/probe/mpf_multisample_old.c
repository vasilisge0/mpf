#include "mpf.h"

///* optimzed */
//void mpf_pattern_multisample
//(
//  MPContext *context
//)
//{
//  MPSparseCsr A;
//  MPSparseCsr B;
//  MPSparseCsr C;
//
//  MPSparseHandle A_h;
//  MPSparseHandle B_h;
//  MPSparseHandle C_h;
//
//  MPF_Int p = 0;
//  MPF_Int i = 0;
//
//  MPF_Int status = 0;
//
//  MPF_Int stride = context->probing.average_multipath.stride;
//  //MPF_Int *tempf_array = context->memory_probing;
//  MPF_Int *endpoints = context->probing.average_multipath.endpoints_array;
//  MPF_Int *operator_path = context->probing.average_multipath.operator_path;
//  MPF_Int depth = context->n_levels;
//
//  context->runtime_contract = 0.0;
//  context->runtime_expand = 0.0;
//
//  sparse_index_base_t index;
//  MPF_Int nr, nc;
//
//  double *d = NULL;
//
//  struct timespec start;
//  struct timespec finish;
//
//  context->m_P = context->m_A;
//  if (context->P == NULL)
//  {
//    context->P = mpf_malloc(sizeof(MPPatternCsr));
//  }
//
//  context->A.csr.m = context->m_A;
//  context->A.csr.nz = context->nz_A;
//  mpf_sparse_d_csr_allocate(context->m_A, context->nz_A, &A);
//  mpf_sparse_d_csr_allocate(context->m_A, context->nz_A, &B);
//  mpf_sparse_d_csr_allocate(context->m_A, context->nz_A, &C);
//
//  /* iterates through the number of levels */
//  i = context->probe_id;
//
//  /* generates list of sampling operators that is used */
//  mpf_multipath_path_unpack(endpoints[i], stride, operator_path, depth);
//  #if MP_PRINTOUT
//    printf("i: %d\n", i);
//    printf("endpoint[%d]: %d -> %d, depth: %d\n", i,
//      endpoints[i], operator_path[0], depth);
//  #endif
//
//  /* contract */
//  clock_gettime(CLOCK_MONOTONIC, &start);
//  switch(operator_path[0])
//  {
//    case 0:
//      mpf_sample_contract_dynamic(context, &context->A.csr, &B, 0);
//      mpf_sample_contract_dynamic(context, &context->A.csr, &C, 0);
//    case 1:
//      mpf_sample_contract_dynamic(context, &context->A.csr, &B, 0);
//      mpf_sample_contract_dynamic(context, &context->A.csr, &C, 1);
//    case 2:
//      mpf_sample_contract_dynamic(context, &context->A.csr, &B, 1);
//      mpf_sample_contract_dynamic(context, &context->A.csr, &C, 1);
//    default:
//      break;
//  }
//  clock_gettime(CLOCK_MONOTONIC, &finish);
//  context->runtime_contract += mpf_time(start, finish);
//  #if MP_PRINTOUT
//    printf("operator_path[%d]: %d\n", i, operator_path[i]);
//    printf("B.m: %d, C.m: %d, %d\n", B.m, C.m, context->m_P);
//  #endif
//
//  context->m_P = (context->m_P+stride-1)/stride;
//
//  /* expand */
//  clock_gettime(CLOCK_MONOTONIC, &start);
//  status = mkl_sparse_d_create_csr(&A_h, INDEXING, context->m_P,
//    context->m_P, A.rows_start, A.rows_end, A.cols, A.data);
//  status = mkl_sparse_d_create_csr(&B_h, INDEXING, context->m_P,
//    context->m_P, B.rows_start, B.rows_end, B.cols, B.data);
//  status = mkl_sparse_d_create_csr(&C_h, INDEXING, context->m_P,
//    context->m_P, C.rows_start, C.rows_end, C.cols, C.data);
//  status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B_h, C_h,
//    &context->P_handle);
//  status = mkl_sparse_order(context->P_handle);
//  mkl_sparse_destroy(B_h);
//  mkl_sparse_destroy(C_h);
//  if (operator_path[0] == 1)
//  {
//    /* add the transpose */
//    mkl_sparse_copy(context->P_handle, context->A_descr, &C_h);
//    mkl_sparse_copy(context->P_handle, context->A_descr, &A_h);
//    mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, A_h, 1.0, C_h,
//      (sparse_matrix_t*)&context->P_handle);
//  }
//  clock_gettime(CLOCK_MONOTONIC, &finish);
//  context->runtime_expand += mpf_time(start, finish);
//
//  for (p = 1; p < depth; ++p)
//  {
//    #if MP_PRINTOUT
//      printf("\n\n >> p: %d\n\n", p);
//    #endif
//    status = mkl_sparse_d_export_csr(
//      context->P_handle, &index, &A.m, &nc,
//      &A.rows_start,
//      &A.rows_end,
//      &A.cols,
//      (double**)&A.data);
//
//    /* contract */
//    clock_gettime(CLOCK_MONOTONIC, &start);
//    switch(operator_path[0])
//    {
//      case 0:
//        mpf_sample_contract_dynamic(context, &A, &B, 0);
//        mpf_sample_contract_dynamic(context, &A, &C, 0);
//      case 1:
//        mpf_sample_contract_dynamic(context, &A, &B, 0);
//        mpf_sample_contract_dynamic(context, &A, &C, 1);
//      case 2:
//        mpf_sample_contract_dynamic(context, &A, &B, 1);
//        mpf_sample_contract_dynamic(context, &A, &C, 1);
//      default:
//        break;
//    }
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    context->runtime_contract += mpf_time(start, finish);
//
//    /* expand */
//    status = mkl_sparse_d_create_csr(&B_h, INDEXING, context->m_P,
//      context->m_P, B.rows_start, B.rows_end, B.cols, B.data);
//    status = mkl_sparse_d_create_csr(&C_h, INDEXING, context->m_P,
//      context->m_P, C.rows_start, C.rows_end, C.cols, C.data);
//    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B_h, C_h,
//      &context->P_handle);
//    status = mkl_sparse_order(context->P_handle);
//    mkl_sparse_destroy(B_h);
//    mkl_sparse_destroy(C_h);
//    if (operator_path[0] == 1)
//    {
//      /* add the transpose */
//      status = mkl_sparse_copy(context->P_handle, context->A_descr, &C_h);
//      status = mkl_sparse_copy(context->P_handle, context->A_descr, &A_h);
//      status = mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, A_h, 1.0, C_h,
//        (sparse_matrix_t*)&context->P_handle);
//      mkl_sparse_copy(C_h, context->A_descr, &context->P_handle); // this shoudl be commented change
//    }
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    context->runtime_expand += mpf_time(start, finish);
//  }
//  status = mkl_sparse_d_export_csr(
//    context->P_handle, &index, &nr, &nc,
//    &((MPPatternCsr*)context->P)->rows_start,
//    &((MPPatternCsr*)context->P)->rows_end,
//    &((MPPatternCsr*)context->P)->cols,
//    (double**)&d);
//
//  context->runtime_probing = context->runtime_contract
//    + context->runtime_expand;
//
//  #if MP_PRINTOUT
//    printf("nr (P): %d, nc (P): %d\n", nr, nc);
//    printf("(contract, expand, total) -> (%d, %d, %d)",
//      context->runtime_contract,
//      context->runtime_expand,
//      context->runtime_total);
//  #endif
//
//  /* frees memory */
//}

void mpf_multipath_path_unpack
(
  MPF_Int starting_node,
  MPF_Int stride,
  MPF_Int *operator_path,
  MPF_Int path_length
)
{
  MPF_Int i = 0;
  MPF_Int parent = 0;
  MPF_Int child = starting_node;

  printf(">>>> in unpack\n");
  for (i = path_length-1; i >= 0; --i)
  {
    printf("i: %d\n", i);
    mpf_multipath_node_unpack(child, path_length, &parent, &operator_path[i]);
    child = parent;
  }
  printf("out\n");
}

void mpf_avg_multipath_get_offsets
(
  MPF_Probe *context,
  MPF_Int edge_id,
  MPF_Int *row_offset,
  MPF_Int *col_offset
)
{
  MPF_Int stride = 0;
  *col_offset = context->mappings_array[edge_id] % (stride*(stride+1)/2);
  *row_offset = context->mappings_array[edge_id] / (stride*(stride+1)/2);
}

void mpf_multipath_node_unpack
(
  MPF_Int node_id,
  MPF_Int depth,
  MPF_Int *parent,
  MPF_Int *expand_op
)
{
  MPF_Int n_nodes_prev = 1;

  printf("n_nodes_prev: %d\n", n_nodes_prev);
  /* get the parent and the operator */
  *parent = node_id / 3;
  *expand_op = node_id % 3;
}
