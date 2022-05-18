//#include "mp.h"
//#include "mp_solve.h"
//#include "mp_aux.h"
//#include "mp_probe.h"
//
///* optimzed */
//void mp_pattern_multisample_merge
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
//  MPInt p = 0;
//  MPInt i = 0;
//
//  MPInt status = 0;
//
//  MPInt stride = context->probing.average_multipath.stride;
//  //MPInt *temp_array = context->memory_probing;
//  MPInt *endpoints = context->probing.average_multipath.endpoints_array;
//  MPInt *operator_path = context->probing.average_multipath.operator_path;
//  MPInt depth = context->n_levels;
//
//  context->runtime_contract = 0.0;
//  context->runtime_expand = 0.0;
//
//  sparse_index_base_t index;
//  MPInt nr, nc;
//
//  double *d = NULL;
//
//  struct timespec start;
//  struct timespec finish;
//
//  context->m_P = context->m_A;
//  if (context->P == NULL)
//  {
//    context->P = mp_malloc(sizeof(MPPatternCsr));
//  }
//
//  context->A.csr.m = context->m_A;
//  context->A.csr.nz = context->nz_A;
//  mp_sparse_d_csr_allocate(context->m_A, context->nz_A, &A);
//  mp_sparse_d_csr_allocate(context->m_A, context->nz_A, &B);
//  mp_sparse_d_csr_allocate(context->m_A, context->nz_A, &C);
//
//  for (i = 0; i < context->probe_iterations; ++i)
//  {
//    context->m_P = context->m_A;
//
//    /* generates list of sampling operators that is used */
//    mp_multipath_path_unpack(endpoints[i], stride, operator_path, depth);
//    #if MP_PRINTOUT
//      printf("i: %d\n", i);
//      printf("endpoint[%d]: %d -> %d, depth: %d\n", i,
//        endpoints[i], operator_path[0], depth);
//    #endif
//
//    /* contract */
//    clock_gettime(CLOCK_MONOTONIC, &start);
//    switch(operator_path[0])
//    {
//      case 0:
//        mp_sample_contract_dynamic(context, &context->A.csr, &B, 0);
//        mp_sample_contract_dynamic(context, &context->A.csr, &C, 0);
//      case 1:
//        mp_sample_contract_dynamic(context, &context->A.csr, &B, 0);
//        mp_sample_contract_dynamic(context, &context->A.csr, &C, 1);
//      case 2:
//        mp_sample_contract_dynamic(context, &context->A.csr, &B, 1);
//        mp_sample_contract_dynamic(context, &context->A.csr, &C, 1);
//      default:
//        break;
//    }
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    context->runtime_contract += mp_time(start, finish);
//    #if MP_PRINTOUT
//      printf("operator_path[%d]: %d\n", i, operator_path[i]);
//      printf("B.m: %d, C.m: %d, %d\n", B.m, C.m, context->m_P);
//    #endif
//
//    context->m_P = (context->m_P+stride-1)/stride;
//
//    /* expand */
//    clock_gettime(CLOCK_MONOTONIC, &start);
//    status = mkl_sparse_d_create_csr(&A_h, INDEXING, context->m_P,
//      context->m_P, A.rows_start, A.rows_end, A.cols, A.data);
//    status = mkl_sparse_d_create_csr(&B_h, INDEXING, context->m_P,
//      context->m_P, B.rows_start, B.rows_end, B.cols, B.data);
//    status = mkl_sparse_d_create_csr(&C_h, INDEXING, context->m_P,
//      context->m_P, C.rows_start, C.rows_end, C.cols, C.data);
//    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B_h, C_h,
//      &context->A_h);
//    status = mkl_sparse_order(context->A_h);
//    mkl_sparse_destroy(B_h);
//    mkl_sparse_destroy(C_h);
//    if (operator_path[0] == 1)
//    {
//      /* add the transpose */
//      mkl_sparse_copy(A_h, context->A_descr, &C_h);
//      mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, A_h, 1.0, C_h,
//        (sparse_matrix_t*)&context->P_handle);
//    }
//    clock_gettime(CLOCK_MONOTONIC, &finish);
//    context->runtime_expand += mp_time(start, finish);
//
//    for (p = 1; p < depth; ++p)
//    {
//      #if MP_PRINTOUT
//        printf("\n\n >> p: %d\n\n", p);
//      #endif
//      status = mkl_sparse_d_export_csr(
//        A_h, &index, &A.m, &nc,
//        &A.rows_start,
//        &A.rows_end,
//        &A.cols,
//        (double**)&A.data);
//
//      /* contract */
//      clock_gettime(CLOCK_MONOTONIC, &start);
//      switch(operator_path[0])
//      {
//        case 0:
//          mp_sample_contract_dynamic(context, &A, &B, 0);
//          mp_sample_contract_dynamic(context, &A, &C, 0);
//        case 1:
//          mp_sample_contract_dynamic(context, &A, &B, 0);
//          mp_sample_contract_dynamic(context, &A, &C, 1);
//        case 2:
//          mp_sample_contract_dynamic(context, &A, &B, 1);
//          mp_sample_contract_dynamic(context, &A, &C, 1);
//        default:
//          break;
//      }
//      clock_gettime(CLOCK_MONOTONIC, &finish);
//      context->runtime_contract += mp_time(start, finish);
//
//      /* expand */
//      status = mkl_sparse_d_create_csr(&B_h, INDEXING, context->m_P,
//        context->m_P, B.rows_start, B.rows_end, B.cols, B.data);
//      status = mkl_sparse_d_create_csr(&C_h, INDEXING, context->m_P,
//        context->m_P, C.rows_start, C.rows_end, C.cols, C.data);
//      mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B_h, C_h,
//        &A_h);
//      status = mkl_sparse_order(A_h);
//      mkl_sparse_destroy(B_h);
//      mkl_sparse_destroy(C_h);
//      if (operator_path[0] == 1)
//      {
//        /* add the transpose */
//        status = mkl_sparse_copy(A_h, context->A_descr, &C_h);
//        status = mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, A_h, 1.0, C_h,
//          &B_h);
//        status = mkl_sparse_copy(B_h, context->A_descr, &A_h);
//      }
//      clock_gettime(CLOCK_MONOTONIC, &finish);
//      context->runtime_expand += mp_time(start, finish);
//    }
//
//    status = mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, A_h, 1.0, C_h,
//      (sparse_matrix_t*)&context->P_handle);
//
//    status = mkl_sparse_d_export_csr(
//      context->P_handle, &index, &nr, &nc,
//      &((MPPatternCsr*)context->P)->rows_start,
//      &((MPPatternCsr*)context->P)->rows_end,
//      &((MPPatternCsr*)context->P)->cols,
//      (double**)&d);
//
//    context->runtime_probing = context->runtime_contract
//      + context->runtime_expand;
//  }
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
