#include "mpf.h"

///* --------------------- sparse matrix approximation ------------------------ */
//
//void mp_batch_d_spai
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Sparse *fA
//)
//{
//  MPF_Int n_max_B = context->n_max_B;
//  MPF_Int n_batches = (n_max_B + context->batch - 1) / context->batch;
//  MPF_Int nnz = 0;
//
//  MPF_Int *col = mp_malloc((sizeof *col)*context->m_B);
//  MPF_Int *col_rev = mp_malloc((sizeof *col_rev)*context->m_B);
//
//  mp_inverse_xy_adapt_partial_init(context, col, col_rev, &P);
//  if (context->data_type == MPF_REAL)
//  {
//    /* map B and X to pre allocated outer memory */
//    context->B = context->memory_outer;
//    context->X = (void*)&((double*)context->B)[m_B*blk];
//    if (context->reduction_type == MPF_OPERATION_REDUCTION_ASSIGN)
//    {
//      context->buffer = context->X;
//    }
//    double *B = context->B;
//
//    for (MPF_Int i = 0; i < n_batches; ++i)
//    {
//      /* update current rhs and block_size */
//      MPF_Int current_rhs = blk*i;
//      MPF_Int current_blk = (1-i/(n_batches-1))*context->batch
//        +(i/(n_batches-1))*(n_max_B-current_rhs);
//
//      /* generate B */
//      mp_d_generate_B(context->blk_max_fA, context->colorings_array,
//        current_rhs, context->B.m, current_blk, B.data);
//
//      /* initialize X */
//      mp_zeros_d_set(context->B.layout, context->B.m, current_blk, context->X,
//        context->X.m);
//
//      /* solve AX = B */
//      context->solver_inner_function(&context, A, &context->B, &context->X);
//
//      /* gather entries of diag_fA (symmetric reconstruction) */
//      mpf_inverse_partial_reconstruct_adapt(context, current_rhs, current_blk,
//        &context->color_to_node_map, &A, &nnz, col, col_rev);
//    }
//    fA->m = A->m;
//    fA->n = A->n;
//  }
//
//  /* sort nonzero entries in fA (edges) */
//  {
//    MPF_Int start = 0;
//    MPF_Int end = 0;
//    MPF_HeapMin_Fibonacci *T = &context->heap.fibonacci;
//    for (MPF_Int i = 0; i < diag_fA->m; ++i)
//    {
//      start = context->fA.csr.rows_start[i];
//      end = context->fA.csr.rows_end[i]-1;
//      mp_id_heapsort(T, end-start+1, &context->fA.csr.cols[start],
//        &((double*)context->fA.csr.data)[start], context->mem_outer);
//    }
//    T = NULL;
//  }
//
//  mp_free(col);
//  mp_free(col_rev);
//}
//
//void mp_batch_matrix_solve
//(
//  MPContext *context
//)
//{
//  MPInt i = 0; MPInt j = 0;
//  MPInt current_blk = 0;
//  MPInt current_rhs = 0;
//  MPInt n_max_B = context->n_max_B;
//  MPInt blk = context->blk_solver;
//  MPInt n_blocks = context->n_blocks;
//  MPInt m_B = context->m_B;
//  MPInt nnz = 0;
//  MPInt *col = mp_malloc((sizeof *col)*context->m_B);
//  MPInt *col_rev = mp_malloc((sizeof *col_rev)*context->m_B);
//
//  MPPatternCsr P;
//  mp_csr_sparse_to_pattern_convert(&context->A.csr, &P);
//
//  /* @GENERALIZE: if an other solver is used modify this and add another meta
//     structure in the MPSolverMeta union */
//  KrylovMeta meta = context->meta_solver.krylov;
//
//  context->n_blocks_solver
//    = (MPInt)((double)context->n_max_B/(double)context->blk_solver+0.5);
//    //= (context->n_max_B+context->blk_solver-1)/context->blk_solver;
//  n_blocks = context->n_blocks_solver;
//
//  mp_inverse_xy_adapt_partial_init(context, col, col_rev, &P);
//
//  /* map B and X to pre allocated outer memory */
//  MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//  context->B = context->memory_outer;
//  context->X = (void *) &((MPComplexDouble *)context->B)[m_B*blk];
//  if (context->reduction_type == MP_OPERATION_REDUCTION_ASSIGN)
//  {
//    context->buffer = context->X;
//  }
//  MPComplexDouble *B = context->B;
//  MPComplexDouble *X = context->X;
//
//  for (i = 0; i < n_blocks; ++i)
//  {
//    /* update current rhs and block_size */
//    current_rhs = blk*i;
//    current_blk = (1-i/(n_blocks-1))*blk
//      + (i/(n_blocks-1))*(n_max_B-current_rhs);
//    meta.blk = current_blk;
//
//    /* generate B and initialize X */
//    mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//      current_rhs, m_B, current_blk, B);
//    mp_zeros_z_set(context->layout_B, m_B, current_blk,
//      context->X, m_B);
//
//    /* solve AX = B */
//    context->solver_inner_function(meta, context->A_descr, context->A_handle,
//      context->m_B, context->B, context->X, context->memory_inner, NULL);
//
//    /* gather entries of diag_fA */
//    mp_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//      m_B, X, current_rhs, current_blk);
//    for (j = 0; j < current_blk; ++j)
//    {
//      mp_zaxpy(m_B, &ONE_C, &((MPComplexDouble*)context->buffer)[m_B*j], 1,
//        context->diag_fA, 1);
//    }
//  }
//
//  /* sort nonzero entries in fA (edges) */
//  {
//    MPInt start = 0;
//    MPInt end = 0;
//    MPHeapMin_Fibonacci *T = &context->heap.fibonacci;
//    for (i = 0; i < m_B; ++i)
//    {
//      start = context->fA.csr.rows_start[i];
//      end = context->fA.csr.rows_end[i]-1;
//      mp_id_heapsort(T, end-start+1, &context->fA.csr.cols[start],
//        &((double*)context->fA.csr.data)[start], context->memory_outer);
//    }
//    T = NULL;
//  }
//
//  mp_free(col);
//  mp_free(col_rev);
//}
