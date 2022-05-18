#include "mpf.h"

//void mpf_batch_d_dynamic_spbasis
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA
//)
//{
//  /* sets n_blocks */
//  MPF_Int n_batches = (n_max_B + context->batch - 1)/ context->batch;
//  context->blk_max_fA = mpf_blk_max_fA_get(context);
//
//  #if MPF_PRINTOUT
//    printf(" blk_max_fA: %d\n", blk_max_fA);
//    printf("   n_blocks: %d\n", (int)context->n_blocks);
//    printf("    n_max_b: %d\n", (int)context->n_max_B);
//    printf("        blk: %d\n", (int)context->blk_solver);
//    printf("        m_B: %d\n", (int)m_B);
//    printf("bytes_inner: %d\n", (int)context->bytes_inner);
//  #endif
//
//  /* map B and X to pre allocated outer memory */
//  context->B.data = context->mem_outer;
//  context->X.data = (void*)&((double*)context->B.data)[context->B.m*context->batch];
//  context->buffer = context->X;
//
//  for (MPF_Int i = 0; i < n_batches; ++i)
//  {
//    /* update rhs and block size */
//    MPF_Int current_rhs = blk*i;
//    MPF_Int current_blk = (1-i/(n_batches-1))*context->batch
//      + (i/(n_batches-1))*(n_max_B-current_rhs);
//
//    /* generate B and solve AX = B */
//    mpf_d_generate_B(blk_max_fA, context->colorings_array, current_rhs,
//      context->B.m, current_blk, context->B.data);
//
//    /* initializes X */
//    mpf_zeros_d_set(context->B.layout, context->X.m, current_blk,
//      context->X.data, context->X.m);
//
//    /* solve AX = B */
//    context->solver_inner_function(context, A, &context->B, &context->X);
//
//    /* compute B(.*)X and reduce diagonal blocks */
//    mpf_d_select_X(blk_max_fA, context->X.m, current_blk, context->B.data,
//      context->X.data, context->buffer);
//
//    for (MPF_Int j = 0; j < current_blk; ++j)
//    {
//      mpf_daxpy(context->diag_fA_.m, 1.0, &((double*)context->buffer)[context->diag_fA.m*j], 1,
//        context->diag_fA.data, 1);
//    }
//  }
//}
//
//void mpf_batch_z_dynamic_spbasis
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA
//)
//{
//  /* sets n_blocks */
//  MPF_Int n_batches = (n_max_B + context->batch - 1)/ context->batch;
//  context->blk_max_fA = mpf_blk_max_fA_get(context);
//
//  #if MPF_PRINTOUT
//    printf(" blk_max_fA: %d\n", blk_max_fA);
//    printf("   n_blocks: %d\n", (int)context->n_blocks);
//    printf("    n_max_b: %d\n", (int)context->n_max_B);
//    printf("        blk: %d\n", (int)context->blk_solver);
//    printf("        m_B: %d\n", (int)m_B);
//    printf("bytes_inner: %d\n", (int)context->bytes_inner);
//  #endif
//
//  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
//  context->B.data = context->mem_outer;
//  context->X.data = (void*)&((MPF_ComplexDouble*)context->B.data)[context->B.m*context->B.n];
//  context->buffer = context->X;
//
//  for (MPF_Int i = 0; i < n_batches; ++i)
//  {
//    /* update rhs and block size */
//    MPF_Int current_rhs = context->batch*i;
//    MPF_Int current_blk = (1-i/(n_batches-1))*blk + (i/(n_batches-1))
//      *(context->B.n-current_rhs);
//
//    /* generate B */
//    mpf_z_generate_B(blk_max_fA, context->colorings_array, current_rhs,
//      context->B.m, current_blk, context->B.data);
//
//    /* initialize X */
//    mpf_zeros_z_set(context->B.layout, context->X.m, current_blk, context->X.data,
//      context->X.m);
//
//    /* solve AX = B */
//    context->solver_inner_function(context, A, &context->B, &context->X);
//
//    /* compute B(.*)X and reduce diagonal blocks */
//    mpf_z_select_X_dynamic(context->blk_max_fA, context->colorings_array,
//      context->X.m, context->X.data, current_rhs, current_blk);
//
//    for (MPF_Int j = 0; j < current_blk; ++j)
//    {
//      mpf_zaxpy(context->diag_fA.m, &ONE_C, &((MPF_ComplexDouble*)context->buffer)[context->diag_fA.m*j], 1,
//        context->diag_fA, 1);
//    }
//  }
//}
