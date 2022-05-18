// #include "mpf.h"
//
// /*============================================================================*/
// /* mp_batch_2pass_solve                                                       */
// /* batch_solve method that parses twice the main loop to use diagonal or      */
// /* diagonal block preconditioner                                              */
// /*============================================================================*/
// void mp_batch_d_2pass
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Dense diag_fA
// )
// {
//   MPF_Int n_batches = (context->n_max_B+context->batch-1)/context->batch;
//
//   /* solver info */
//   #if mpf_printout
//     printf("  n_batches: %d\n", (int)n_batches);
//     printf("    n_max_b: %d\n", (int)context->n_max_b);
//     printf("        blk: %d\n", (int)context->blk_solver);
//     printf("        m_b: %d\n", (int)m_b);
//     printf("bytes_inner: %d\n", (int)context->bytes_inner);
//     printf("     offset: %d\n", context->probing.blocking.offset);
//   #endif
//
//   /* maps b, x and buffer to pre allocated outer memory */
//   context->b.data = context->mem_outer;
//   context->x.data = (void*)&((double*)context->b)[context->b.m*context->b.n];
//   context->buffer = context->x;
//
//   for (mpf_int i = 0; i < n_batches; ++i)
//   {
//     /* updates current rhs and block_size */
//     mpf_int current_rhs = blk*i;
//     mpf_int current_blk = (1-(i+1)/(n_batches))*context->batch
//       +((i+1)/(n_batches))*(n_max_b-current_rhs);
//
//     /* generates b */
//     mp_d_generate_b(context->blk_max_fa, context->colorings_array,
//       current_rhs, context->b.m, current_blk, context->b.data);
//
//     /* initializes x */
//     mp_zeros_d_set(context->b.layout, context->x.m, current_blk, context->x.data, context->x.m);
//
//     /* solves ax = b */
//     context->inner_function(a, &context->b, &context->x);
//
//     /* selects specified entries of x */
//     mp_d_blk_xy_select_x_dynamic(context->blk_max_fa,
//       context->colorings_rray, context->x.m, context->x.data, context->blk_fa,
//       current_rhs, current_blk);
//
//     /* gathers entries of diag_fa */
//     for (mpf_int j = 0; j < current_blk; ++j)
//     {
//       mp_daxpy(context->b.m, 1.0, &((double*)context->buffer)[context->b.m*j], 1,
//         &((double*)context->diag_fa.data)[context->diag_fa.m*((j+current_rhs)%context->blk_fa)], 1);
//     }
//   }
//
//   /* converts diagonal blocks from symmetric to general format */
//   mp_diag_d_sym2gen(context->b.m, context->blk_fa, context->n_colors,
//     context->blk_max_fa, context->blk_fa, context->diag_fa.data,
//     context->colorings_array);
//
//   ///* add here preconditioner initialization */
//   //if (context->meta_solver.krylov.precond_type != mpf__precond_none)
//   //{
//   //  //mp_precond_csr_diag_blocks_init(context->m_b, context->blk_fa,
//   //  //  context->diag_fa, &context->meta_solver.krylov.m.csr);
//
//   //  printf("context->diag_fa[0]: %1.4e\n", ((double*)context->diag_fa)[0]);
//   //  printf("\n\n stop \n\n");
//
//   //  /* second pass */
//   //  for (i = 0; i < n_blocks; ++i)
//   //  {
//   //    /* updates current rhs and block_size */
//   //    current_rhs = blk*i;
//   //    current_blk = (1-(i+1)/(n_blocks))*blk
//   //      +((i+1)/(n_blocks))*(n_max_b-current_rhs);
//   //    meta.blk = current_blk;
//
//   //    /* generates b */
//   //    mp_d_generate_b(context->blk_max_fa, context->memory_colorings,
//   //      current_rhs, m_b, current_blk, context->b);
//
//   //    /* initializes x */
//   //    vdmul(m_b*current_blk, context->b, context->diag_fa, context->x);
//   //    //mp_zeros_d_set(context->layout_b, m_b, current_blk, context->x, m_b);
//
//   //    /* solves ax = b */
//   //    //printf("solving ax = b, meta.blk: %d\n", meta.blk);
//   //    //context->solver_inner_function(context->meta_solver, context->a_descr,
//   //    //  context->a_handle, context->m_b, context->b, context->x,
//   //    //  context->memory_inner, null);
//
//   //    mp_solve_global_pcg_dsy_constrained(context->meta_solver.krylov, context->a_descr,
//   //      context->a_handle, context->m_b, context->b, context->x,
//   //      context->memory_inner, null);
//
//   //    /* selects specified entries of x */
//   //    mp_d_blk_xy_select_x_dynamic(context->blk_max_fa,
//   //      context->memory_colorings, m_b, context->x, context->blk_fa,
//   //      current_rhs, current_blk);
//
//   //    /* gathers entries of diag_fa */
//   //    for (j = 0; j < current_blk; ++j)
//   //    {
//   //      mp_daxpy(m_b, 1.0, &((double*)context->buffer)[m_b*j], 1,
//   //        &((double*)context->diag_fa)[m_b*((j+current_rhs)%blk_fa)], 1);
//   //    }
//   //  }
//
//   //  /* converts diagonal blocks from symmetric to general format */
//   //  mp_diag_d_sym2gen(context->m_B, context->blk_fA, context->n_colors,
//   //    context->blk_max_fA, context->blk_fA, context->diag_fA,
//   //    context->memory_colorings);
//   //}
// }
//
// void mp_batch_z_2pass
// (
//   MPContext *context
// )
// {
//   MPInt i = 0;
//   MPInt j = 0;
//   MPInt current_blk = 0;
//   MPInt current_rhs = 0;
//   MPInt n_max_B = context->n_max_B;
//   MPInt blk = context->blk_solver;
//   MPInt n_blocks = context->n_blocks;
//   MPInt m_B = context->m_B;
//   MPInt blk_fA = context->blk_fA;
//
//   /* @GENERALIZE: if an other type of solver is used modify this and add another
//      meta structure in the MPSolverMeta union in /include/mp.h */
//
//   KrylovMeta meta = context->meta_solver.krylov;
//
//   context->n_blocks_solver
//     = (context->n_max_B+context->blk_solver-1)/context->blk_solver;
//   n_blocks = context->n_blocks_solver;
//
//   /* solver info */
//
//   printf("   n_blocks: %d\n", (int)n_blocks);
//   printf("    n_max_b: %d\n", (int)context->n_max_B);
//   printf("        blk: %d\n", (int)context->blk_solver);
//   printf("        m_B: %d\n", (int)m_B);
//   printf("bytes_inner: %d\n", (int)context->bytes_inner);
//   printf("     offset: %d\n", context->probing.blocking.offset);
//
//   /* map B and X to pre allocated outer memory */
//
//   MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//   context->B = context->memory_outer;
//   context->X = (void*)&((MPComplexDouble*)context->B)[m_B*blk];
//   if (context->reduction_type == MP_OPERATION_REDUCTION_ASSIGN)
//   {
//     context->buffer = context->X;
//   }
//   MPComplexDouble *B = context->B;
//   //MPComplexDouble *X = context->X;
//
//   for (i = 0; i < n_blocks; ++i)
//   {
//     /* update current rhs and block_size */
//     current_rhs = blk*i;
//     current_blk
//       = (1-i/(n_blocks-1))*blk + (i/(n_blocks-1))*(n_max_B-current_rhs);
//     meta.blk = current_blk;
//
//     /* generates B  */
//     mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//       current_rhs, m_B, current_blk, B);
//
//     /* initializes X */
//     mp_zeros_z_set(context->layout_B, m_B, current_blk,
//       context->X, m_B);
//
//     /* solves AX = B */
//     context->solver_inner_function(meta, context->A_descr, context->A_handle,
//       context->m_B, context->B, context->X, context->memory_inner, NULL);
//
//     /* selects specified entries of X */
//     mp_z_blk_xy_select_X_dynamic(context->blk_max_fA,
//       context->memory_colorings, m_B, context->X, context->blk_fA,
//       current_rhs, current_blk);
//
//     /* gathers entries of diag_fA */
//     for (j = 0; j < current_blk; ++j)
//     {
//       mp_zaxpy(m_B, &ONE_C, &((MPComplexDouble*)context->buffer)[m_B*j], 1,
//         &((MPComplexDouble*)context->diag_fA)[m_B*((j+current_rhs)%blk_fA)],1);
//     }
//   }
//
//   /* converts diagonal blocks from symmetric to general format */
//   mp_diag_z_sym2gen(context->m_B, context->blk_fA, context->n_colors,
//     context->blk_max_fA, context->blk_fA, context->diag_fA,
//     context->memory_colorings);
// }
