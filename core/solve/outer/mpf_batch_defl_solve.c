// #include "mpf.h"
//
// void mpf_batch_d_defl_solve
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   MPF_Int n_batches = (MPF_Int)((double)context->max_n_rhs/(double)context->batch+0.5);
//
//   printf("mpf_batch_d_defl_solve\n");
//   printf("----------------------\n");
//   printf("n_blocks: %d\n", (int)context->n_blocks);
//   printf(" n_max_b: %d\n", (int)context->n_max_B);
//   printf("     blk: %d\n", (int)context->blk_solver);
//   printf("     m_B: %d\n", (int)m_B);
//
//   /* map B and X to pre allocated outer memory */
//   context->B.data = context->mem_outer;
//   context->X.data = (void*)&((double*)context->B.data)[context->B.m*context->B.n];
//   context->buffer = context->X.data;  /* temp accumulator target */
//
//   /* generate B and initialize X */
//   mpf_d_generate_B(context->max_blk_fA, context->colorings_array,
//     current_rhs, context->B.m, context->B.n_max, context->B.data);
//   mpf_zeros_d_set(context->B.layout, context->B.m, context->B.n_max,
//     context->X.data, context->B.m);
//
//   for (MPF_Int i = 0; i < n_blocks; ++i) /* @DEBUG: works, needs to add seeding */
//   {
//     /* update current rhs and block_size */
//     MPF_Int current_rhs = context->batch*i;
//     MPF_Int current_batch = (1-i/(n_blocks-1))*context->batch
//       +(i/(n_blocks-1))*(context->max_n_rhs-current_rhs);
//     context->batch = current_batch;
//
//     /* solve AX = B */
//     context->solver_inner_function(context, A, &context->B, &context->X);
//
//     /* sparsify X */
//     context->buffer = &context->X.data[context->X.m*current_rhs];
//     mpf_d_select_X_dynamic(context->max_blk_fA, context->colorings_array,
//       context->X.m, context->buffer, current_rhs, current_blk);
//
//     /* gather entries of diag_fA */
//     for (MPF_Int j = 0; j < current_blk; ++j)
//     {
//       mpf_daxpy(context->B.m, 1.0, &((double*)context->buffer)[context->B.m*j], 1,
//         diag_fA.data, 1);
//     }
//   }
// }
//
// void mpf_batch_z_defl_solve
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   MPF_Int n_blocks = (MPF_Int)((double)context->n_max_B /(double)context->batch+0.5);
//
//   printf("n_blocks: %d\n", (int)context->n_blocks);
//   printf(" n_max_b: %d\n", (int)context->n_max_B);
//   printf("     blk: %d\n", (int)context->blk_solver);
//   printf("     m_B: %d\n", (int)m_B);
//   printf("bytes_inner: %d\n", (int)context->bytes_inner);
//
//   /* map B and X to pre allocated outer memory */
//   MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
//   context->B.data = context->mem_outer;
//   context->X.data = (void *) &((MPF_ComplexDouble *)context->B.data)[context->B.m*context->B.n];
//   context->buffer = context->X;
//
//   for (MPF_Int i = 0; i < n_blocks; ++i)
//   {
//     /* update current rhs and block_size */
//     MPF_Int current_rhs = blk*i;
//     MPF_Int current_blk = (1-i/(n_blocks-1))*blk
//       + (i/(n_blocks-1))*(n_max_B-current_rhs);
//
//     /* generate B */
//     mpf_z_generate_B(context->blk_max_fA, context->colorings_array,
//       current_rhs, context->B.m, current_blk, context->B.data);
//
//     /* initialize X */
//     mpf_zeros_z_set(context->B.layout, context->B.m, current_blk,
//       context->X, context->B.m);
//
//     /* solve AX = B */
//     context->solver_inner_function(context, A, &context->B, &context->X);
//
//     /* gather entries of diag_fA */
//     mpf_z_select_X_dynamic(context->blk_max_fA, context->colorings_array,
//       context->X.m, X, current_rhs, current_blk);
//
//     for (MPF_Int j = 0; j < current_blk; ++j)
//     {
//       mpf_zaxpy(m_B, &ONE_C, &((MPF_ComplexDouble*)context->buffer)[context->B.m*j], 1,
//         context->diag_fA.data, 1);
//     }
//   }
// }
//
// void mpf_batch_eig_defl_solve
// (
//   MPF_Context *context
// )
// {
//   MPF_Int current_blk = 0;
//   MPF_Int current_rhs = 0;
//
//   /* @GENERALIZE: if an other solver is used modify this and add another meta
//      structure in the MPF_SolverMeta union */
//   KrylovMeta meta = context->meta_solver.krylov;
//
//   context->n_blocks_solver
//     = (MPF_Int)((double)context->n_max_B/(double)context->blk_solver+0.5);
//   n_blocks = context->n_blocks_solver;
//
//   printf("   n_blocks: %d\n", (int)context->n_blocks);
//   printf("    n_max_b: %d\n", (int)context->n_max_B);
//   printf("        blk: %d\n", (int)context->blk_solver);
//   printf("        m_B: %d\n", (int)m_B);
//   printf("bytes_inner: %d\n", (int)context->bytes_inner);
//   printf("    n_max_B: %d\n", context->n_max_B);
//
//   if (context->data_type == MPF_REAL)
//   {
//     /* map B and X to pre allocated outer memory */
//     context->B = context->memory_outer;
//     context->X = (void*)&((double*)context->B)[m_B*context->n_max_B];
//     if (context->reduction_type == MPF_OPERATION_REDUCTION_ASSIGN)
//     {
//       context->buffer = context->X;
//     }
//
//     MPF_Int m_H = meta.iterations;
//     MPF_Int n_H = meta.iterations;
//     double *B = context->B;
//     double *X = context->X;
//
//     /* unpacks defl_memory */
//     double *Vdefl = context->memory_defl;
//     double *Hdefl = &Vdefl[m_B*blk*meta.iterations];
//     double *refs_defl_array = &Hdefl[m_H*n_H];
//     double *Tdefl = &refs_defl_array[m_H-1];
//     //double *Mdefl = &Tdefl[m_B*meta.blk];
//
//
//     /* @OPTIMIZE */
//     double *S = mpf_malloc((sizeof *S)*m_B*blk);
//     double *Bt = mpf_malloc((sizeof *Bt)*m_B*blk);
//
//     /* generate B and initialize X */
//     mpf_d_generate_B(context->blk_max_fA, context->memory_colorings,
//       current_rhs, m_B, context->n_max_B, B);
//     mpf_zeros_d_set(context->layout_B, m_B, context->n_max_B, context->X, m_B);
//
//     for (i = 0; i < n_blocks; ++i)
//     {
//       /* update current rhs and block_size */
//       current_rhs = blk*i;
//       current_blk = (1-i/(n_blocks-1))*blk
//         +(i/(n_blocks-1))*(n_max_B-current_rhs);
//       meta.blk = current_blk;
//
//       memcpy(Bt, &((double*)context->B)[m_B*i], (sizeof *Tdefl)*context->m_A);
//
//       if (i)
//       {
//         memcpy(S, &B[m_B*current_rhs], (sizeof *S)*current_blk*m_B);
//
//         /* that seems to contribute enormously */
//         context->defl_operands.seed(context->A_handle, context->A_descr, m_B,
//           meta.iterations, meta.iterations, meta.iterations, current_blk,
//           S, context->memory_defl);
//
//         memcpy(&X[m_B*current_rhs], S, (sizeof *X)*m_B);
//
//         memcpy(Tdefl, &((double*)context->B)[m_B],(sizeof *Tdefl)*context->m_A);
//         mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
//           context->A_descr, S, 1.0, Tdefl);
//         mpf_matrix_d_announce(&B[m_B], 10, 1, m_B, "B(init)");
//       }
//
//       /* solve AX = B */
//       //if (i == 0)
//       //{
//         context->solver_inner_function(meta, context->A_descr, context->A_handle,
//           context->m_B, &B[m_B*current_rhs], &X[m_B*current_rhs],
//           context->memory_inner, i, context->defl_operands, context->n_defl,
//           context->memory_defl, NULL);
//       //}
//
//       mpf_matrix_d_announce(&X[m_B*current_rhs], 20, 1, m_B, "(out) X");
//       mpf_matrix_d_announce(S, 20, 1, m_B, "(out) S");
//
//       /* apply seeding to rhs to update X */
//       if (i)
//       {
//         memcpy(Tdefl, Bt, (sizeof *Tdefl)*m_B);
//         mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
//           context->A_descr, &X[current_rhs*m_B], 1.0, Tdefl);
//
//         mpf_daxpy(m_B*current_blk, 1.0, S, 1, &X[current_rhs*m_B], 1);
//         mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
//           context->A_descr, &X[current_rhs*m_B], 1.0, Bt);
//       }
//
//       /* sparsify X */
//       context->buffer = &X[m_B*current_rhs];
//       mpf_d_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//         m_B, context->buffer, current_rhs, current_blk);
//
//       /* gather entries of diag_fA */
//       for (j = 0; j < current_blk; ++j)
//       {
//         mpf_daxpy(m_B, 1.0, &((double*)context->buffer)[m_B*j], 1,
//           context->diag_fA, 1);
//       }
//
//       mpf_matrix_d_announce(context->diag_fA, 10, 1, m_B, "diag_fA");
//     }
//     mpf_free(S);
//     mpf_free(Bt);
//   }
//   else if (context->data_type == MPF_COMPLEX)
//   {
//     /* map B and X to pre allocated outer memory */
//     MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
//     context->B = context->memory_outer;
//     context->X = (void *) &((MPF_ComplexDouble *)context->B)[m_B*blk];
//     if (context->reduction_type == MPF_OPERATION_REDUCTION_ASSIGN)
//     {
//       context->buffer = context->X;
//     }
//     MPF_ComplexDouble *B = context->B;
//     MPF_ComplexDouble *X = context->X;
//
//     for (i = 0; i < n_blocks; ++i)
//     {
//       /* update current rhs and block_size */
//       current_rhs = blk*i;
//       current_blk = (1-i/(n_blocks-1))*blk
//         + (i/(n_blocks-1))*(n_max_B-current_rhs);
//       meta.blk = current_blk;
//
//       /* generate B and initialize X */
//       mpf_z_generate_B(context->blk_max_fA, context->memory_colorings,
//         current_rhs, m_B, current_blk, B);
//       mpf_zeros_z_set(context->layout_B, m_B, current_blk,
//         context->X, m_B);
//
//       /* solve AX = B */
//       context->solver_inner_function(meta, context->A_descr, context->A_handle,
//         context->m_B, context->B, context->X, context->memory_inner, NULL);
//
//       /* gather entries of diag_fA */
//       mpf_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//         m_B, X, current_rhs, current_blk);
//       for (j = 0; j < current_blk; ++j)
//       {
//         mpf_zaxpy(m_B, &ONE_C, &((MPF_ComplexDouble*)context->buffer)[m_B*j], 1,
//           context->diag_fA, 1);
//       }
//     }
//   }
// }
//
// void mpf_batch_eig_defl_solve
// (
//   MPF_Context *context,
//   MPF_Sparse *A
// )
// {
// //  MPF_Int n_blocks = (MPF_Int)((double)context->solver.n_max_B
// //    /(double)context->solver.batch+0.5);
// //
// //  printf("mpf_batch_eig_defl_solve\n");
// //  printf("------------------------\n");
// //  printf("   n_blocks: %d\n", (int)context->n_blocks);
// //  printf("    n_max_b: %d\n", (int)context->n_max_B);
// //  printf("        blk: %d\n", (int)context->blk_solver);
// //  printf("        m_B: %d\n", (int)m_B);
// //  printf("bytes_inner: %d\n", (int)context->bytes_inner);
// //  printf("    n_max_B: %d\n", context->n_max_B);
// //
// //  if (context->data_type == MPF_REAL)
// //  {
// //    /* map B and X to pre allocated outer memory */
// //    context->B.data = context->mem_outer;
// //    context->X.data = (void*)&((double*)context->B.data)[context->B.m*context->n_max_B];
// //    context->buffer = context->X;
// //
// //    MPF_Int m_H = meta.iterations;
// //    MPF_Int n_H = meta.iterations;
// //    double *B = context->B;
// //    double *X = context->X;
// //
// //    /* unpacks defl_memory */
// //    double *Vdefl = context->memory_defl;
// //    double *Hdefl = &Vdefl[m_B*blk*meta.iterations];
// //    double *refs_defl_array = &Hdefl[m_H*n_H];
// //    double *Tdefl = &refs_defl_array[m_H-1];
// //    //double *Mdefl = &Tdefl[m_B*meta.blk];
// //
// //
// //    /* @OPTIMIZE */
// //    double *S = mpf_malloc((sizeof *S)*m_B*blk);
// //    double *Bt = mpf_malloc((sizeof *Bt)*m_B*blk);
// //
// //    /* generate B and initialize X */
// //    mpf_d_generate_B(context->blk_max_fA, context->memory_colorings,
// //      current_rhs, m_B, context->n_max_B, B);
// //    mpf_zeros_d_set(context->layout_B, m_B, context->n_max_B, context->X, m_B);
// //
// //    for (i = 0; i < n_blocks; ++i)
// //    {
// //      /* update current rhs and block_size */
// //      current_rhs = blk*i;
// //      current_blk = (1-i/(n_blocks-1))*blk
// //        +(i/(n_blocks-1))*(n_max_B-current_rhs);
// //      meta.blk = current_blk;
// //
// //      memcpy(Bt, &((double*)context->B)[m_B*i], (sizeof *Tdefl)*context->m_A);
// //
// //      if (i)
// //      {
// //        memcpy(S, &B[m_B*current_rhs], (sizeof *S)*current_blk*m_B);
// //
// //        /* that seems to contribute enormously */
// //        context->defl_operands.seed(context->A_handle, context->A_descr, m_B,
// //          meta.iterations, meta.iterations, meta.iterations, current_blk,
// //          S, context->memory_defl);
// //
// //        memcpy(&X[m_B*current_rhs], S, (sizeof *X)*m_B);
// //
// //        memcpy(Tdefl, &((double*)context->B)[m_B],(sizeof *Tdefl)*context->m_A);
// //        mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
// //          context->A_descr, S, 1.0, Tdefl);
// //        mpf_matrix_d_announce(&B[m_B], 10, 1, m_B, "B(init)");
// //      }
// //
// //      /* solve AX = B */
// //      //if (i == 0)
// //      //{
// //        context->solver_inner_function(meta, context->A_descr, context->A_handle,
// //          context->m_B, &B[m_B*current_rhs], &X[m_B*current_rhs],
// //          context->memory_inner, i, context->defl_operands, context->n_defl,
// //          context->memory_defl, NULL);
// //      //}
// //
// //      mpf_matrix_d_announce(&X[m_B*current_rhs], 20, 1, m_B, "(out) X");
// //      mpf_matrix_d_announce(S, 20, 1, m_B, "(out) S");
// //
// //      /* apply seeding to rhs to update X */
// //      if (i)
// //      {
// //        memcpy(Tdefl, Bt, (sizeof *Tdefl)*m_B);
// //        mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
// //          context->A_descr, &X[current_rhs*m_B], 1.0, Tdefl);
// //
// //        mpf_daxpy(m_B*current_blk, 1.0, S, 1, &X[current_rhs*m_B], 1);
// //        mpf_sparse_d_mv(MPF_SPARSE_NON_TRANSPOSE, -1.0, context->A_handle,
// //          context->A_descr, &X[current_rhs*m_B], 1.0, Bt);
// //      }
// //
// //      /* sparsify X */
// //      context->buffer = &X[m_B*current_rhs];
// //      mpf_d_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
// //        m_B, context->buffer, current_rhs, current_blk);
// //
// //      /* gather entries of diag_fA */
// //      for (j = 0; j < current_blk; ++j)
// //      {
// //        mpf_daxpy(m_B, 1.0, &((double*)context->buffer)[m_B*j], 1,
// //          context->diag_fA, 1);
// //      }
// //
// //      mpf_matrix_d_announce(context->diag_fA, 10, 1, m_B, "diag_fA");
// //    }
// //    mpf_free(S);
// //    mpf_free(Bt);
// //  }
// //  else if (context->data_type == MPF_COMPLEX)
// //  {
// //    /* map B and X to pre allocated outer memory */
// //    MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
// //    context->B = context->memory_outer;
// //    context->X = (void *) &((MPF_ComplexDouble *)context->B)[m_B*blk];
// //    if (context->reduction_type == MPF_OPERATION_REDUCTION_ASSIGN)
// //    {
// //      context->buffer = context->X;
// //    }
// //    MPF_ComplexDouble *B = context->B;
// //    MPF_ComplexDouble *X = context->X;
// //
// //    for (i = 0; i < n_blocks; ++i)
// //    {
// //      /* update current rhs and block_size */
// //      current_rhs = blk*i;
// //      current_blk = (1-i/(n_blocks-1))*blk
// //        + (i/(n_blocks-1))*(n_max_B-current_rhs);
// //      meta.blk = current_blk;
// //
// //      /* generate B and initialize X */
// //      mpf_z_generate_B(context->blk_max_fA, context->memory_colorings,
// //        current_rhs, m_B, current_blk, B);
// //      mpf_zeros_z_set(context->layout_B, m_B, current_blk,
// //        context->X, m_B);
// //
// //      /* solve AX = B */
// //      context->solver_inner_function(meta, context->A_descr, context->A_handle,
// //        context->m_B, context->B, context->X, context->memory_inner, NULL);
// //
// //      /* gather entries of diag_fA */
// //      mpf_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
// //        m_B, X, current_rhs, current_blk);
// //      for (j = 0; j < current_blk; ++j)
// //      {
// //        mpf_zaxpy(m_B, &ONE_C, &((MPF_ComplexDouble*)context->buffer)[m_B*j], 1,
// //          context->diag_fA, 1);
// //      }
// //    }
// //  }
// }
