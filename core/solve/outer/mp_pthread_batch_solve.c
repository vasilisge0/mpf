// #include "mpf.h"
//
// /* ------------------------- pthread batch solvers -------------------------- */
//
// void mpf_batch_d_solve_pthread
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   pthread_mutex_init(&mutex, NULL);
//   n_batches = (context->n_max_B+context->batch-1)/context->batch;
//
//   /* debug information */
//   #if MP_PRINTOUT
//     printf("n_blocks: %d\n", (int)n_blocks);
//     printf("n_colors: %d\n", context->n_colors);
//     printf(" n_max_b: %d\n", (int)context->n_max_B);
//     printf("     blk: %d\n", (int)context->blk_solver);
//     printf("     m_B: %d\n", (int)m_B);
//     printf("bytes_inner: %d\n", (int)context->bytes_inner);
//     printf("offset: %d\n", context->probing.blocking.offset);
//   #endif
//
//   /* map B and X to pre allocated outer memory */
//   context->B.data = context->mem_outer;
//   context->X.data = (void*)&((double*)context->B.data)[context->B.m*context->B.n];
//   context->buffer = context->X;
//
//   /* initializes thread kernel input */
//   MPPthreadInputsSolver t_input[context->n_threads];
//
//   for (MPF_Int i = 0; i < context->n_threads; ++i)
//   {
//     t_input[i].pthread_id = -1;
//     t_input[i].mp_thread_id = i;
//     t_input[i].layout_B = context->layout_B;
//     t_input[i].meta = context->meta_solver.krylov;
//     t_input[i].m_B = context->m_B;
//     t_input[i].blk = context->blk_solver;
//     t_input[i].blk_fA = context->blk_fA;
//     t_input[i].blk_max_fA = context->blk_max_fA;
//     t_input[i].n_blocks = n_blocks;
//     t_input[i].n_threads = n_threads;
//     t_input[i].n_max_B = context->n_max_B;
//     t_input[i].A_descr = context->A_descr;
//     //t_input[i].A_handle = context->A_handle;
//     mkl_sparse_copy(context->A_handle, context->A_descr,
//       &t_input[i].A_handle);
//     t_input[i].solve_func = context->solver_inner_function;
//     t_input[i].memory_colorings = context->memory_colorings;
//     t_input[i].B = mp_malloc(context->bytes_outer);
//     t_input[i].X = &((double*)t_input[i].B)[context->m_B*blk];
//     t_input[i].buffer = mp_malloc(sizeof(double)*context->m_B*context->blk_solver);
//     t_input[i].diag_fA = mp_malloc(context->bytes_fA_data);
//
//     mp_matrix_d_set(MP_COL_MAJOR, context->m_A*context->blk_fA, 1,
//       t_input[i].diag_fA, 1, 0.0);
//
//     t_input[i].memory_inner = mp_malloc(context->bytes_inner);
//   }
//
//   /* parallel execution of inner solver */
//   struct timespec start;
//   struct timespec finish;
//   clock_gettime(CLOCK_MONOTONIC, &start);
//   for (MPF_Int i = 0; i < n_threads; ++i)
//   {
//     pthread_create
//     (
//       &t_input[i].pthread_id,
//       NULL,
//       mp_pthread_kernel_d_solve,
//       &t_input[i]
//     );
//   }
//
//   /* join threads */
//   for (MPF_Int i = 0; i < context->n_threads; ++i)
//   {
//     pthread_join(t_input[i].pthread_id, NULL);
//   }
//   clock_gettime(CLOCK_MONOTONIC, &finish);
//   context->solver.runtime = mp_time(start, finish);
//
//   /* reduction (serial first) */
//   for (MPF_Int i = 0; i < context->n_threads; ++i)
//   {
//     mp_daxpy(context->B.m*context->diag_fA.n, 1.0, t_input[i].diag_fA.data,
//       1, context->diag_fA.data, 1);
//   }
//
//   mp_diag_d_sym2gen(diag_fA->m, diag_fA->n, context->n_colors,
//     context->blk_max_fA, context->blk_fA, context->diag_fA.data,
//     context->colorings_array);
//
//   for (MPF_Int i = 0; i < context->n_threads; ++i)
//   {
//     mp_free(t_input[i].B.data);
//     mp_free(t_input[i].buffer);
//     mp_free(t_input[i].diag_fA.data);
//     mp_free(t_input[i].mem_inner);
//     mkl_sparse_destroy(t_input[i].A.handle);
//   }
//
//   pthread_mutex_destroy(&mutex);
// }
//
// void mp_batch_z_solve_pthread
// (
//   MPContext *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   pthread_mutex_init(&mutex, NULL);
//   n_batches = (context->n_max_B+context->batch-1)/context->batch;
//
//   /* debug information */
//   #if MP_PRINTOUT
//     printf("n_blocks: %d\n", (int)n_blocks);
//     printf("n_colors: %d\n", context->n_colors);
//     printf(" n_max_b: %d\n", (int)context->n_max_B);
//     printf("     blk: %d\n", (int)context->blk_solver);
//     printf("     m_B: %d\n", (int)m_B);
//     printf("bytes_inner: %d\n", (int)context->bytes_inner);
//     printf("offset: %d\n", context->probing.blocking.offset);
//   #endif
//
//   /* map B and X to pre allocated outer memory */
//   MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//   context->B.data = context->mem_outer;
//   context->X.data = (void*)&((MPComplexDouble *)context->B)[context->B.m*context->B.n];
//   context->buffer = context->X;
//
//   for (MPF_Int i = 0; i < n_blocks; ++i)
//   {
//     /* update current rhs and block_size */
//     MPF_Int current_rhs = context->batch*i;
//     MPF_Int current_blk = (1-i/(n_blocks-1))*context->batch
//       + (i/(n_blocks-1))*(context->n_max_B-current_rhs);
//
//     /* generate B */
//     mp_z_generate_B(context->blk_max_fA, context->colorings_array,
//       current_rhs, context->B.m, current_blk, context->B.data);
//
//     /* intiialize X */
//     mp_zeros_z_set(context->B.layout, context->X.m, current_blk,
//       context->X.data, context->X.m);
//
//     /* solve AX = B */
//     context->solver_inner_function(context, A, context->B, context->X);
//
//     /* gather entries of diag_fA */
//     mp_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//       m_B, X, current_rhs, current_blk);
//     for (MPF_Int j = 0; j < current_blk; ++j)
//     {
//       mp_zaxpy(context->diag_fA.m, &ONE_C, &((MPComplexDouble*)context->buffer)[context->diag_fA.m*j],
//         1, context->diag_fA.data, 1);
//     }
//   }
//
//   pthread_mutex_destroy(&mutex);
// }
//
// void *mp_pthread_kernel_d_solve
// (
//   void *input_packed
// )
// {
//   /* update current rhs and block_size */
//   MPInt i = 0;
//   MPInt j = 0;
//   MPPthreadInputsSolver *input = input_packed;
//   MPInt current_rhs = 0;
//   MPInt current_blk = 0;
//   MPInt blk = input->meta.blk;
//   MPInt n_blocks = input->n_blocks;
//   MPInt mp_thread_id = input->mp_thread_id;
//
//   struct timespec start;
//   struct timespec finish;
//   double elapsed = 0.0;
//
//   MPInt range_max = (n_blocks-1)/input->n_threads+1;
//   MPInt rhs_start = range_max*(mp_thread_id);
//   MPInt range_min = n_blocks - rhs_start;
//   MPInt range = (1-(input->mp_thread_id+1)/input->n_threads)*range_max + ((input->mp_thread_id+1)/input->n_threads)*range_min;
//   MPInt rhs_end = rhs_start + range;
//
//   int thread_clock_id;
//   pthread_getcpuclockid(pthread_self(), &thread_clock_id);
//
//   for (i = rhs_start; i < rhs_end; ++i)
//   {
//     current_rhs = blk*i;
//     current_blk = (1-(i+1)/(n_blocks))*blk
//       + ((i+1)/(n_blocks))*(input->n_max_B-current_rhs);
//     input->meta.blk = current_blk;
//
//     /* generate B and initialize X */
//     mp_d_generate_B(input->blk_max_fA, input->memory_colorings,
//       current_rhs, input->m_B, current_blk, input->B);
//     mp_zeros_d_set(input->layout_B, input->m_B, current_blk, input->X, input->m_B);
//
//     /* solve AX = B */
//     clock_gettime(thread_clock_id, &start);
//     input->solve_func(input->meta, input->A_descr, input->A_handle,
//       input->m_B, input->B, input->X, input->memory_inner, NULL);
//     clock_gettime(thread_clock_id, &finish);
//     elapsed = mp_time(start, finish);
//
//     /* gather entries of diag_fA */
//     mp_d_blk_xy_select_X_dynamic(input->blk_max_fA, input->memory_colorings,
//       input->m_B, input->X, input->blk_fA, current_rhs, current_blk);
//
//     for (j = 0; j < current_blk; ++j)
//     {
//       mp_daxpy(input->m_B, 1.0, &((double*)input->X)[input->m_B*j], 1,
//         &((double*)input->diag_fA)[input->m_B*((j+current_rhs)%input->blk_fA)], 1);
//     }
//     #if MP_PRINTOUT
//       printf("[(id: %d, i: %d in [%d, %d]) | %1.2e\n",
//         mp_thread_id, i, rhs_start, rhs_end, elapsed);
//     #endif
//   }
//
//   return NULL;
// }
