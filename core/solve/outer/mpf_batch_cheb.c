// #include "mpf.h"
//
// void mpf_batch_d_cheb
// (
//   MPF_Context *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   MPF_Int n_max_B = context->n_max_B;
//   MPF_Int blk = context->blk_solver;
//   MPF_Int n_blocks = context->n_blocks;
//   MPF_Int m_B = context->m_B;
//   MPF_Int blk_fA = context->blk_fA;
//   VSLStreamStatePtr stream;
//   vslNewStream(&stream, VSL_BRNG_MCG31, 1);
//   double lmax = 0.0;
//   double lmin = 0.0;
//
//   MPF_Int n_batches = (context->n_max_B + context->batch - 1)/context->batch;
//
//   /* debug information */
//   printf("n_batches: %d\n", (int)n_batches);
//   printf("  n_max_b: %d\n", (int)context->n_max_B);
//   printf("      blk: %d\n", (int)context->blk_solver);
//   printf("      m_B: %d\n", (int)m_B);
//   printf(" bytes_inner: %d\n", (int)context->bytes_inner);
//   printf(" offset: %d\n", context->probing.blocking.offset);
//
//   /* map B and X to pre allocated outer memory */
//   context->B.data = context->memory_outer;
//   context->X.data = (void*)&((double*)context->B.data)[context->B.m*context->B.n];
//   context->buffer = context->X;
//
//   double *memory_ev_min = &X.data[context->X.m*batch];
//   double *memory_ev_max = &memory_ev_min[context->X.m*batch];
//   MPF_Int M = context->cheb_ev_iterations;
//   MPF_Int iters_ev = 10;
//   context->cheb_coeffs = &((double*)context->mem_inner)[M*M];
//
//   /* computes maximum eigenvalue */
//   context->solver_ev_min_func(stream, context->meta_solver_ev_min,
//     A->descr, A->handle, A->m, memory_ev_min, &lmin, NULL);
//   context->A_lmin = lmin;
//   context->solver_ev_max_func(stream, A->descr, A->handle,
//     A->m, memory_ev_max, iters_ev, &lmax);
//   context->A_lmax = lmax;
//   printf("\n\n*** Executing chebyshev solver in [lmin: %1.4E, lmax: %1.4E]\
//     \n\n", lmin, lmax);
//
//   /* approximates chebyshev coefficients using chebyshev interpolation. Uses
//      chebyshev nodes of the first kind. */
//   mp_dsy_cheb_poly
//   (
//     context->cheb_M, /* number of points = polynomial_degree-1 */
//     context->A_lmin, /* minimum eigenvalue */
//     context->A_lmax, /* maximum eigenvalue */
//     context->target_func,  /* values of f(x) for x in [lmin, lmax] */
//     context->mem_inner, /* used memory */
//     context->cheb_coeffs
//   );
//
//   for (MPF_Int i = 0; i < n_batches; ++i)
//   {
//     /* update current rhs and block_size */
//     MPF_Int current_rhs = context->batch*i;
//     MPF_Int current_blk = (1-i/(n_batches-1))*context->batch
//       +(i/(n_batches-1))*(n_max_B-current_rhs);
//
//     /* generate B and initialize X */
//     mp_d_generate_B(context->blk_max_fA, context->colorings_array,
//       current_rhs, context->B.m, current_blk, B->data);
//
//     /* solve AX = B */
//     context->inner_function(context->cheb_M, lmin, lmax,
//       context->cheb_coeffs, A->descr, A->handle, A->m,
//       context->batch, B->data, X->data, context->mem_inner, NULL);
//
//     /* selects specified entries from X */
//     mp_d_blk_xy_select_X_dynamic(context->blk_max_fA,
//       context->colorings_array, X->m, X->data, context->blk_fA, current_rhs,
//       current_blk);
//
//     /* gathers entries of diag_fA */
//     for (MPF_Int j = 0; j < current_blk; ++j)
//     {
//       mp_daxpy(context->B.m, 1.0, &((double*)context->buffer)[context->m*j], 1,
//         &((double*)diag_fA->data)[diag_fA->m*((j+current_rhs)%context->blk_fA)], 1);
//     }
//   }
//
//   /* converts diagonal blocks from symmetric to general format */
//   mp_diag_d_sym2gen(context->m_B, context->blk_fA, context->n_colors,
//     context->blk_max_fA, context->blk_fA, context->diag_fA,
//     context->memory_colorings);
//
//   vslDeleteStream(&stream);
// }
//
// void mpf_batch_z_cheb
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//   MPF_Int n_max_B = context->n_max_B;
//   MPF_Int n_blocks = context->n_blocks;
//   VSLStreamStatePtr stream;
//   vslNewStream(&stream, VSL_BRNG_MCG31, 1);
//   double lmax = 0.0;
//   double lmin = 0.0;
//
//   n_batch = (n_max_B + context->n_batch - 1)/context->batch;
//
//   /* debug information */
//   printf("n_batches: %d\n", (int)n_batches);
//   printf("  n_max_b: %d\n", (int)context->n_max_B);
//   printf("      blk: %d\n", (int)context->blk_solver);
//   printf("      m_B: %d\n", (int)m_B);
//   printf(" bytes_inner: %d\n", (int)context->bytes_inner);
//   printf(" offset: %d\n", context->probing.blocking.offset);
//
//   /* THIS NEEDS RETHINKING                     */
//   /* CASE 1: real spectrum (hermitian)         */
//   /* CASE 2: imaj spectrum (complex symmetric) */
//
//   /* map B and X to pre allocated outer memory */
//   MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//   context->B.data = context->mem_outer;
//   context->X.data = (void *) &((MPF_ComplexDouble *)context->B.data)[context->B.m*context->B.n];
//   context->buffer = context->X;
//
//   for (MPF_Int i = 0; i < n_batches; ++i)
//   {
//     /* update current rhs and block_size */
//     MPF_Int current_rhs = context->batch*i;
//     MPF_Int current_blk = (1-i/(n_batch-1))*context->batch
//       + (i/(n_batches-1))*(n_max_B-current_rhs);
//
//     /* generate B */
//     mp_z_generate_B(context->blk_max_fA, context->colorings_array,
//       current_rhs, context->B.m, current_blk, B);
//
//     /* initialize X */
//     mp_zeros_z_set(context->B.layout, context->X.m, current_blk,
//       context->X, context->X.m);
//
//     /* solve AX = B */
//     context->solver_inner_function(meta, context->A_descr, context->A_handle,
//       context->m_B, context->B, context->X, context->memory_inner, NULL);
//
//     /* gather entries of diag_fA */
//     mp_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//       m_B, X, current_rhs, current_blk);
//     for (j = 0; j < current_blk; ++j)
//     {
//       mp_zaxpy(m_B, &ONE_C, &((MPF_ComplexDouble*)context->buffer)[m_B*j], 1,
//         context->diag_fA, 1);
//     }
//   }
//
//   vslDeleteStream(&stream);
// }
