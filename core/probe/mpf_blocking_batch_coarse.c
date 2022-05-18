#include "mpf.h"

void mpf_blocking_batch_coarse_init
(
  MPF_ContextHandle context,
  MPF_Int blk,
  MPF_Int n_levels,
  MPF_Int blk_probing_low_mem,
  MPF_Int expansion_degree
)
{
  context->probe.type = MPF_PROBE_BATCH_COMPACT_BLOCKING;
  context->probe.stride = blk;
  context->probe.max_blk = (MPF_Int)pow((double)blk, (double)n_levels);
  context->probe.n_levels = n_levels;
  context->probe.find_pattern_function = mpf_batch_compact_blocking;
  context->probe.iterations = 1;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.color_function = &mpf_color;
  context->probe.alloc_function = &mpf_probe_alloc;
  context->probe.expansion_degree = expansion_degree;
  context->probe.batch = blk_probing_low_mem;
}

/* mpf_blocking_mkl_fA_low_mem                                                 */
/* probing implementation that uses spmm function for the expansion operation */
void mpf_batch_compact_blocking
(
  MPF_Probe *probe,
  MPF_Sparse *A
)
{
  /* indices */
  MPF_Int p = 0;

  /* unpack buffer memory */
  MPF_Int *temp_array = (MPF_Int*)probe->buffer;
  MPF_Int *temp_i_array = &(temp_array)[probe->n_nodes]; /* temp inverse table */

  /* used for timing */
  struct timespec start;
  struct timespec finish;

  #if MPF_MEASURE
    probe->runtime_total = 0.0;
    probe->runtime_contract = 0.0;
    probe->runtime_expand = 0.0;
    probe->runtime_other = 0.0;
    probe->runtime_color = 0.0;
  #endif

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  /* array of coarsened versions of A */
  MPF_Sparse *Ac = (MPF_Sparse*)mpf_malloc(sizeof(MPF_Sparse)*probe->n_levels);
  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    Ac[i].m = A->m/((i+1)*2);
    Ac[i].n = A->n/((i+1)*2);
    Ac[i].nz = A->nz/((i+1)*2);
    Ac[i].descr = A->descr;
    mpf_sparse_d_csr_alloc(&Ac[i]);
  }
  probe->P.descr = A->descr;

  /* Pc (compact sparsity pattern) */
  MPF_Sparse Pc;
  Pc.m = A->m;
  Pc.n = A->n;
  Pc.nz = A->nz;
  Pc.descr = A->descr;
  Pc.matrix_type = A->matrix_type;
  Pc.data_type = A->data_type;
  printf("before\n");
  mpf_sparse_d_csr_alloc(&Pc);
  printf("after\n");

  /* initializes Pc */

  mpf_zeros_i_set(MPF_COL_MAJOR, probe->stride, 1, Pc.mem.csr.rs, probe->stride);
  mpf_zeros_i_set(MPF_COL_MAJOR, probe->stride, 1, Pc.mem.csr.re, probe->stride);
  mpf_zeros_i_set(MPF_COL_MAJOR, probe->m*probe->stride, 1, Pc.mem.csr.cols, Pc.m*probe->stride);
  mpf_zeros_d_set(MPF_COL_MAJOR, probe->m*probe->stride, 1, (double*)Pc.mem.csr.data, A->m*probe->stride);

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    probe->runtime_other += mpf_time(start, finish);
  #endif


  /* compute m_coarse (number of compact nodes) */
  MPF_Int m_coarse = A->m;
  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    m_coarse = (m_coarse+probe->stride-1)/probe->stride;
  }

  /* initializes list */
  MPF_LinkedList *list = mpf_linked_list_create(m_coarse);

  /* initializes colorings to [-1...-1] array */
  mpf_matrix_i_set(MPF_COL_MAJOR, m_coarse, 1, probe->colorings_array, m_coarse, -1);

  MPF_Int status = 0; /* used for debuging MKL kernels */
  if (probe->stride > 1)
  {
    /* initializes batches and number of batches */
    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    probe->batch = pow((double)probe->stride, (double)probe->n_levels-1);
    MPF_Int n_batches = (A->m/probe->stride+probe->batch-1) /probe->batch;

    /* constructs compact patterns of A stored in Ac */
    mpf_compact_hierarchy(probe, A, Ac);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_other += mpf_time(start, finish);
    #endif

    /* applies updates for all blocks */
    for (MPF_Int i = 0; i < n_batches; ++i)
    {
      /*----------------*/
      /* initializes Pc */
      /*----------------*/
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif

      /* set number of rows/cols and nonzeros of sparse matrix P */
      probe->P.nz = 0;
      probe->P.m = probe->batch;
      probe->P.n = Ac[0].n;

      /* sets current row and block that the batch probing method is working on
         at the moment  */
      MPF_Int current_row = probe->batch*i;
      MPF_Int current_blk = ((1-(i+1)/n_batches))*probe->batch
        + ((i+1)/n_batches)*(probe->P.n-current_row);

      /* creates a sparse pattern for the compressed batch of P,
         [current_row, current_row+current_batch] stored in Pc, to be used for
         the expansion step. */
      mpf_sparse_d_copy(current_row, current_row+current_blk, &Ac[0],
        0, current_blk, &Pc);

      status = mkl_sparse_d_create_csr
      (
        &Pc.handle,
        INDEXING,
        Pc.m,
        Pc.n,
        Pc.mem.csr.rs,
        Pc.mem.csr.re,
        Pc.mem.csr.cols,
        (double*)Pc.mem.csr.data
      );

      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        probe->runtime_other += mpf_time(start, finish);
      #endif

      /*-----------------------------------*/
      /* iterations 0 to probe->n_levels-2 */
      /*-----------------------------------*/
      for (p = 0; p < probe->n_levels-1; ++p)
      {

        /* applies expansion */
        MPF_Int niters = probe->expansion_degree;
        for (MPF_Int z = 0; z < niters-1; ++z)
        {
          #if MPF_MEASURE
            clock_gettime(CLOCK_MONOTONIC, &start);
          #endif

          status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
            Pc.handle, Ac[p].handle, &probe->P.handle);
          mkl_sparse_order(probe->P.handle);

          #if MPF_MEASURE
            clock_gettime(CLOCK_MONOTONIC, &finish);
            probe->runtime_expand += mpf_time(start, finish);
          #endif

          /* copies result to Pc.handle */
          mkl_sparse_destroy(Pc.handle);
          mkl_sparse_copy(probe->P.handle, A->descr, &Pc.handle);
          mkl_sparse_destroy(probe->P.handle);
        }

        /* last expansion */
        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
          Pc.handle, Ac[p].handle, &probe->P.handle);
        status = mkl_sparse_order(probe->P.handle);

        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &finish);
          probe->runtime_expand += mpf_time(start, finish);
        #endif

        /* contract A and block of rows */
        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        status = mkl_sparse_d_export_csr
        (
          probe->P.handle,
          &probe->P.index,
          &probe->P.m,
          &probe->P.n,
          &probe->P.mem.csr.rs,
          &probe->P.mem.csr.re,
          &probe->P.mem.csr.cols,
          (double**)&probe->P.mem.csr.data
        );
        mpf_block_row_contract(probe->stride, &probe->P, current_row, temp_array,
          temp_i_array, &Pc);

        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &finish);
          probe->runtime_contract += mpf_time(start, finish);
        #endif

        /* copies compact representation to probe->P.handle  */
        status = mkl_sparse_destroy(Pc.handle);
        status = mkl_sparse_d_create_csr
        (
          &Pc.handle,
          INDEXING,
          current_blk,
          Pc.m,
          Pc.mem.csr.rs,
          Pc.mem.csr.re,
          Pc.mem.csr.cols,
          (double*)Pc.mem.csr.data
        );
      }

      /*-----------------------------*/
      /* iteration probe->n_levels-1 */
      /*-----------------------------*/
      if (p == probe->n_levels-1)
      {
        /* applies expansion */
        status = mkl_sparse_destroy(probe->P.handle);
        MPF_Int niters = probe->expansion_degree;
        for (MPF_Int z = 0; z < niters-1; ++z)
        {
          #if MPF_MEASURE
            clock_gettime(CLOCK_MONOTONIC, &start);
          #endif

          status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
            Pc.handle, Ac[p].handle, &probe->P.handle);
          status = mkl_sparse_order(probe->P.handle);

          #if MPF_MEASURE
            clock_gettime(CLOCK_MONOTONIC, &finish);
            probe->runtime_expand += mpf_time(start, finish);
          #endif

          /* copies result to P.handle */
          status = mkl_sparse_destroy(Pc.handle);
          status = mkl_sparse_copy(probe->P.handle, probe->P.descr, &Pc.handle);
          status = mkl_sparse_destroy(probe->P.handle);
        }

        /* last expansion of the last iteration */
        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
          Pc.handle, Ac[p].handle, &probe->P.handle);
        status = mkl_sparse_order(probe->P.handle);

        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &finish);
          probe->runtime_expand += mpf_time(start, finish);
        #endif
      }

      /*--------------------------------------------------*/
      /* constructs compact pattern row (P) to be colored */
      /*--------------------------------------------------*/
      status = mkl_sparse_d_export_csr
      (
        probe->P.handle,
        &probe->P.index,
        &probe->P.m,
        &probe->P.n,
        &probe->P.mem.csr.rs,
        &probe->P.mem.csr.re,
        &probe->P.mem.csr.cols,
        (double**)&probe->P.mem.csr.data
      );

      /* color output */
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif

      mpf_color_partial(probe, i, 1, list);

      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        probe->runtime_color += mpf_time(start, finish);
      #endif

      /* free memory */
      mkl_sparse_destroy(Pc.handle);
      mkl_sparse_destroy(probe->P.handle);
    }

    #if MPF_MEASURE
      probe->runtime_total
        = probe->runtime_contract
        + probe->runtime_expand
        + probe->runtime_color
        + probe->runtime_other;
    #endif
  }

  mpf_sparse_csr_free(&Pc);

  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    mpf_sparse_csr_free(&Ac[i]);
  }

  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    mkl_sparse_destroy(Ac[i].handle);
    //mpf_sparse_csr_free(&Ac.mem.csr);
  }

  mpf_free(Ac);
  mpf_linked_list_destroy(list);

  probe->P.m = m_coarse;  /* required by color to node map function */
}
