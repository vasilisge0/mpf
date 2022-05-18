#include "mpf.h"

void mpf_blocking_batch_init
(
  MPF_ContextHandle context,
  MPF_Int blk,                  /* used for probing */
  MPF_Int n_levels,
  MPF_Int batch,
  MPF_Int expansion_degree
)
{
  context->probe.type = MPF_PROBE_BATCH_BLOCKING;
  context->probe.stride = blk;
  context->probe.n_levels = n_levels;
  context->probe.P.m = 0; // changed (probe->P.m)
  context->probe.iterations = 1;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.find_pattern_function = mpf_batch_blocking;
  context->probe.color_function = &mpf_color;
  context->probe.alloc_function = &mpf_probe_alloc;
  context->probe.expansion_degree = expansion_degree;
}

/* mpf_blocking_mkl_fA_low_mem                                                */
/* probing implementation that uses spmm function for the expansion operation */
/* only for 1 level */
void mpf_batch_blocking
(
  MPF_Probe *probe,  // main probe
  MPF_Sparse *A
)
{
  /* not used thus far */
  MPF_Int *temp_array = NULL;
  MPF_Int *temp_i_array = NULL;

  //mpf_probe_unpack_sort_mem(probe, temp_array, temp_i_array);
  temp_array = (MPF_Int*)probe->buffer;
  temp_i_array = &(temp_array)[probe->n_nodes]; /* temp inverse table */

  /* initialization */
  mpf_matrix_i_set(MPF_COL_MAJOR, probe->n_nodes, 1, temp_array,
    probe->n_nodes, 0);
  mpf_matrix_i_set(MPF_COL_MAJOR, probe->n_nodes, 1, temp_i_array,
     probe->n_nodes, -1);

  /* sparse handles */
  MPF_Sparse Ac;/* array of handles */
  MPF_Sparse Pc;/* 4 */

  Ac.descr = A->descr;
  Pc.descr = A->descr;

  /* data */
  //MPF_Sparse P;

  /* allocates memory for all sparse matrices in array Ac */
  MPF_Int status = mkl_sparse_copy(A->handle, A->descr, &Ac.handle); 
  printf("status: %d\n", status);
  //mpf_sparse_export_csr(&Ac);
  mpf_sparse_d_export_csr(Ac.handle, &Ac.index,
    &Ac.m, &Ac.n, &Ac.mem.csr.rs, &Ac.mem.csr.re, &Ac.mem.csr.cols,
    (double**)&Ac.mem.csr.data);

  /* alloates memory for Pc */
  mkl_sparse_copy(A->handle, A->descr, &Pc.handle);
  //mpf_sparse_export_csr(&Pc);
  mpf_sparse_d_export_csr(Pc.handle, &Pc.index,
    &Pc.m, &Pc.n, &Pc.mem.csr.rs, &Pc.mem.csr.re, &Pc.mem.csr.cols,
    (double**)&Pc.mem.csr.data);
  printf("A->m: %d, A->n: %d\n", A->m, A->n);
  printf("Pc.m: %d, Pc.n: %d\n", Pc.m, Pc.n);
  Pc.mem.csr.rs[0] = 0;

  /* initializes */
  mpf_zeros_i_set(MPF_COL_MAJOR, probe->batch, 1, Pc.mem.csr.rs,
    probe->batch);
  mpf_zeros_i_set(MPF_COL_MAJOR, probe->batch, 1, Pc.mem.csr.re,
    probe->batch);
  mpf_zeros_i_set(MPF_COL_MAJOR, A->m*probe->batch, 1, Pc.mem.csr.cols,
    A->m*probe->batch);
  mpf_zeros_d_set(MPF_COL_MAJOR, A->m*probe->batch, 1, (double*)Pc.mem.csr.data,
    A->m*probe->batch);

  /* initializes handles*/
  //mkl_sparse_copy(A->handle, A->descr, &A->handle);

  /* for timing */
  struct timespec start;
  struct timespec finish;
  probe->runtime_contract = 0.0;

  /* initialize coloring list */
  MPF_Int m = A->m;
  m = (m+probe->stride-1)/probe->stride;

  /* initializes list */
  MPF_LinkedList *list = mpf_linked_list_create(m);

  /* initializes colorings to [-1...-1] array */
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, probe->colorings_array, m, -1);

  printf("Ac.mem.csr.rs[i]: %d", Ac.mem.csr.rs[0]);
  printf(" -> %d\n", Ac.mem.csr.cols[Ac.mem.csr.rs[0]]);

  if (probe->stride > 1)
  {
    probe->batch = 1;
    MPF_Int n_blocks = (A->m/probe->stride+probe->batch-1)/probe->batch;

    /*----------------------------------------*/
    /* computes hierarchy of Ac matrices */
    /*----------------------------------------*/

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    mpf_block_contract(probe->stride, temp_array, temp_i_array, A, &Ac);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_contract += mpf_time(start, finish);
    #endif

    MPF_Int status = mkl_sparse_d_create_csr(&Ac.handle, INDEXING,
      Ac.m, Ac.n, Ac.mem.csr.rs, Ac.mem.csr.re,
      Ac.mem.csr.cols, (double*)Ac.mem.csr.data);
    //mkl_sparse_destroy(A->handle);

    /* applies updates for all blocks */
    for (MPF_Int i = 0; i < n_blocks; ++i)
    {
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif
      probe->P.nz = 0;
      probe->P.m = (A->m+probe->stride-1)/probe->stride;

      MPF_Int nz_Ac = Ac.mem.csr.re[i]-Ac.mem.csr.rs[i];
      Pc.mem.csr.rs[0] = 0;
      Pc.mem.csr.re[0] = Ac.mem.csr.re[i]-Ac.mem.csr.rs[i];
      memcpy(Pc.mem.csr.cols, &Ac.mem.csr.cols[Ac.mem.csr.rs[i]],
        sizeof(MPF_Int)*nz_Ac);
      memcpy(Pc.mem.csr.data, &((double*)Ac.mem.csr.data)[Ac.mem.csr.rs[i]],
        sizeof(double)*nz_Ac);

      MPF_Int status = mkl_sparse_d_create_csr
      (
        &Pc.handle,
        INDEXING,
        1,
        (A->m+probe->stride-1)/probe->stride,
        Pc.mem.csr.rs,
        Pc.mem.csr.re,
        Pc.mem.csr.cols,
        (double*)Pc.mem.csr.data
      );
      printf("status: %d\n", status);
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        probe->runtime_contract += mpf_time(start, finish);
      #endif

      /* last iteration */
      MPF_Int niters = pow((double)2.0, (double)probe->n_levels)-1.0;
      if (niters >= probe->expansion_degree)
      {
        niters = probe->expansion_degree;
      }
      else
      {
        printf("In mpf_partial_color exit\n");
        return;
      }

      //probe->expansion_degree = niters;
      printf("expansion_degree: %d\n", probe->expansion_degree);
      for (MPF_Int z = 0; z < probe->expansion_degree; ++z)
      {
        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &start);
        #endif
        status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
          Pc.handle, Ac.handle, &probe->P.handle);
        printf("status (spmm): %d\n", status);
        mkl_sparse_order(probe->P.handle);
        printf("status (order): %d\n", status);
        #if MPF_MEASURE
          clock_gettime(CLOCK_MONOTONIC, &finish);
          probe->runtime_expand += mpf_time(start, finish);
        #endif

        if (z < niters-1)
        {
          mkl_sparse_destroy(Pc.handle);
          mkl_sparse_copy(probe->P.handle, A->descr, &Pc.handle);
          mkl_sparse_destroy(probe->P.handle);
          printf("destroy\n");
          printf("z: %d, niters-1: %d\n", z, niters-1);
          printf("niters: %d\n", niters);
        }
      }

      /* constructs P object */
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
      printf("status: %d\n", status);

      /* color output */
      printf("probe->P.mem.csr.rs[0]: %d\n", probe->P.mem.csr.rs[0]);
      mpf_color_partial(probe, i, 1, list);

      mkl_sparse_destroy(Pc.handle);
      mkl_sparse_destroy(probe->P.handle);
    }
    probe->runtime_total = probe->runtime_contract
      + probe->runtime_expand;
  }
////  else if (blk == 1)
////  {
////    probe->P_handle = NULL; /* initializes probe->P_handle */
////    probe->runtime_expand = 0.0;
////    for (p = 0; p < probe->n_levels; ++p)
////    {
////      /* reset test_array and temp_inverted_array */
////      mkl_sparse_destroy(probe->P_handle);
////      clock_gettime(CLOCK_MONOTONIC, &start);
////      status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Ac_handle,
////        A_handle, &probe->P_handle);
////      status = mkl_sparse_order(probe->P_handle);
////      clock_gettime(CLOCK_MONOTONIC, &finish);
////      probe->runtime_expand += mpf_time(start, finish);
////      printf("mpf_time: %1.4e\n", mpf_time(start, finish));
////      /* copy matrices */
////      mkl_sparse_destroy(A_handle);
////      status = mkl_sparse_copy(probe->P_handle, probe->A_descr, &A_handle);
////      mkl_sparse_destroy(Ac_handle);
////      status = mkl_sparse_copy(probe->P_handle, probe->A_descr,
////        &Ac_handle);
////
////      /* to get probe->nz_P */
////      status = mkl_sparse_d_export_csr(A_handle, &index, &nr, &nc,
////        &A_csr->rs, &A_csr->re, &A_csr->cols, (double**)&d);
////    }
////    probe->runtime_probing = probe->runtime_expand;
////  }
//
  //mpf_free(Pc.mem.csr.rs);
  //mpf_free(Pc.mem.csr.re);
  //mpf_free(Pc.mem.csr.cols);
  //mpf_free(Pc.mem.csr.data);

  //mpf_free(Ac.mem.csr.rs);
  //mpf_free(Ac.mem.csr.re);
  //mpf_free(Ac.mem.csr.cols);
  //mpf_free(Ac.mem.csr.data);

  mkl_sparse_destroy(Ac.handle);

  mpf_linked_list_destroy(list); // !perhaps initialized outsize
}
