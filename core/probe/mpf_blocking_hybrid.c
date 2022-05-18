#include "mpf.h"

void mpf_blocking_hybrid
(
  MPF_Probe *probe,
  MPF_Sparse *A
)
{
  printf("blocking hybrid\n");

  MPF_Int *temp_array = (MPF_Int*)probe->buffer;
  MPF_Int *temp_i_array = &temp_array[A->m/probe->stride];

  MPF_Sparse B;
  MPF_Sparse Bc;

  B.descr = A->descr;
  Bc.descr = A->descr;

  mkl_sparse_copy(A->handle, A->descr, &B.handle);

  mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
      &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
  B.nz = A->nz;

  MPF_Int* colorings = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);

  MPF_BucketArray T;

  // encodes merged_nodes
  MPF_BucketArray M[probe->n_levels];
  MPF_Int current_stride = 1;
  for (MPF_Int i = 0; i < probe->n_levels; ++i)
  {
    current_stride = current_stride*probe->stride;
    mpf_bucket_array_init(&M[i], A->m, (A->m+current_stride-1)/current_stride);
    mpf_bucket_array_alloc(&M[i]);
  }

  Bc.m = A->m;
  Bc.n = A->n;
  Bc.nz = A->nz;
  Bc.descr = A->descr;
  Bc.matrix_type = A->matrix_type;
  Bc.data_type = A->data_type;

  mpf_sparse_d_csr_alloc(&Bc);

  mpf_matrix_d_set(MPF_COL_MAJOR, Bc.nz, 1, (double*)Bc.mem.csr.data, Bc.nz, 1.0);

  struct timespec start;
  struct timespec finish;
  probe->runtime_contract = 0.0;

  int status;

  if (probe->stride > 1)
  {
    /* initializes test_array and temp_inverted_array */
    mpf_matrix_i_set(MPF_COL_MAJOR, Bc.m/probe->stride, 1, temp_array,
      Bc.m/probe->stride, 0);
    mpf_matrix_i_set(MPF_COL_MAJOR, Bc.m/probe->stride, 1, temp_i_array,
      Bc.m/probe->stride, -1);

    /* coarsens and expands to approximate S(B^{-1}) */
    for (MPF_Int p = 0; p < probe->n_levels; ++p)
    {
      /* coloring */
      MPF_Int ncolors;
      mpf_color_decoupled(&B, colorings, &ncolors);
      mpf_bucket_array_init(&T, A->m, ncolors);
      mpf_bucket_array_alloc(&T);
      printf("A->m: %d, ncolors: %d, T.n_bins: %d\n", A->m, ncolors, T.n_bins);
      T.n_bins = T.max_n_bins;
      for (MPF_Int i = 0; i < A->m; ++i)
      {
        mpf_bucket_array_insert(&T, colorings[i], i);
      }

      /* contract and expand */
      clock_gettime(CLOCK_MONOTONIC, &start);
      mpf_contract_block_hybrid(probe->stride, temp_array, temp_i_array, &T, &B, &M[p], &Bc);

      status = mkl_sparse_d_create_csr(&Bc.handle, INDEXING, Bc.m,
        Bc.n, Bc.mem.csr.rs, Bc.mem.csr.re, Bc.mem.csr.cols, (double*)Bc.mem.csr.data);
      printf("status: %d\n", status);

      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_contract += mpf_time(start, finish);
//
//      status = mkl_sparse_destroy(B.handle);
//      printf("status (destroy): %d\n", status);
//
//      status = mkl_sparse_copy(Bc.handle, B.descr, &B.handle);
//      printf("status: %d\n", status);
//
//      /* expand */
//      #if MPF_MEASURE
//        clock_gettime(CLOCK_MONOTONIC, &start);
//      #endif
//
//      if (p > 0)
//      {
//        status = mkl_sparse_destroy(probe->P.handle);
//      }
//      ////status = mkl_sparse_set_memory_hint(B_handle, SPARSE_MEMORY_AGGRESSIVE);
//      status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Bc.handle,
//        B.handle, &probe->P.handle);
//      printf("status: %d\n", status);
//
//      status = mkl_sparse_order(probe->P.handle);
//      printf("status: %d\n", status);
//
//      /* color approximate pattern */
//      mpf_sparse_export_csr(&probe->P);
//
//      #if MPF_MEASURE
//        clock_gettime(CLOCK_MONOTONIC, &finish);
//        probe->runtime_expand += mpf_time(start, finish);
//      #endif
//
//      /* export */
//      status = mkl_sparse_destroy(Bc.handle);
//      printf("status: %d\n", status);
//
//      status = mkl_sparse_destroy(B.handle); // testing uncommented
//      printf("status: %d\n", status);
//
//      status = mkl_sparse_copy(probe->P.handle, B.descr, &Bc.handle);
//      printf("status: %d\n", status);
//
//      status = mkl_sparse_copy(probe->P.handle, B.descr, &B.handle);
//      printf("status: %d\n", status);
//
//      /* to get probe->nz_P */
//      status = mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
//        &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
//      status = mkl_sparse_d_export_csr(Bc.handle, &Bc.index, &Bc.m, &Bc.n,
//        &Bc.mem.csr.rs, &Bc.mem.csr.re, &Bc.mem.csr.cols, (double**)&Bc.mem.csr.data);
//
//      //mkl_sparse_destroy(Bc.handle);
//      printf("status: %d\n", status);
      mpf_bucket_array_free(&T);
    }

    probe->runtime_total =
        probe->runtime_contract
      + probe->runtime_expand;
  }
  else if (probe->stride == 1)
  {
    probe->P.handle = NULL; /* initializes probe->P_handle */
    probe->runtime_expand = 0.0;
    mkl_sparse_copy(B.handle, B.descr, &Bc.handle);
    for (MPF_Int p = 0; p < probe->n_levels; ++p)
    {
      /* reset test_array and temp_inverted_array */
      mkl_sparse_destroy(probe->P.handle);
      clock_gettime(CLOCK_MONOTONIC, &start);
      status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Bc.handle,
        B.handle, &probe->P.handle);
      status = mkl_sparse_order(probe->P.handle);
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_expand += mpf_time(start, finish);

      /* copy matrices */
      mkl_sparse_destroy(B.handle);
      status = mkl_sparse_copy(probe->P.handle, B.descr, &B.handle);
      mkl_sparse_destroy(Bc.handle);
      status = mkl_sparse_copy(probe->P.handle, B.descr,
        &Bc.handle);
      /* to get probe->nz_P */
      status = mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
        &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
    }
    probe->runtime_total = probe->runtime_expand;
  }


  /* export sparse pattern */
  if (probe->n_levels == 0)
  {
    status = mkl_sparse_d_export_csr(
      B.handle, &probe->P.index, &probe->P.m, &probe->P.n,
      &probe->P.mem.csr.rs,
      &probe->P.mem.csr.re,
      &probe->P.mem.csr.cols,
      (double**)&probe->P.mem.csr.data);
  }
  else if (probe->stride == 1)
  {
    status = mkl_sparse_d_export_csr(
      probe->P.handle, &probe->P.index, &probe->P.m, &probe->P.n,
      &probe->P.mem.csr.rs,
      &probe->P.mem.csr.re,
      &probe->P.mem.csr.cols,
      (double**)&probe->P.mem.csr.data);
  }
  else
  {
    status = mkl_sparse_d_export_csr(
      probe->P.handle, &probe->P.index, &probe->P.m, &probe->P.n,
      &probe->P.mem.csr.rs,
      &probe->P.mem.csr.re,
      &probe->P.mem.csr.cols,
      (double**)&probe->P.mem.csr.data);
      printf("status: %d\n", status);
  }


  /* counts nonzeros */
  MPF_Int p = 0;
  for (MPF_Int i = 0; i < probe->P.m; ++i)
  {
    p += probe->P.mem.csr.re[i] - probe->P.mem.csr.rs[i];
  }
  probe->P.nz = p;

  /* free memory */
  status = mkl_sparse_destroy(B.handle);
  printf("status: %d\n", status);
  mpf_free(colorings);
}
