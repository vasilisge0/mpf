#include "mpf.h"

void mpf_blocking_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
)
{
  context->probe.max_blk = (MPF_Int)pow((double)stride, (double)n_levels);
  context->probe.type = MPF_PROBE_BLOCKING;
  context->probe.stride = stride;
  context->probe.n_levels = n_levels;
  context->probe.iterations = 1;
  context->probe.find_pattern_function = &mpf_blocking;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.color_function = &mpf_color;
  context->probe.alloc_function = &mpf_probe_alloc;
}

/* probing implementation that uses spmm function for the expansion operation */
void mpf_blocking
(
  MPF_Probe *probe,
  MPF_Sparse *A
)
{
  MPF_Int p = 0;

  /* arrays used for coarsening */
  MPF_Int *temp_array = (MPF_Int*)probe->buffer;
  MPF_Int *temp_i_array = &temp_array[A->m/probe->stride];

  /* patterns */
  MPF_Sparse B;
  MPF_Sparse Bc;

  B.descr = A->descr;
  Bc.descr = A->descr;

  #if DEBUG
    printf("A->m: %d, A->n: %d, A->nz: %d, probe->stride: %d\n", A->m, A->n, A->nz, probe->stride);
  #endif

  MPF_Int status = mkl_sparse_copy(A->handle, A->descr, &B.handle);
  printf("status (copy): %d\n", status);

  status = mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
    &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
  printf("status: %d\n", status);

  printf("A->nz: %d\n", A->nz);
  B.nz = A->nz;
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
      /* contract and expand */
      clock_gettime(CLOCK_MONOTONIC, &start);
      mpf_block_contract(probe->stride, temp_array, temp_i_array, &B, &Bc);

      status = mkl_sparse_d_create_csr(&Bc.handle, INDEXING, Bc.m,
        Bc.n, Bc.mem.csr.rs, Bc.mem.csr.re, Bc.mem.csr.cols, (double*)Bc.mem.csr.data);
      #if DEBUG
        printf("status: %d\n", status);
      #endif

      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_contract += mpf_time(start, finish);

      status = mkl_sparse_destroy(B.handle);
      printf("status (destroy): %d\n", status);

      status = mkl_sparse_copy(Bc.handle, B.descr, &B.handle);
      printf("status: %d\n", status);

      /* expand */
      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &start);
      #endif

      if (p > 0)
      {
        status = mkl_sparse_destroy(probe->P.handle);
      }
      ////status = mkl_sparse_set_memory_hint(B_handle, SPARSE_MEMORY_AGGRESSIVE);
      status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Bc.handle,
        B.handle, &probe->P.handle);
      printf("status: %d\n", status);

      status = mkl_sparse_order(probe->P.handle);
      printf("status: %d\n", status);

      #if MPF_MEASURE
        clock_gettime(CLOCK_MONOTONIC, &finish);
        probe->runtime_expand += mpf_time(start, finish);
      #endif

      /* export */
      status = mkl_sparse_destroy(Bc.handle);
      printf("status: %d\n", status);

      status = mkl_sparse_destroy(B.handle); // testing uncommented
      printf("status: %d\n", status);

      status = mkl_sparse_copy(probe->P.handle, B.descr, &Bc.handle);
      printf("status: %d\n", status);

      status = mkl_sparse_copy(probe->P.handle, B.descr, &B.handle);
      printf("status: %d\n", status);

      /* to get probe->nz_P */
      status = mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
        &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
      status = mkl_sparse_d_export_csr(Bc.handle, &Bc.index, &Bc.m, &Bc.n,
        &Bc.mem.csr.rs, &Bc.mem.csr.re, &Bc.mem.csr.cols, (double**)&Bc.mem.csr.data);

      //mkl_sparse_destroy(Bc.handle);
      printf("status: %d\n", status);
    }

    probe->runtime_total =
        probe->runtime_contract
      + probe->runtime_expand;
  }
  else if (probe->stride == 1)
  {
    MPF_Int status = 0;
    probe->P.handle = NULL; /* initializes probe->P_handle */
    probe->runtime_expand = 0.0;
    status = mkl_sparse_copy(B.handle, B.descr, &Bc.handle);
    printf("status (init): %d\n", status);
    for (MPF_Int p = 0; p < probe->n_levels; ++p)
    {
      /* reset test_array and temp_inverted_array */
      if (p > 0)
      {
        status = mkl_sparse_destroy(probe->P.handle);
      }
      printf("status (destroy): %d\n", status);
      clock_gettime(CLOCK_MONOTONIC, &start);
      status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Bc.handle,
        B.handle, &probe->P.handle);
      printf("status: %d\n", status);
      status = mkl_sparse_order(probe->P.handle);
      printf("status (order): %d\n", status);
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_expand += mpf_time(start, finish);

      /* copy matrices */
      mkl_sparse_destroy(B.handle);
      status = mkl_sparse_copy(probe->P.handle, B.descr, &B.handle);
      printf("status: %d\n", status);
      mkl_sparse_destroy(Bc.handle);
      status = mkl_sparse_copy(probe->P.handle, B.descr,
        &Bc.handle);
      printf("status: %d\n", status);
      /* to get probe->nz_P */
      status = mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
        &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
      printf("status: %d\n", status);
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
  status = mkl_sparse_d_export_csr(probe->P.handle, &probe->P.index, &probe->P.m,
    &probe->P.n, &probe->P.mem.csr.rs, &probe->P.mem.csr.re, &probe->P.mem.csr.cols,
    (double**)&probe->P.mem.csr.data);
  printf("status: %d\n", status);
  p = 0;
  for (MPF_Int i = 0; i < probe->P.m; ++i)
  {
    p += probe->P.mem.csr.re[i] - probe->P.mem.csr.rs[i];
  }
  probe->P.nz = p;

  /* free memory */
  status = mkl_sparse_destroy(B.handle);
  printf("status: %d\n", status);
}
