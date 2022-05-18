#include "mpf.h"

void mpf_pattern_sample_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels
)
{
  context->probe.max_blk = (MPF_Int)pow((double)stride, (double)n_levels);
  context->probe.n_threads = 1;
  context->probe.type = MPF_PROBE_SAMPLING;
  context->probe.stride = stride;
  context->probe.n_levels = n_levels;
  context->probe.find_pattern_function = &mpf_pattern_sample;
  context->probe.iterations = 1;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.color_function = &mpf_color;
  context->probe.alloc_function = &mpf_probe_alloc;
}

void mpf_pattern_sample
(
  MPF_Probe *probe,
  MPF_Sparse *A
)
{
  MPF_Int p = 0;
  MPF_Int status = 0;
  MPF_Int *temp_array = (MPF_Int*)probe->buffer;
  MPF_Int *temp_inverted_array = &temp_array[A->m/probe->stride];

  MPF_Sparse B;
  B.descr = A->descr;
  MPF_Sparse Ac;
  Ac.descr = A->descr;
  probe->P.descr = A->descr;

  mkl_sparse_copy(A->handle, A->descr, &B.handle);
  mkl_sparse_copy(A->handle, A->descr, &Ac.handle);

  /* Ac metadata */
  Ac.nz = A->nz;
  Ac.m = A->m;
  Ac.n = A->n;

  /* B metadata */
  B.nz = A->nz;
  B.m = A->m;
  B.n = A->n;

  status = mkl_sparse_d_export_csr(Ac.handle, &Ac.index, &Ac.m, &Ac.n,
    &Ac.mem.csr.rs, &Ac.mem.csr.re, &Ac.mem.csr.cols, (double**)&Ac.mem.csr.data);
  mpf_matrix_d_set(MPF_COL_MAJOR, Ac.nz, 1, (double*)Ac.mem.csr.data, Ac.nz, 1.0);
  mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
    &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);
  mpf_matrix_d_set(MPF_COL_MAJOR, Ac.nz, 1, (double*)Ac.mem.csr.data, Ac.nz, 1.0);

  struct timespec start;
  struct timespec finish;

  /* set timers to zero */
  probe->runtime_contract = 0.0;
  probe->runtime_expand = 0.0;

  /* iterates through the number of levels */
  for (p = 0; p < probe->n_levels; ++p)
  {
    /* contract */
    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    mpf_matrix_i_set(MPF_COL_MAJOR, Ac.m, 1, temp_array, Ac.m, 0);
    mpf_matrix_i_set(MPF_COL_MAJOR, Ac.m, 1, temp_inverted_array, Ac.m, -1);
    mpf_contract_sampling(probe, A, &Ac);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_contract += mpf_time(start, finish);
    #endif

    /* expand */
    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    status = mkl_sparse_d_create_csr(&Ac.handle, INDEXING, Ac.m, Ac.n,
      Ac.mem.csr.rs, Ac.mem.csr.re, Ac.mem.csr.cols, (double*)Ac.mem.csr.data);
    status = mkl_sparse_destroy(B.handle);
    status = mkl_sparse_copy(Ac.handle, A->descr, &B.handle);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B.handle, Ac.handle,
      &probe->P.handle);
    status = mkl_sparse_order(probe->P.handle);

    #if MPF_MEASURE
      clock_gettime(CLOCK_MONOTONIC, &finish);
      probe->runtime_expand += mpf_time(start, finish);
    #endif

    status = mkl_sparse_destroy(B.handle);
    status = mkl_sparse_destroy(Ac.handle);

    status = mkl_sparse_copy(probe->P.handle, A->descr, &B.handle);
    mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
      &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols,
      (double**)&B.mem.csr.data);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    #if MPF_PRINTOUT == 1
      printf("p: %d\n", p);
      printf("[%d, %d]\n", B.mem.csr.rs[0], B.mem.csr.re[0]);
    #endif
  }

  status = mkl_sparse_d_export_csr(
    probe->P.handle, &probe->P.index, &probe->P.m, &probe->P.n,
    &probe->P.mem.csr.rs,
    &probe->P.mem.csr.re,
    &probe->P.mem.csr.cols,
    (double**)&probe->P.mem.csr.data);

  /* free Ac */
  //mpf_sparse_csr_free(&Ac); // not required since used destroy

  #if MP_PRINTOUT
    printf("status: %d\n", status);
  #endif

  /* collect total runtime_for probing */
  probe->runtime_total = probe->runtime_contract
    + probe->runtime_expand;
}
