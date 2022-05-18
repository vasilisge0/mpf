#include "mpf.h"

void mpf_pattern_multisample_init
(
  MPF_ContextHandle context,
  MPF_Int stride,
  MPF_Int n_levels,
  MPF_Int n_endpoints
)
{
  srand(time(0));
  context->probe.n_threads = 1;
  context->probe.type = MPF_PROBE_AVG_PATH_SAMPLING;
  context->probe.stride = stride;
  context->probe.n_levels = n_levels;
  context->probe.n_endpoints = n_endpoints;
  context->probe.find_pattern_function = mpf_pattern_multisample;
  context->probe.P.m = 0;
  context->probe.P.nz = 0;
  context->probe.iterations = n_endpoints;
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
  context->probe.color_function = &mpf_color;
  context->probe.alloc_function = &mpf_avg_probe_alloc;
}

/* optimzed */
void mpf_pattern_multisample
(
  MPF_Probe *probe,
  MPF_Sparse *A
)
{
  MPF_Sparse B;
  MPF_Sparse C;
  MPF_Sparse D;

  MPF_Int p = 0;

  MPF_Int *operator_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*probe->n_levels);

  probe->runtime_contract = 0.0;
  probe->runtime_expand = 0.0;

  struct timespec start;
  struct timespec finish;

  B.m = A->m;
  B.n = A->n;
  B.nz = A->nz;

  C.m = A->m;
  C.n = A->n;
  C.nz = A->nz;

  D.m = A->m;
  D.n = A->n;
  D.nz = A->nz;

  B.descr = A->descr;
  C.descr = A->descr;
  D.descr = A->descr;

  mkl_sparse_copy(A->handle, A->descr, &B.handle);
  mkl_sparse_d_export_csr(B.handle, &B.index, &B.m, &B.n,
    &B.mem.csr.rs, &B.mem.csr.re, &B.mem.csr.cols, (double**)&B.mem.csr.data);

  mpf_sparse_d_csr_alloc(&C);
  mpf_sparse_d_csr_alloc(&D);

  /* iterates through the number of levels */
  printf("probe->current_iteration: %d\n", probe->current_iteration);
  MPF_Int i = probe->current_iteration;
  printf("test\n");
  printf("probe->endpoints_array[i]: %d\n", probe->endpoints_array[i]);

  /* generates list of sampling operators that is used */
  mpf_multipath_path_unpack(probe->endpoints_array[i], probe->stride,
    operator_array, probe->n_levels);

  #if MPF_PRINTOUT
    printf("i: %d\n", i);
    printf("endpoint[%d]: %d -> %d, depth: %d\n", i,
      probe->endpoints_array[i], operator_array[0], probe->n_levels);
  #endif

  /* contract */
  clock_gettime(CLOCK_MONOTONIC, &start);
  printf("operator_array[0]: %d\n", operator_array[0]);
  switch(operator_array[0])
  {
    case 0:
      mpf_contract_dynamic_sample(probe, A, &B, 0);
      mpf_contract_dynamic_sample(probe, A, &C, 0);
    case 1:
      mpf_contract_dynamic_sample(probe, A, &B, 0);
      mpf_contract_dynamic_sample(probe, A, &C, 1);
    case 2:
      mpf_contract_dynamic_sample(probe, A, &B, 1);
      mpf_contract_dynamic_sample(probe, A, &C, 1);
    default:
      break;
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
  probe->runtime_contract += mpf_time(start, finish);
  #if MPF_PRINTOUT
    printf("operator_array[%d]: %d\n", i, operator_array[i]);
  #endif

  // is it required(?)
  //probe->P.m = (probe->P.m+probe->stride-1)/probe->stride;

  /* === expand === */

  clock_gettime(CLOCK_MONOTONIC, &start);
  mkl_sparse_d_create_csr(&B.handle, INDEXING, probe->P.m,
    probe->P.m, B.mem.csr.rs, B.mem.csr.re,
    B.mem.csr.cols, (double*)B.mem.csr.data);
  mkl_sparse_d_create_csr(&C.handle, INDEXING, C.m,
    C.n, C.mem.csr.rs, C.mem.csr.re, C.mem.csr.cols, (double*)C.mem.csr.data);
  mkl_sparse_d_create_csr(&D.handle, INDEXING, D.m,
    D.n, D.mem.csr.rs, D.mem.csr.re, D.mem.csr.cols, (double*)D.mem.csr.data);

  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B.handle, C.handle,
    &probe->P.handle);

  mkl_sparse_order(probe->P.handle);
  mkl_sparse_destroy(B.handle);
  mkl_sparse_destroy(C.handle);
  if (operator_array[0] == 1)
  {
    /* add the transpose */
    mkl_sparse_copy(probe->P.handle, probe->P.descr, &C.handle);
    mkl_sparse_copy(probe->P.handle, probe->P.descr, &B.handle);
    mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, B.handle, 1.0, C.handle,
      (sparse_matrix_t*)&probe->P.handle);
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
  probe->runtime_expand += mpf_time(start, finish);

  for (p = 1; p < probe->n_levels; ++p)
  {
    #if MPF_PRINTOUT
      printf("\n\n >> p: %d\n\n", p);
    #endif

    MPF_Int status = mkl_sparse_copy(probe->P.handle, probe->P.descr, &B.handle); // this shoudl be commented change
    mkl_sparse_d_export_csr(
      B.handle, &B.index, &B.m, &B.n,
      &B.mem.csr.rs,
      &B.mem.csr.re,
      &B.mem.csr.cols,
      (double**)&B.mem.csr.data);

    /* contract */
    clock_gettime(CLOCK_MONOTONIC, &start);
    switch(operator_array[0])
    {
      case 0:
        mpf_contract_dynamic_sample(probe, A, &B, 0);
        mpf_contract_dynamic_sample(probe, A, &C, 0);
      case 1:
        mpf_contract_dynamic_sample(probe, A, &B, 0);
        mpf_contract_dynamic_sample(probe, A, &C, 1);
      case 2:
        mpf_contract_dynamic_sample(probe, A, &B, 1);
        mpf_contract_dynamic_sample(probe, A, &C, 1);
      default:
        break;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    probe->runtime_contract += mpf_time(start, finish);

    /* epxansion */
    mkl_sparse_d_create_csr(&B.handle, INDEXING, B.m,
      B.n, B.mem.csr.rs, B.mem.csr.re, B.mem.csr.cols, (double*)B.mem.csr.data);
    mkl_sparse_d_create_csr(&C.handle, INDEXING, C.m,
      C.n, C.mem.csr.rs, C.mem.csr.re, C.mem.csr.cols, (double*)C.mem.csr.data);

    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, B.handle, C.handle,
      &probe->P.handle);

    mkl_sparse_order(probe->P.handle);
    mkl_sparse_destroy(B.handle);
    mkl_sparse_destroy(C.handle);

    if (operator_array[0] == 1)
    {
      /* add the transpose */
      mkl_sparse_copy(probe->P.handle, probe->P.descr, &C.handle);
      mkl_sparse_copy(probe->P.handle, probe->P.descr, &B.handle);
      mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, B.handle, 1.0,
        C.handle, &probe->P.handle);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    probe->runtime_expand += mpf_time(start, finish);
  }

  mkl_sparse_d_export_csr(
    probe->P.handle, &probe->P.index, &probe->P.m, &probe->P.n,
    &probe->P.mem.csr.rs,
    &probe->P.mem.csr.re,
    &probe->P.mem.csr.cols,
    (double**)&probe->P.mem.csr.data);

  probe->runtime_total = probe->runtime_contract
    + probe->runtime_expand;

  #if MPF_PRINTOUT
    printf("nr (P): %d, nc (P): %d\n", probe->P.m, probe->P.n);
    printf("(contract, expand, total) -> (%1.1E, %1.1E, %1.1E)",
      probe->runtime_contract,
      probe->runtime_expand,
      probe->runtime_total);
  #endif

  mpf_free(operator_array);
}
