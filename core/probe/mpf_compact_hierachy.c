#include "mpf.h"

/* computes hierarchy of compact matrices (Ac) */
void mpf_compact_hierarchy
(
  MPF_Probe *probe,
  MPF_Sparse *A,
  MPF_Sparse *Ac_array
)
{
  MPF_Int *temp_array = (MPF_Int*)probe->buffer;
  MPF_Int *temp_i_array = &(temp_array)[probe->n_nodes]; /* temp inverse table */

  /* apply block coarsening */
  mpf_block_contract(probe->stride, temp_array, temp_i_array, A, &Ac_array[0]);

  /* creates  Ac_array_handle[0] for first compact pattern */
  MPF_Int status = mkl_sparse_d_create_csr(
    &Ac_array[0].handle, INDEXING, Ac_array[0].m, Ac_array[0].n,
    Ac_array[0].mem.csr.rs, Ac_array[0].mem.csr.re,
    Ac_array[0].mem.csr.cols, (double*)Ac_array[0].mem.csr.data);
  printf("status: %d\n", status);

  /* initializes Ac_array and Ac_array_hanlde objects */
  for (MPF_Int i = 1; i < probe->n_levels; ++i)
  {
    mpf_block_contract(probe->stride, temp_array, temp_i_array, &Ac_array[i-1],
      &Ac_array[i]);

    status = mkl_sparse_d_create_csr(
      &Ac_array[i].handle, INDEXING, Ac_array[i].m, Ac_array[i].m,
      Ac_array[i].mem.csr.rs, Ac_array[i].mem.csr.re,
      Ac_array[i].mem.csr.cols, (double*)Ac_array[i].mem.csr.data);
    printf("status (csr_create): %d\n", status);
  }
}
