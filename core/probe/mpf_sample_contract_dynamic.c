#include "mpf.h"

void mpf_contract_dynamic_sample
(
  MPF_Probe *context,
  MPF_Sparse *A,
  MPF_Sparse *B,
  MPF_Int coarse_op
)
{
  #if MPF_PRINTOUT
    printf("coarse_op: %d\n", coarse_op);
  #endif
  MPF_Int n_edges_new = 0;
  MPF_Int z = 0;
  MPF_Int c = 0;  /* used for columns */
  MPF_Int swap = 0;

  B->mem.csr.rs[0] = 0;
  B->mem.csr.re[0] = 0;
  B->descr.mode = A->descr.mode;

  if (coarse_op == 0)
  {
    for (MPF_Int i = 0; i < A->m; i += context->stride) /* parse each row */
    {
      if (i > 0)
      {
        B->mem.csr.rs[i/context->stride] = B->mem.csr.re[i/context->stride-1];
        B->mem.csr.re[i/context->stride] = B->mem.csr.rs[i/context->stride];
      }

      for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
      {
        c = A->mem.csr.cols[j];
        if (c % context->stride == 0)
        {
          B->mem.csr.cols[n_edges_new] = c/context->stride;
          ++n_edges_new;
          B->mem.csr.re[i/context->stride] = n_edges_new;
        }
      }
    }
  }
  else if (coarse_op == 1)
  {
    for (MPF_Int i = 0; i < A->m; i += 1) /* parse each row */
    {
      if (i > 0)
      {
        B->mem.csr.rs[i/context->stride] = B->mem.csr.re[i/context->stride-1];
        B->mem.csr.re[i/context->stride] = B->mem.csr.rs[i/context->stride];
      }

      for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
      {
        c = A->mem.csr.cols[j];
        if (c % context->stride == 0)
        {
          B->mem.csr.cols[n_edges_new] = c/context->stride;
          ++n_edges_new;
          B->mem.csr.re[i/context->stride] = n_edges_new;
        }
        else if (i % context->stride > 0)
        {
          z = B->mem.csr.rs[i/context->stride];
          while ((z > B->mem.csr.rs[i/context->stride]) && (B->mem.csr.cols[z] < B->mem.csr.cols[z-1]))
          {
            swap = B->mem.csr.cols[z-1];
            B->mem.csr.cols[z-1] = B->mem.csr.cols[z];
            B->mem.csr.cols[z] = swap;
            z -= 1;
          }
        }
      }

    }
  }

  B->m = (A->m+context->stride-1)/context->stride;
  B->nz = n_edges_new;
  #if MPF_PRINTOUT
    printf("stride: %d\n", context->stride);
    printf("n_edges_new: %d\n", n_edges_new);
  #endif
}
