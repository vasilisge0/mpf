#include "mpf.h"

void mpf_contract_sampling
(
  MPF_Probe *context,
  MPF_Sparse *A,
  MPF_Sparse *Asample
)
{
  MPF_Int n_edges_new = 0;

  Asample->mem.csr.rs[0] = 0;
  Asample->mem.csr.re[0] = 0;

  for (MPF_Int i = 0; i < context->P.m; i += context->stride) /* pass through each row */
  {
    if (i > 0)
    {
      Asample->mem.csr.rs[i/context->stride]
        = Asample->mem.csr.re[i/context->stride-1];
    }

    for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
    {
      MPF_Int c = A->mem.csr.cols[j];
      if (c % context->stride == 0)
      {
        Asample->mem.csr.cols[n_edges_new] = c/context->stride;
        n_edges_new++;
        Asample->mem.csr.re[i/context->stride] = n_edges_new;
      }
    }

    #if DEBUG == 1
      printf("i; %d -> %d : [%d, %d]\n", i, i/context->stride,
        Asample->mem.csr.rs[i/context->stride], Asample->mem.csr.re[i/context->stride]);
    #endif
  }
  context->P.m = (context->P.m+context->stride-1)/context->stride;
  context->P.nz = n_edges_new;
}
