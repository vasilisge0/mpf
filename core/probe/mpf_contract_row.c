#include "mpf.h"

void mpf_row_contract
(
  MPF_Int curr_rhs,
  MPF_Int stride,
  MPF_Int *n_blocks,
  MPF_Sparse *P,
  MPF_Int *row,
  MPF_Int *row_rev
)
{
  MPF_Int count = 0;
  for (MPF_Int i = 0; i < stride; ++i)
  {
    for (MPF_Int j = P->mem.csr.rs[i+curr_rhs]; j < P->mem.csr.re[i+curr_rhs]; ++j)
    {
      if ((row_rev[count] == -1) && (row[P->mem.csr.cols[j]/stride] != 1))
      {
        row[P->mem.csr.cols[j]/stride] = 1;
        row_rev[count] = P->mem.csr.cols[j]/stride;
        count += 1;
      }
    }
  }
  *n_blocks = count;
}
