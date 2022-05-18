#include "mpf.h"

void mpf_block_contract
(
  MPF_Int blk,
  MPF_Int *temp_array,      /* temporary storage array */
  MPF_Int *temp_i_array,    /* provides inverted indexing to temp_array */
  MPF_Sparse *A,            /* input pattern in csr format */
  MPF_Sparse *C
)
{
  MPF_Int swap = -1;
  MPF_Int c = 0;
  MPF_Int z = 0;
  MPF_Int m_init = A->m;
  MPF_Int m_new = 0;

  C->mem.csr.rs[0] = 0;
  C->mem.csr.re[0] = 0;

  C->m = (A->m+blk-1)/blk;
  C->n = (A->n+blk-1)/blk;

  mpf_zeros_i_set(MPF_COL_MAJOR, C->m, 1, temp_array, C->m);
  mpf_matrix_i_set(MPF_COL_MAJOR, C->m, 1, temp_i_array, C->m, -1);

  /* parses through nodes of every line */
  for (MPF_Int i = 0; i < m_init; ++i)
  {
    /* initialize C_csr->rs and Ablk_csr->re arrays */
    if ((i > 0) && (i % blk == 0))
    {
      /* reset only the previously accesed entries of temp_array and */
      /* temp_inverted_array                                         */
      for (MPF_Int j = 0; j < m_new; ++j)
      {
        temp_array[temp_i_array[j]] = 0;
        temp_i_array[j] = -1;
      }
      m_new = 0; /* reset n_nodes_new */
      C->mem.csr.rs[i/blk] = C->mem.csr.re[i/blk-1];
      C->mem.csr.re[i/blk] = C->mem.csr.rs[i/blk];
    }

    /* access neighbors of ith node of G(A_csr) */
    for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
    {
      c = A->mem.csr.cols[j]; /* id of j-th neighbor */
      if (temp_array[c/blk] == 0)
      {
        C->mem.csr.cols[C->mem.csr.re[i/blk]] = c/blk;  /* new neightbor block */
        temp_array[c/blk] = 1;           /* marks block node */
        temp_i_array[m_new] = c/blk;     /* adds marked block node to the inverted index array */
        z = C->mem.csr.re[i/blk];
        C->mem.csr.re[i/blk] += 1;
        m_new += 1;
        while ((z > C->mem.csr.rs[i/blk]) &&
               (C->mem.csr.cols[z] < C->mem.csr.cols[z-1]))
        {
          swap = C->mem.csr.cols[z-1];
          C->mem.csr.cols[z-1] = C->mem.csr.cols[z];
          C->mem.csr.cols[z] = swap;
          z -= 1;
        }
      }
    }
  }

  C->nz = C->mem.csr.re[C->m-1];
  for (MPF_Int j = 0; j < m_new; ++j)
  {
    temp_array[temp_i_array[j]] = 0;
    temp_i_array[j] = -1;
  }
}
