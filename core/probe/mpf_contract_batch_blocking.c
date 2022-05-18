#include "mpf.h"

void mpf_block_row_contract
(
  MPF_Int blk,
  MPF_Sparse *A,    /* input pattern in csr format */
  MPF_Int current_row,
  MPF_Int *tempf_array,    /* temporary storage array */
  MPF_Int *tempf_i_array,  /* provides inverted indexing to tempf_array */
  MPF_Sparse *C            /* output */
)
{
  MPF_Int swap = -1;
  MPF_Int max_val = -1;
  MPF_Int c = 0;
  MPF_Int z = 0;
  MPF_Int m_new = 0;

  MPF_Int nr = 0;
  MPF_Int nc = 0;
  sparse_index_base_t index;

  mkl_sparse_d_export_csr(A->handle, &index, &nr, &nc,
    &A->mem.csr.rs, &A->mem.csr.re, &A->mem.csr.cols,
    (double**)&A->mem.csr.data);

  C->descr = A->descr;
  C->mem.csr.re[0] = 0;

  MPF_Int r = 0;
  for (MPF_Int i = 0; i < C->m; ++i)
  {
    r = i/blk;

    /* initialize Ablk_csr->rs and Ablk_csr->re arrays */
    if ((r > 0) && (i % blk == 0))
    {
      /* reset only the previously accesed entries of tempf_array and */
      /* tempf_inverted_array                                         */
      for (MPF_Int j = 0; j < m_new; ++j)
      {
        tempf_array[tempf_i_array[j]] = 0;
        tempf_i_array[j] = -1;
      }
      m_new = 0; /* reset n_nodes_new */
      C->mem.csr.rs[r] = C->mem.csr.re[r-1];
      C->mem.csr.re[r] = C->mem.csr.rs[r];
    }

    /* access neighbors of ith node of G(A_csr) */
    for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
    {
      c = A->mem.csr.cols[j]; /* id of j-th neighbor */
      if (tempf_array[c/blk] == 0)
      {
        C->mem.csr.cols[C->mem.csr.re[r]] = c/blk;  /* new neightbor block */
        tempf_array[c/blk] = 1;       /* marks block node */
        tempf_i_array[m_new] = c/blk; /* adds marked block node to the inverted index array */
        z = C->mem.csr.re[r];
        if (m_new > max_val)
        {
          max_val = m_new;
        }
        m_new += 1;
        ((double*)C->mem.csr.data)[C->mem.csr.re[r]] = 1.0;
        while ((z > C->mem.csr.re[r]) &&
               (C->mem.csr.cols[z] < C->mem.csr.cols[z-1]))
        {
          swap = C->mem.csr.cols[z-1];
          C->mem.csr.cols[z-1] = C->mem.csr.cols[z];
          C->mem.csr.cols[z] = swap;
          z -= 1;
        }
        C->mem.csr.re[r] += 1;
      }
    }
  }

  C->m = (C->m+blk-1)/blk;
  C->n = (C->n+blk-1)/blk;
  C->nz = C->mem.csr.re[C->m-1];
  for (MPF_Int j = 0; j < m_new; ++j)
  {
    tempf_array[tempf_i_array[j]] = 0;
    tempf_i_array[j] = -1;
  }
  m_new = 0;
}
