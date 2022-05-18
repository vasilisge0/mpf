#include "mpf.h"

void mpf_contract_block_hybrid
(
  MPF_Int blk,

  /* used for checking overlapping rows */
  MPF_Int *temp_array,      /* temporary storage array */
  MPF_Int *temp_i_array,    /* provides inverted indexing to temp_array */

  /* bucket arrays for current colorings T and for merging M */
  MPF_BucketArray *T,
  MPF_Sparse *A,            /* input pattern in csr format */
  MPF_BucketArray *M,
  MPF_Sparse *C             /* coarsened pattern */
)
{
  MPF_Int swap = -1;      /* for swapping overlapping row entries */
  MPF_Int c = 0;          /* for accessing columns */
  MPF_Int z = 0;          /* (?) */
  MPF_Int m_init = A->m;  /* initial number of rows */
  MPF_Int m_new = 0;      /* new (coarsened) number of rows */

  /* initialize rs and re */
  C->mem.csr.rs[0] = 0;
  C->mem.csr.re[0] = 0;

  /* coarsened rows and columns */
  C->m = (A->m+blk-1)/blk;
  C->n = (A->n+blk-1)/blk;

  /* buffer holding new cols to be merged */
  MPF_Int buffer[blk];

  /* initialize arrays used for checking overlapping rows/cols */
  mpf_zeros_i_set(MPF_COL_MAJOR, C->m, 1, temp_array, C->m);
  mpf_matrix_i_set(MPF_COL_MAJOR, C->m, 1, temp_i_array, C->m, -1);

  MPF_Int min = 0;

  /* parsing number of bins (which means number of colors) */
  printf("nbins: %d\n", T->n_bins);
  for (MPF_Int k = 0; k < T->n_bins; ++k)
  {
    MPF_Int count = 0;

    /* deplete one bin */
    for (MPF_Int p = T->bins_start[k]; p != T->bins_end[k]; p = T->next[k])
    {
      MPF_Int i = T->values[p];
      buffer[count] = i;
      count += 1;

      /* initialize C->rs and C->re arrays */
      if ((count == blk) || (p == T->bins_end[k]))
      {
        /* reset only the previously accesed entries of temp_array and */
        /* temp_inverted_array                                         */
        for (MPF_Int j = 0; j < m_new; ++j)
        {
          temp_array[temp_i_array[j]] = 0;
          temp_i_array[j] = -1;
        }

        m_new = 0;
        count = 0;
        min += 1;
        C->mem.csr.rs[min] = C->mem.csr.re[min-1];
        C->mem.csr.re[min] = C->mem.csr.rs[min];
      }

      /* access neighbors of ith node of G(A_csr) */
      for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
      {
        c = A->mem.csr.cols[j]; /* id of j-th neighbor */
        if (temp_array[min] == 0)
        {
          C->mem.csr.cols[C->mem.csr.re[min]] = c/blk;  /* new block-neightbor */
          temp_array[min] = 1;                          /* marks block node */
          temp_i_array[m_new] = min;                    /* adds marked block node to the inverted index array */
          z = C->mem.csr.re[min];
          C->mem.csr.re[min] += 1;
          m_new += 1;

          /* reorder columns */
          while ((z > C->mem.csr.rs[min]) &&
                 (C->mem.csr.cols[z] < C->mem.csr.cols[z-1]))
          {
            swap = C->mem.csr.cols[z-1];
            C->mem.csr.cols[z-1] = C->mem.csr.cols[z];
            C->mem.csr.cols[z] = swap;
            z -= 1;
          }
        }
      }

      /* insert to M BucketArray the entries stored in the buffer */
      mpf_bucket_array_insert(M, min, i);
    }
  }
}
