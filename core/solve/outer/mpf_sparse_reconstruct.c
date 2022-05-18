#include "mpf.h"

/* ------------------------ sparse reconstruction --------------------------- */

void mpf_spai_init
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Sparse *P = solver->Pmask;
  MPF_Sparse *fA = (MPF_Sparse*)solver->fA;
  MPF_Int blk_fA = solver->blk_fA;
  MPF_Int m_B = solver->ld;

  mpf_get_max_nrhs(probe, solver);
  solver->n_batches = (solver->n_max_B+solver->batch-1) / solver->batch;

  /* row counters */
  MPF_Int nnz = 0;
  MPF_Int nnz_row = 0;
  MPF_Int count = 0;

  MPF_Int *buffer = (MPF_Int*)solver->buffer;
  MPF_Int *buffer_rev = &((MPF_Int*)solver->buffer)[m_B]; // need to allocate that

  fA->mem.csr.rs[0] = 0;
  fA->mem.csr.re[0] = 0;

  for (MPF_Int i = 0; i < fA->m; i += blk_fA) /* parses rows of fA */
  {
    mpf_zeros_i_set(MPF_COL_MAJOR, m_B, 1, buffer, m_B);  // should change m_B numbers (not ld) to nnz
    mpf_matrix_i_set(MPF_COL_MAJOR, m_B, 1, buffer_rev, m_B, -1);
    mpf_row_contract(i, blk_fA, &count, P, buffer, buffer_rev);

    for (MPF_Int j = 0; j < blk_fA; ++j)  /* parses block_size entries in row */
    {
      fA->mem.csr.rs[i+j] = nnz;
      fA->mem.csr.re[i+j] = nnz;
      nnz_row = 0;

      for (MPF_Int k = 0; k < count; ++k) /* parses only nonzero blocks */
      {
        nnz_row += (1-buffer_rev[k]/(m_B/blk_fA))*blk_fA + (buffer_rev[k]/(m_B/blk_fA))*(m_B%blk_fA);
      }
      nnz += nnz_row;
    }
  }
}

void mpf_spai_d_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Sparse *P = solver->Pmask;
  MPF_Sparse *fA = (MPF_Sparse*)solver->fA;
  MPF_Int BLK_MAX_fA = solver->max_blk_fA;
  MPF_BucketArray *H = &solver->color_to_node_map;
  MPF_Int *buffer = (MPF_Int*)solver->buffer;
  MPF_Int *buffer_rev = &buffer[solver->ld];

  for (MPF_Int i = 0; i < solver->batch; ++i)
  {
    MPF_Int color_start = (i+solver->current_rhs)/BLK_MAX_fA;
    MPF_Int color_end = (i+solver->current_rhs+solver->current_batch-1)/BLK_MAX_fA;

    for (MPF_Int color = color_start; color <= color_end; ++color)
    {
      /* same-color nodes */
      for (MPF_Int node = H->bins_start[color]; node != -1; node = H->next[node])
      {
        MPF_Int r_prev = H->values[node];
        MPF_Int r_A = r_prev*BLK_MAX_fA+i;

        /* testing now */
        mpf_d_reconstruct_blk_row_sy(r_prev, r_A, P, fA,
          buffer, buffer_rev, &((double*)solver->X.data)[solver->ld*i], fA->nz, fA->m,
          solver->B.m, solver->max_blk_fA, solver->blk_fA, i+solver->current_rhs);
      }
    }
  }
}

void mpf_d_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  double *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
)
{
  MPF_Int i_fA = 0;
  MPF_Int c_prev = 0;
  MPF_Int r_fA = 0;
  MPF_Int c_fA = 0;
  MPF_Int blk_fA_n = 0;
  MPF_Int m_fA11 = m-(m%blk_max);
  MPF_Int blk_up = 0;
  MPF_Int nnz = 0;
  //MPF_Int k = 0;

  r_fA = blk_max*r_prev + curr_rhs%blk_max;
  mpf_zeros_i_set(MPF_COL_MAJOR, m, 1, row, m);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, row_rev, m, -1);
  mpf_row_contract((r_fA/blk)*blk, blk, &nnz, P, row, row_rev);


  if (blk_max > 1)
  {
    if (m % blk_max)
    {
      blk_up = (m%blk);
    }
    else
    {
      blk_up = blk;
    }
  }
  else
  {
    blk_up = blk;
  }

  for (MPF_Int i = 0; i < nnz; ++i)
  {
    c_fA = blk*row_rev[i];
    c_prev = c_fA/blk_max;

    blk_fA_n = (1-c_prev/(m_P-1))                       * blk
             + (c_prev/(m_P-1))   * (1-c_fA/(m_fA11+1)) * blk
             + (c_prev/(m_P-1))   * (c_fA/(m_fA11+1))   * blk_up;

    for (MPF_Int t = 0; t < blk_fA_n; ++t)
    {
      if (r_fA <= c_fA+t)
      {
        i_fA = fA->mem.csr.re[r_fA];
        fA->mem.csr.cols[i_fA] = c_fA+t;
        ((double *)fA->mem.csr.data)[i_fA] = X[blk*(row_rev[i]) + t];
        fA->mem.csr.re[r_fA] += 1;

        /* low triangular off-diagonal entries of blocks */
        if (r_fA < c_fA+t)
        {
          i_fA = fA->mem.csr.re[c_fA+t];
          fA->mem.csr.cols[i_fA] = r_fA;
          ((double *)fA->mem.csr.data)[i_fA] = X[blk*(row_rev[i]) + t];
          fA->mem.csr.re[c_fA+t] += 1;
        }
      }
    }
  }
}

void mpf_d_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Dense *diag_fA = (MPF_Dense*)solver->fA;
  mpf_d_select(solver->max_blk_fA, probe->colorings_array, solver->X.m,
    (double*)solver->X.data, solver->blk_fA, solver->current_rhs, solver->current_batch);
  solver->buffer = solver->X.data;

  for (MPF_Int j = 0; j < solver->current_batch; ++j)
  {
    mpf_daxpy(solver->ld, 1.0, &(((double*)(solver->X.data))[solver->ld*j]), 1,
      &(((double*)(diag_fA->data))[solver->ld*((j+solver->current_rhs)%solver->blk_fA)]), 1);
  }
}

void mpf_spai_zhe_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Sparse *P = solver->Pmask;
  MPF_Sparse *fA = (MPF_Sparse*)solver->fA;
  MPF_Int BLK_MAX_fA = solver->max_blk_fA;
  MPF_BucketArray *H = &solver->color_to_node_map;
  MPF_Int *buffer = (MPF_Int*)solver->buffer;
  MPF_Int *buffer_rev = &buffer[solver->ld];

  for (MPF_Int i = 0; i < solver->batch; ++i)
  {
    MPF_Int color_start = (i+solver->current_rhs)/BLK_MAX_fA;
    MPF_Int color_end = (i+solver->current_rhs+solver->current_batch-1)/BLK_MAX_fA;

    for (MPF_Int color = color_start; color <= color_end; ++color)
    {
      /* same-color nodes */
      for (MPF_Int node = H->bins_start[color]; node != -1; node = H->next[node])
      {
        MPF_Int r_prev = H->values[node];
        MPF_Int r_A = r_prev*BLK_MAX_fA+i;

        /* testing now */
        mpf_zhe_reconstruct_blk_row_sy(r_prev, r_A, P, fA,
          buffer, buffer_rev, (MPF_ComplexDouble*)solver->X.data, fA->nz, fA->m,
          solver->B.m, solver->max_blk_fA, solver->blk_fA, i+solver->current_rhs);
      }
    }
  }
}

void mpf_zhe_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  MPF_ComplexDouble *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
)
{
  MPF_Int i_fA = 0;
  MPF_Int c_prev = 0;
  MPF_Int r_fA = 0;
  MPF_Int c_fA = 0;
  MPF_Int blk_fA_n = 0;
  MPF_Int m_fA11 = m-(m%blk_max);
  MPF_Int blk_up = 0;
  MPF_Int nnz = 0;
  //MPF_Int k = 0;

  r_fA = blk_max*r_prev + curr_rhs%blk_max;
  mpf_zeros_i_set(MPF_COL_MAJOR, m, 1, row, m);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, row_rev, m, -1);
  mpf_row_contract((r_fA/blk)*blk, blk, &nnz, P, row, row_rev);

  if (blk_max > 1)
  {
    if (m % blk_max)
    {
      blk_up = (m%blk);
    }
    else
    {
      blk_up = blk;
    }
  }
  else
  {
    blk_up = blk;
  }

  for (MPF_Int i = 0; i < nnz; ++i)
  {
    c_fA = blk*row_rev[i];
    c_prev = c_fA/blk_max;

    blk_fA_n = (1-c_prev/(m_P-1))                       * blk
             + (c_prev/(m_P-1))   * (1-c_fA/(m_fA11+1)) * blk
             + (c_prev/(m_P-1))   * (c_fA/(m_fA11+1))   * blk_up;

    for (MPF_Int t = 0; t < blk_fA_n; ++t)
    {
      if (r_fA <= c_fA+t)
      {
        i_fA = fA->mem.csr.re[r_fA];
        fA->mem.csr.cols[i_fA] = c_fA+t;
        ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA].real = X[blk*(row_rev[i]) + t].real;
        ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA].imag = X[blk*(row_rev[i]) + t].imag;
        fA->mem.csr.re[r_fA] += 1;

        /* low triangular off-diagonal entries of blocks */
        if (r_fA < c_fA+t)
        {
          i_fA = fA->mem.csr.re[c_fA+t];
          fA->mem.csr.cols[i_fA] = r_fA;
          ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA].real = X[blk*(row_rev[i]) + t].real;
          ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA].imag = -X[blk*(row_rev[i]) + t].imag;
          fA->mem.csr.re[c_fA+t] += 1;
        }
      }
    }
  }
}

void mpf_spai_zsy_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Sparse *P = solver->Pmask;
  MPF_Sparse *fA = (MPF_Sparse*)solver->fA;
  MPF_Int BLK_MAX_fA = solver->max_blk_fA;
  MPF_BucketArray *H = &solver->color_to_node_map;
  MPF_Int *buffer = (MPF_Int*)solver->buffer;
  MPF_Int *buffer_rev = &buffer[solver->ld];

  for (MPF_Int i = 0; i < solver->batch; ++i)
  {
    MPF_Int color_start = (i+solver->current_rhs)/BLK_MAX_fA;
    MPF_Int color_end = (i+solver->current_rhs+solver->current_batch-1)/BLK_MAX_fA;

    for (MPF_Int color = color_start; color <= color_end; ++color)
    {
      /* same-color nodes */
      for (MPF_Int node = H->bins_start[color]; node != -1; node = H->next[node])
      {
        MPF_Int r_prev = H->values[node];
        MPF_Int r_A = r_prev*BLK_MAX_fA+i;

        /* testing now */
        mpf_zsy_reconstruct_blk_row_sy(r_prev, r_A, P, fA,
          buffer, buffer_rev, (MPF_ComplexDouble*)solver->X.data, fA->nz, fA->m,
          solver->B.m, solver->max_blk_fA, solver->blk_fA, i+solver->current_rhs);
      }
    }
  }
}

void mpf_zsy_reconstruct_blk_row_sy
(
  MPF_Int r_prev,
  MPF_Int r_A,
  MPF_Sparse *P,
  MPF_Sparse *fA,
  MPF_Int *row,
  MPF_Int *row_rev,
  MPF_ComplexDouble *X,
  MPF_Int nnz_row,
  MPF_Int m,
  MPF_Int m_P,
  MPF_Int blk_max,
  MPF_Int blk,
  MPF_Int curr_rhs
)
{
  MPF_Int i_fA = 0;
  MPF_Int c_prev = 0;
  MPF_Int r_fA = 0;
  MPF_Int c_fA = 0;
  MPF_Int blk_fA_n = 0;
  MPF_Int m_fA11 = m-(m%blk_max);
  MPF_Int blk_up = 0;
  MPF_Int nnz = 0;
  //MPF_Int k = 0;

  r_fA = blk_max*r_prev + curr_rhs%blk_max;
  mpf_zeros_i_set(MPF_COL_MAJOR, m, 1, row, m);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, row_rev, m, -1);
  mpf_row_contract((r_fA/blk)*blk, blk, &nnz, P, row, row_rev);

  if (blk_max > 1)
  {
    if (m % blk_max)
    {
      blk_up = (m%blk);
    }
    else
    {
      blk_up = blk;
    }
  }
  else
  {
    blk_up = blk;
  }

  for (MPF_Int i = 0; i < nnz; ++i)
  {
    c_fA = blk*row_rev[i];
    c_prev = c_fA/blk_max;

    blk_fA_n = (1-c_prev/(m_P-1))                       * blk
             + (c_prev/(m_P-1))   * (1-c_fA/(m_fA11+1)) * blk
             + (c_prev/(m_P-1))   * (c_fA/(m_fA11+1))   * blk_up;

    for (MPF_Int t = 0; t < blk_fA_n; ++t)
    {
      if (r_fA <= c_fA+t)
      {
        i_fA = fA->mem.csr.re[r_fA];
        fA->mem.csr.cols[i_fA] = c_fA+t;
        ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA] = X[blk*(row_rev[i]) + t];
        fA->mem.csr.re[r_fA] += 1;

        /* low triangular off-diagonal entries of blocks */
        if (r_fA < c_fA+t)
        {
          i_fA = fA->mem.csr.re[c_fA+t];
          fA->mem.csr.cols[i_fA] = r_fA;
          ((MPF_ComplexDouble *)fA->mem.csr.data)[i_fA] = X[blk*(row_rev[i]) + t];
          fA->mem.csr.re[c_fA+t] += 1;
        }
      }
    }
  }
}

void mpf_z_reconstruct
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Dense *diag_fA = (MPF_Dense*)solver->fA;
  mpf_d_select(solver->max_blk_fA, probe->colorings_array, probe->m,
    (double*)solver->X.data, solver->blk_fA, solver->current_rhs, solver->current_batch);

  solver->buffer = solver->X.data;

  MPF_ComplexDouble ONE = mpf_scalar_z_init(1.0, 0.0);

  for (MPF_Int j = 0; j < solver->current_batch; ++j)
  {
    mpf_zaxpy(solver->ld, &ONE, &(((MPF_ComplexDouble*)(solver->X.data))[solver->ld*j]), 1,
      &(((MPF_ComplexDouble*)(diag_fA->data))[solver->ld*((j+solver->current_rhs)%solver->blk_fA)]), 1);
  }
}

MPF_Int mpf_spai_get_nz
(
  MPF_Sparse *A,
  MPF_Int blk_rec,
  MPF_Int *temp_array,
  MPF_Int *temp_i_array
)
{
  MPF_Int global_nz = 0;
  MPF_Int nz = 0;

  mpf_matrix_i_set(MPF_COL_MAJOR, A->m, 1, temp_array, A->m, 0);
  mpf_matrix_i_set(MPF_COL_MAJOR, A->m, 1, temp_i_array, A->m, -1);

  MPF_Int count = 0;
  for (MPF_Int i = 0; i < A->m; ++i)
  {
    if (((i % blk_rec) == 0) && (i > 0))
    {
      global_nz += nz;
      nz = 0;
      for (MPF_Int j = 0; j < count; ++j)
      {
        temp_array[temp_i_array[j]] = 0;
        temp_i_array[j] = -1;
      }
      count = 0;
    }

    for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
    {
      MPF_Int c = A->mem.csr.cols[j];
      if (temp_array[c/blk_rec] == 0)
      {
        temp_array[c] = 1;
        temp_i_array[count] = c;

        MPF_Int blk_row = (1-((i/blk_rec)+1)/(A->m/blk_rec))*blk_rec + (((i/blk_rec)+1)/(A->m/blk_rec))*(A->m-(i/blk_rec)*blk_rec);
        MPF_Int blk_col = (1-((c/blk_rec)+1)/(A->m/blk_rec))*blk_rec + (((c/blk_rec)+1)/(A->m/blk_rec))*(A->m-(c/blk_rec)*blk_rec);

        if ((blk_row == 0) || (blk_col == 0))
        {
          printf("\n >>> ERROR <<< \n");
        }

        nz += blk_row*blk_col;
        count += 1;
      }
    }
  }

  if (nz > 0)
  {
    global_nz += nz;
  }
  printf("global_nz: %d\n", global_nz);
  printf("total: %d\n", A->mem.csr.re[0]-A->mem.csr.rs[0]+A->mem.csr.re[1]-A->mem.csr.rs[1]);
  printf("%d: %d\n", A->mem.csr.re[0], A->mem.csr.rs[0]);
  printf("%d: %d\n", A->mem.csr.re[1], A->mem.csr.rs[1]);
  printf("blk_rec: %d\n", blk_rec);

  return global_nz;
}
