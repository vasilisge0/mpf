#include "mpf.h"


/* seed */

void mpf_dsy_seed
(
  const MPF_SparseHandle A_handle,  /* (1) */
  const MPF_SparseDescr A_descr,    /* (2) */

  MPF_Int m_V,                      /* (3) rows V */
  MPF_Int n_V,                      /* (4) num_cols_V */
  MPF_Int m_H,                      /* (5) num_rows_H */
  MPF_Int n_H,                      /* (6) num_cols_H */
  MPF_Int n_B,                      /* (7) num_cols_rhs */

  double *U,                        /* (8) input */
  double *memory_defl               /* (10) */
)
{
  /* unpacking memory_defl */
  double *Vdefl = memory_defl;
  double *Hdefl = &Vdefl[m_V*n_V];
  double *refs_defl_array = &Hdefl[m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  //double *Mdefl = &Tdefl[m_V*n_B];

  mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, n_B, m_V, 1.0, Vdefl,
    m_V, U, m_V, 0.0, Tdefl, n_V);

  mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
  mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);

  mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_V, n_B, n_V, 1.0, Vdefl,
    m_V, Tdefl, m_V, 0.0, U, m_V);
}

/* ----------------------------- deflation ---------------------------------- */

/* assumes column major ordering */
void mpf_dsy_X_defl
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,        /* (1) rows V */
  MPF_Int n_V,        /* (3) num_cols_V */
  MPF_Int m_H,        /* (4) num_rows_H */
  MPF_Int n_H,        /* (5) num_cols_H */
  MPF_Int n_B,        /* (2) num_cols_rhs */

  double *U,          /* (6) input */
  double *W,          /* (6) output */
  double *memory_defl /* (8) */
)
{
  /* unpacking memory_defl */
  double *Vdefl = memory_defl;
  double *Hdefl = &Vdefl[m_V*n_V];
  double *refs_defl_array = &Hdefl[m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m_V*n_B];

  //new
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_handle, A_descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, U, n_B, m_V, 0.0, W, m_V);

  mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, n_B, m_V, 1.0, Vdefl,
    m_V, W, m_V, 0.0, Tdefl, n_V);

  printf("Mdefl[0]: %1.4E\n", Mdefl[0]);

  /* solves linear system */
  mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
  mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);

  printf("Mdefl[0]: %1.4E\n", Mdefl[0]);

  //mpf_matrix_d_announce(Tdefl, 10, 1, n_V, "Tdefl");

  mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_V, n_B, n_V, 1.0, Vdefl,
    m_V, Tdefl, m_V, 0.0, Mdefl, m_V);

  //mpf_matrix_d_announce(Mdefl, 10, 1, n_V, "Mdefl");

  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A_handle, A_descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_V, 1.0, W, m_V);

  //mpf_matrix_d_announce(W, 10, 1, m_V, "W (in x_defl [b])");
}

/* working version */
void mpf_dsy_X_defl_rec
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,        /* (1) rows V */
  MPF_Int n_V,        /* (3) num_cols_V */
  MPF_Int m_H,        /* (4) num_rows_H */
  MPF_Int n_H,        /* (5) num_cols_H */
  MPF_Int n_B,        /* (2) num_cols_rhs */

  double *U,          /* (6) input */
  double *W,          /* (6) output */
  double *memory_defl /* (8) */
)
{
  /* unpacking memory_defl */
  double *Vdefl = memory_defl;
  double *Hdefl = &Vdefl[m_V*n_V];
  double *refs_defl_array = &Hdefl[m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  double *Mdefl = &Tdefl[m_V*n_B];

  //new
  mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_handle, A_descr,
    SPARSE_LAYOUT_COLUMN_MAJOR, U, n_B, m_V, 0.0, Mdefl, m_V);

  mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, n_B, m_V, 1.0, Vdefl,
    m_V, Mdefl, m_V, 0.0, Tdefl, n_V);

  /* solves linear system */
  mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
  mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);

  mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_V, n_B, n_V, 1.0, Vdefl,
    m_V, Tdefl, n_V, 0.0, Mdefl, m_V);

  memcpy(W, U, (sizeof *W)*m_V*n_B);
  mpf_daxpy(m_V, -1.0, Mdefl, 1, W, 1);

  //mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_handle, A_descr,
  //  SPARSE_LAYOUT_COLUMN_MAJOR, Mdefl, n_B, m_V, 0.0, W, m_V);
}

void mpf_dsy_B_defl
(
  const MPF_SparseHandle A_handle,
  const MPF_SparseDescr A_descr,

  MPF_Int m_V,          /* (1) rows V */
  MPF_Int n_V,          /* (3) num_cols_V */
  MPF_Int m_H,          /* (4) num_rows_H */
  MPF_Int n_H,          /* (5) num_cols_H */
  MPF_Int n_B,          /* (2) num_cols_rhs */

  double *U,            /* (6) input */
  double *memory_defl   /* (8) */
)
{
  /* unpacking memory_defl */
  double *Vdefl = memory_defl;
  double *Hdefl = &Vdefl[m_V*n_V];
  double *refs_defl_array = &Hdefl[m_H*n_H];
  double *Tdefl = &refs_defl_array[m_H-1];
  //double *Mdefl = &Tdefl[m_V*n_B];

  mpf_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_V, n_B, m_V, 1.0, Vdefl,
    m_V, U, m_V, 0.0, Tdefl, n_V);

  /* solves linear system */
  mpf_qr_dsy_rhs_givens(m_H, n_H, 1, Hdefl, m_H, Tdefl, refs_defl_array);
  mpf_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    m_H, n_B, 1.0, Hdefl, n_H, Tdefl, n_V);

  mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_V, n_B, n_V, 1.0, Vdefl,
    m_V, Tdefl, n_V, 0.0, U, m_V);
}

/* ----------------------------- thresholing -------------------------------- */

void mpf_sparse_coo_d_threshold_apply
(
  MPF_Int blk_r,
  MPF_Int blk_c,
  MPF_Int I_r,
  MPF_Int I_c,
  MPF_Int m_V,
  MPF_Int n_V,
  double *M,
  MPF_Int ld_M,
  double threshold,
  MPF_Sparse *A,  /* requires coo format */
  MPF_Int memory_inc
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  for (i = 0; i < blk_c; ++i)
  {
    for (j = 0; j < blk_r; ++j)
    {
      if (fabs(M[ld_M*i+j]) > threshold)
      {
        if (A->nz == A->nz_max)
        {
          A->nz_max += memory_inc;
          A->mem.coo.rows = (MPF_Int*)mkl_realloc(A->mem.coo.rows,
              sizeof(MPF_Int)*A->nz_max);
          A->mem.coo.cols = (MPF_Int*)mkl_realloc(A->mem.coo.cols,
              sizeof(MPF_Int)*A->nz_max);
          A->mem.coo.data = (MPF_Int*)mkl_realloc(A->mem.coo.data,
              sizeof(double)*A->nz_max);
        }

        A->mem.coo.rows[A->nz] = I_r + j;
        A->mem.coo.cols[A->nz] = I_c + i;
        ((double *)A->mem.coo.data)[A->nz] = M[ld_M*i+j];
        A->nz+=1;
      }
    }
  }
}

/* -- Set of sparsified_ utiliy functions (used in sparsified, combressed  -- */
/* -- Krylov solvers)                                                      -- */

void vector_d_sparsify
(
  MPF_Int m_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_BucketArray *H
)
{
  MPF_Int i = 0;
  MPF_Int nz = 0;
  for (i = H->bins_start[color]; i != -1; i = H->next[i])
  {
    v_out_vector[nz] = v_in_vector[partition_size*H->values[i]+offset];
    nz += 1;
  }
}

void vecblk_d_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int nz_temp = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  for (j = 0; j < n_B; ++j)
  {
    color = (current_rhs+j)/partition_size;
    offset = (current_rhs+j)%partition_size;
    for (i = H->bins_start[color]; i != -1; i = H->next[i])
    {
      v_out_vector[nz_temp] =
        v_in_vector[m_B*j+partition_size*H->values[i]+offset];
      nz_temp += 1;
    }
  }
  *nz = nz_temp;
}

void vecblk_d_block_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  double *v_in_vector,
  double *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int blk = n_B;
  MPF_Int nz_temp = 0;
  for (j = 0; j < n_B; ++j)
  {
    color = (current_rhs+j)/partition_size;
    offset = ((current_rhs+j)%partition_size)/blk;
    for (i = H->bins_start[color]; i != -1; i = H->next[i])
    {
      for (k = 0; k < n_B; ++k)
      {
        v_out_vector[nz_temp] =
          v_in_vector[m_B*j+partition_size*H->values[i]+k+offset*blk];
        nz_temp += 1;
      }
    }
  }
  *nz = nz_temp;
}

void vector_z_sparsify
(
  MPF_Int m_v,
  MPF_ComplexDouble *v_in_vector,
  MPF_ComplexDouble *v_out_vector,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_BucketArray *H
)
{
  MPF_Int i = 0;
  MPF_Int nz = 0;
  for (i = H->bins_start[color]; i != -1; i = H->next[i])
  {
    v_out_vector[nz] = v_in_vector[partition_size*H->values[i]+offset];
    nz += 1;
  }
}

void vecblk_z_block_sparsify
(
  MPF_Int m_B,
  MPF_Int n_B,
  MPF_ComplexDouble *v_in_vector,
  MPF_ComplexDouble *v_out_vector,
  MPF_Int partition_size,
  MPF_Int current_rhs,
  MPF_BucketArray *H,
  MPF_Int *nz
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int blk = n_B;
  MPF_Int nz_temp = 0;
  for (j = 0; j < n_B; ++j)
  {
    color = (current_rhs+j)/partition_size;
    offset = ((current_rhs+j)%partition_size)/blk;
    for (i = H->bins_start[color]; i != -1; i = H->next[i])
    {
      for (k = 0; k < n_B; ++k)
      {
        v_out_vector[nz_temp] =
          v_in_vector[m_B*j+partition_size*H->values[i]+k+offset*blk];
        nz_temp += 1;
      }
    }
  }
  *nz = nz_temp;
}

/*============================================================================*/
/*     V_basis: compressed basis vector                                       */
/*    c_vector: coefficents                                                   */
/*           X: output                                                        */
/* blk = 1 for standard krylov methods                                        */
/*============================================================================*/
void krylov_dge_sparse_basis_combine
(
  MPF_Int m_V,
  MPF_Int n_V,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  double *c_vector,
  double *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int nz = 0;
  MPF_Int index = 0;
  for (i = 0; i < n_V; ++i)
  {
    nz = 0;
    for (j = H->bins_start[color]; j != -1 ; j = H->next[j])
    {
      index = partition_size*H->values[j]+offset;
      X[index] += V_basis[m_V*i+nz]*c_vector[i];
      nz += 1;
    }
  }
}


/*============================================================================*/
/*  V_basis: compressed basis vector                                          */
/* c_vector: coefficents                                                      */
/*        X: output                                                           */
/* @NOTE: Beware as it produces an entry-wise compressed vector block, and    */
/* as thus not the same dense X                                               */
/* that is produced from regular block_lanczos.                               */
/* blk = 1 for standard krylov methods                                        */
/*============================================================================*/
void block_krylov_dge_sparse_basis_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int nz = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int test = 0;
  MPF_Int index = 0;

  for (i = 0; i < n_V; ++i)
  {
    test = 0;
    color = (current_rhs+i%blk)/partition_size;
    offset = (current_rhs+i%blk)%partition_size;
    for (j = H->bins_start[color]; j != -1; j = H->next[j])
    {
      index = partition_size*H->values[j]+offset+(i%blk)*m_X;
      X[index] += V_basis[nz]*c_vector[m_c*(i%blk)+i];
      nz += 1;
    }
    test += 1;
  }
}

/*============================================================================*/
/*      V_basis: compressed basis vector                                      */
/*     c_vector: coefficents                                                  */
/*            X: output                                                       */
/* @NOTE: Beware as it produces an entry-wise compressed vector block, and as */
/* thus not the same dense X                                                  */
/* that is produced from regular block_lanczos.                               */
/* blk = 1 for standard krylov methods                                        */
/* because basis is the same for every column of the coefficents vecblk       */
/*============================================================================*/
void block_krylov_dge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Int z = 0;
  MPF_Int nz = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int index = 0;

  for (k = 0; k < blk; ++k)
  {
    nz = 0;
    for (i = 0; i < n_V; ++i)
    {
      color = (current_rhs+i%blk)/partition_size;
      //@DEBUG
      offset = ((current_rhs+i%blk)%partition_size)/blk;
      for (j = H->bins_start[color]; j != -1; j = H->next[j])
      {
        for (z = 0; z < blk; ++z)
        {
          index = partition_size*H->values[j] + z + m_X*k + blk*offset;
          X[index] += V_basis[nz]*c_vector[m_c*k+i];
          nz += 1;
        }
      }
    }
  }
}

void block_krylov_zge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int k = 0;
  MPF_Int z = 0;
  MPF_Int nz = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int index = 0;
  MPF_ComplexDouble tempf_c;

  for (k = 0; k < blk; ++k)
  {
    nz = 0;
    for (i = 0; i < n_V; ++i)
    {
      color = (current_rhs+i%blk)/partition_size;
      //@DEBUG
      offset = ((current_rhs+i%blk)%partition_size)/blk;
      for (j = H->bins_start[color]; j != -1; j = H->next[j])
      {
        for (z = 0; z < blk; ++z)
        {
          index = partition_size*H->values[j] + z + m_X*k + blk*offset;
          tempf_c = mpf_scalar_z_multiply(V_basis[nz], c_vector[m_c*k+i]);
          X[index] = mpf_scalar_z_add(X[index], tempf_c);
          nz += 1;
        }
      }
    }
  }
}

/*============================================================================*/
/*      V_basis: compressed basis vector                                      */
/*     c_vector: coefficents                                                  */
/*            X: output                                                       */
/* @NOTE: Beware as it produces an entry-wise compressed vector block, and as */
/* thus                                                                       */
/* not the same dense X that is produced from regular block_lanczos.          */
/* blk = 1 for standard krylov methods                                        */
/* basis is the same for every column of the coefficents vecblk               */
/*============================================================================*/
void global_krylov_dge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  double *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  double *c_vector,
  double *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int z = 0;
  MPF_Int nz = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int index;
  nz = 0;

  for (i = 0; i < n_V; ++i)
  {
    color = (current_rhs+i%blk)/partition_size;
    //@DEBUG
    offset = ((current_rhs+i%blk)%partition_size)/blk;
    for (j = H->bins_start[color]; j != -1; j = H->next[j])
    {
      for (z = 0; z < blk; ++z)
      {
        index = partition_size*H->values[j] + z + m_X*(i%blk) + blk*offset;
        X[index] += V_basis[nz]*c_vector[i/blk];
        nz += 1;
      }
    }
  }

  mpf_matrix_d_announce(X, blk, blk, m_X, "X");
}

/*============================================================================*/
/*     V_basis: compressed basis vector                                       */
/*    c_vector: coefficents                                                   */
/*           X: output                                                        */
/* @NOTE: Beware as it produces an entry-wise compressed vector block, and as */
/* thus not the same dense X that is produced from regular block_lanczos.     */
/* blk = 1 for standard krylov methods                                        */
/* basis is the same for every column of the coefficents vecblk               */
/*============================================================================*/
void global_krylov_zge_sparse_basis_block_combine
(
  MPF_Int current_rhs,
  MPF_Int n_V,
  MPF_Int blk,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int m_c,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int z = 0;
  MPF_Int nz = 0;
  MPF_Int color = 0;
  MPF_Int offset = 0;
  MPF_Int index;
  nz = 0;
  MPF_ComplexDouble tempf_c = mpf_scalar_z_init(0.0, 0.0);

  for (i = 0; i < n_V; ++i)
  {
    color = (current_rhs+i%blk)/partition_size;
    //@DEBUG
    offset = ((current_rhs+i%blk)%partition_size)/blk;
    for (j = H->bins_start[color]; j != -1; j = H->next[j])
    {
      for (z = 0; z < blk; ++z)
      {
        index = partition_size*H->values[j] + z + m_X*(i%blk) + blk*offset;
        tempf_c = mpf_scalar_z_multiply(V_basis[nz], c_vector[i/blk]);
        X[index] = mpf_scalar_z_add(X[index], tempf_c);
        nz += 1;
      }
    }
  }
}


//@NOTE: original code.
//void block_krylov_dge_sparse_basis_combine(MPF_Int current_rhs,
//                                           MPF_Int num_cols_V,
//                                           MPF_Int blk,
//                                           double *V_basis,
//                                           MPF_BucketArray *H,
//                                           MPF_Int partition_size,
//                                           MPF_Int num_rows_c,
//                                           double *c_vector,
//                                           double *X,
//                                           MPF_Int num_rows_X)
///*
//     V_basis: compressed basis vector
//    c_vector: coefficents
//           X: output
//
//@NOTE: Beware as it produces an entry-wise compressed vector block, and as
//thus not the same dense X
//that is produced from regular block_lanczos.
//*/
//// blk = 1 for standard krylov methods
//{
//    MPF_Int i = 0;
//    MPF_Int j = 0;
//    MPF_Int k = 0;
//    MPF_Int nz = 0;
//    MPF_Int color = 0;
//    MPF_Int offset = 0;
//    //for (k = 0; k < blk; k++)
//    //{
//        nz = 0;   // because basis is the same for every column of the
//                  //coefficents vecblk
//
//        for (i = 0; i < num_cols_V; i++)
//        {
//            color = (current_rhs+i%blk)/partition_size;
//            offset = (current_rhs+i%blk)%partition_size;
//            for (j = H->bins_start[color]; j != -1; j = H->next[j])
//            {
//
//                X[num_rows_X*k
//                +partition_size*(H->values[j])
//                //+(i%blk)
//                //+(k)%blk
//                +offset]
//                += V_basis[nz]*c_vector[num_rows_c*k+i];
//
//                if ((i == 1) && (nz == 352) && (k == 0))
//                {
//                    printf("    nz_0: %d\n", H->bins_size[0]);
//                    printf("Hdata[2]: %d\n", H->values[j]);
//                    printf("    id_x: %d\n", num_rows_X*k+partition_size*(H->values[j])+offset);
//                    printf("     val: %1.4E\n", V_basis[nz]*c_vector[num_rows_c*k+i]);
//                    mpf_matrix_d_announce(X, 40, 2, num_rows_X, "X");
//                }
//
//                nz += 1;
//            }
//        }
//    //}
//    printf("V_basis[1]: %1.4E\n", V_basis[1]);
//}

/*============================================================================*/
/*    V_basis: compressed basis vector                                        */
/*   c_vector: coefficents                                                    */
/*          X: output                                                         */
/* blk = 1 for standard krylov methods                                        */
/*============================================================================*/
void krylov_zge_sparse_basis_combine
(
  MPF_Int m_V,
  MPF_Int n_V,
  MPF_ComplexDouble *V_basis,
  MPF_BucketArray *H,
  MPF_Int partition_size,
  MPF_Int color,
  MPF_Int offset,
  MPF_ComplexDouble *c_vector,
  MPF_ComplexDouble *X,
  MPF_Int m_X
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int nz = 0;
  MPF_ComplexDouble tempf_complex = mpf_scalar_z_init(0.0, 0.0);
  for (i = 0; i < n_V; ++i)
  {
    nz = 0;
    for (j = H->bins_start[color]; j != -1 ; j = H->next[j])
    {
      tempf_complex = mpf_scalar_z_multiply(V_basis[m_V*i+nz], c_vector[i]);
      X[partition_size*H->values[j]+offset]
        = mpf_scalar_z_add(X[partition_size*H->values[j]+offset], tempf_complex);
      nz += 1;
    }
  }
}
