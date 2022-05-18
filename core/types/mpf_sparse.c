#include "mpf.h"

void mpf_sparse_coo_to_csr_convert
(
  MPF_Sparse *A,
  MPF_Sparse *B
)
{
  MPF_Int status = mkl_sparse_convert_csr(A->handle,
      SPARSE_OPERATION_NON_TRANSPOSE, &B->handle);
  B->export_mem_function(B);
}

void mpf_sparse_coo_free
(
  MPF_Sparse *A
)
{
  mpf_free(A->mem.coo.rows);
  mpf_free(A->mem.coo.cols);
  mpf_free(A->mem.coo.data);
}

void mpf_sparse_mem_csr_free
(
  MPF_Sparse *A
)
{
  mpf_free(A->mem.csr.rs);
  mpf_free(A->mem.csr.re);
  mpf_free(A->mem.csr.cols);
  mpf_free(A->mem.csr.data);
}

void mpf_sparse_coo_alloc
(
  MPF_Sparse *A
)
{
  A->mem.coo.rows = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->nz);
  A->mem.coo.cols = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->nz);

  if (A->data_type == MPF_REAL)
  {
    A->mem.coo.data = mpf_malloc(sizeof(double)*A->nz);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    A->mem.coo.data = mpf_malloc(sizeof(MPF_ComplexDouble)*A->nz);
  }
}

void mpf_sparse_csr_alloc
(
  MPF_Sparse* A_in
)
{
  MPF_Sparse *A = (MPF_Sparse*) A_in;

  A->mem.csr.rs = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);
  A->mem.csr.re = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);
  A->mem.csr.cols = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->nz);

  if (A->data_type == MPF_REAL)
  {
    A->mem.csr.data = mpf_malloc(sizeof(double)*A->nz);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    A->mem.csr.data = mpf_malloc(sizeof(MPF_ComplexDouble)*A->nz);
  }
}

/*==========================================*/
/* == sparse matrix allocation functions == */
/*==========================================*/

MPF_SparseCoo* mpf_sparse_coo_array_create
(
  MPF_Int array_length
)
{
  MPF_SparseCoo* A = (MPF_SparseCoo*)mpf_malloc(sizeof(MPF_SparseCoo)*array_length);
  MPF_Int i =0;
  for (i = 0; i < array_length; ++i)
  {
    A[i].rows = NULL;
    A[i].cols = NULL;
    A[i].data = NULL;
  }
  return A;
}

MPF_SparseCoo* mpf_sparse_coo_array_alloc
(
  MPF_Int array_length
)
{
  MPF_SparseCoo* A = (MPF_SparseCoo*)mpf_malloc(sizeof(MPF_SparseCoo)*array_length);
  MPF_Int i =0;
  for (i = 0; i < array_length; ++i)
  {
    A[i].rows = NULL;
    A[i].cols = NULL;
    A[i].data = NULL;
  }
  return A;
}

MPF_SparseCsr* mpf_sparse_csr_array_create
(
  MPF_Int array_length
)
{
  MPF_SparseCsr* A = (MPF_SparseCsr*)mpf_malloc(sizeof(MPF_SparseCsr)*array_length);
  MPF_Int i = 0;
  for (i = 0; i < array_length; ++i)
  {
    A[i].rs = NULL;
    A[i].re = NULL;
    A[i].cols = NULL;
    A[i].data = NULL;
  }
  return A;
}

MPF_SparseCsr* mpf_sparse_csr_array_alloc
(
  MPF_Int array_length
)
{
  MPF_SparseCsr* A = (MPF_SparseCsr*)mpf_malloc(sizeof(MPF_SparseCsr)*array_length);

  MPF_Int i = 0;
  for (i = 0; i < array_length; i++)
  {
    A[i].rs = NULL;
    A[i].re = NULL;
    A[i].cols = NULL;
    A[i].data = NULL;
  }
  return A;
}

MPF_SparseHandle* mpf_sparse_handle_create
(
  MPF_Int array_length
)
{
  return (MPF_SparseHandle*)mpf_malloc(sizeof(MPF_SparseHandle)*array_length);
}

void mpf_sparse_array_free
(
  MPF_Sparse *A_array,
  MPF_Int length
)
{
  if (A_array != NULL)
  {
    for (MPF_Int i = 0; i < length; i++)
    {
      if (A_array[i].mem.csr.rs != NULL)
      {
        mpf_free(A_array[i].mem.csr.rs);
        A_array[i].mem.csr.rs = NULL;
      }
      if (A_array[i].mem.csr.cols != NULL)
      {
        mpf_free(A_array[i].mem.csr.cols);
        A_array[i].mem.csr.cols = NULL;
      }
    }
  }
}

/*============================================================================*/
/* mpf_sparse_csr_export                                                       */
/* Exports context->A.csr to seleted pattern of context depending on value of */
/* export_target:                                                             */
/*                                                                            */
/*            MPF_A => context->Ainput_handle -> context->A .csr              */
/*      MPF_A_INPUT => context->Ainput_handle -> context->Ainput.csr         */
/* MPF_SP_FA => context->Ainput_handle -> context->fA.csr              */
/*            MPF_M => context->Ainput_handle -> context->M.csr               */
/*============================================================================*/

void mpf_sparse_export_csr
(
  MPF_Sparse* A
)
{
  printf("exporting\n");
  mpf_sparse_d_export_csr(A->handle, &A->index,
    &A->m, &A->n, &A->mem.csr.rs, &A->mem.csr.re, &A->mem.csr.cols,
    (double**)&A->mem.csr.data);
}

//void mpf_sparse_csr_export
//(
//  MPF_Context *context,
//  MPF_Target export_target
//)
//{
//  sparse_index_base_t tempf_indexing = 0;
//
//  if (export_target == MPF_A)  // internal represenation
//  {
//    if (context->solver_interface == MPF_SOLVER_CPU)
//    {
//      /* DO NOT ALLOCATE MEMORY A_csr*/
//      if (context->data_type == MPF_REAL)
//      {
//        MPF_SparseCsr *A = &context->A.csr;
//        mpf_sparse_d_export_csr(context->A_handle, &tempf_indexing,
//          &A->m, &A->n, &A->rs, &A->re, &(A->cols),
//          (double**)&A->data);
//        //printf("A->rs[0]: %d, A->re[0]\n", A->rs[0], A->re[0]);
//      }
//      else if (context->data_type == MPF_COMPF_LEX)
//      {
//        MPF_SparseCsr *A = &context->A.csr;
//        mpf_sparse_z_export_csr(context->A_handle, &tempf_indexing,
//          &context->m_A, &context->n_A, &A->rs, &(A->re),
//          &A->cols, (MPF_ComplexDouble**)&(A->data));
//      }
//      //context->m_A = context->Ainput.coo.m;
//      //context->nz_A = context->Ainput.coo.nz;
//    }
////    else if (context->solver_interface == MPF_SOLVER_CUDA)
////    {
////      MPF_SparseCsr *A = &context->A.csr;
////      MPF_SparseCsr_Cuda *A_gpu = &context->A_gpu.csr;
////
////      /* @NOTE: DO NOT ALLOCATE MEMORY FOR A_csr (!) */
////      /* memory for coo matrix configuration is used */
////      if (context->data_type == MPF_REAL)
////      {
////        context->cuda_buffer = mpf_malloc(sizeof(double)*A->nz);
////        mpf_sparse_d_export_csr(context->A_handle, &tempf_indexing,
////          &context->m_A, &A->n, &A->rs, &A->re, &A->cols,
////          (double**)&A->data);
////      }
////      else if ((context->data_type == MPF_COMPF_LEX) ||
////               (context->data_type == MPF_COMPF_LEX_SYMMETRIC))
////      {
////        context->cuda_buffer = mpf_malloc(sizeof(cuDoubleComplex)*A->nz);
////        mpf_sparse_z_export_csr(context->A_handle, &tempf_indexing, &A->m,
////          &context->n_A, &A->rs, &A->re, &A->cols,
////          (MPF_ComplexDouble**)&A->data);
////      }
////
////      /* copies rs array */
////      CudaInt *row_buffer = mkl_malloc(sizeof(CudaInt)*(A->m+1), 32); //@OPTIMIZE: alloc buffers beforehand
////      for (i = 0; i < A->m; i++)
////      {
////        row_buffer[i] = (CudaInt)A->rs[i];
////      }
////      row_buffer[A->m+1] = A->nz;
////      cudaMemcpy(A_gpu->d_row_pointers, row_buffer, sizeof(CudaInt)*(A->m+1),
////        cudaMemcpyHostToDevice);
////      mkl_free(row_buffer);
//
//      /* copies cols array */
////      CudaInt *col_buffer = mkl_malloc(sizeof(CudaInt)*A->nz, 32);  //@OPTIMIZE: alloc buffers beforehand
////      for (i = 0; i < A->nz; ++i)
////      {
////        col_buffer[i] = (CudaInt)A->cols[i];
////      }
////      cudaMemcpy(A_gpu->d_cols, col_buffer, sizeof(CudaInt)*A->nz,
////        cudaMemcpyHostToDevice);
////      mkl_free(col_buffer);
////      if (context->data_type == MPF_REAL)
////      {
////        cudaMemcpy(A_gpu->d_data, A->data, sizeof(double)*A->nz,
////          cudaMemcpyHostToDevice);
////      }
////      else if (context->data_type == MPF_COMPF_LEX)
////      {
////        cudaMemcpy(A_gpu->d_data, A->data, sizeof(cuDoubleComplex)*A->nz,
////          cudaMemcpyHostToDevice);
////        MPF_ComplexDouble *t = mpf_malloc(sizeof(cuDoubleComplex)*A->nz);  //@OPTIMIZE alloce beforehand
////        cudaMemcpy(t, A_gpu->d_data, sizeof(cuDoubleComplex)*A->nz,
////          cudaMemcpyDeviceToHost);
////        mpf_free(t);
////      }
//      //mpf_free(tempf_vec);
////    }
//  }
//  else if (export_target == MPF_A_INPUT) /* @NOTE: input representation in COO format, may differ from MPF_A representation or be the same. */
//  {                                     /* usually MPF_A will use CSR format for A.                                                        */
//    MPF_SparseCsr *A = &context->Ainput.csr; /* input target */
//
//    if (context->data_type == MPF_REAL)
//    {
//      mpf_sparse_d_export_csr(context->Ainput_handle, &tempf_indexing,
//        &context->m_A, &context->n_A, &A->rs, &A->re, &A->cols,
//        (double**)&A->data);
//    }
//    else if (context->data_type == MPF_COMPF_LEX)
//    {
//      mpf_sparse_z_export_csr(context->Ainput_handle, &tempf_indexing,
//        &context->m_A, &context->n_A, &A->rs, &A->re, &A->cols,
//        (MPF_ComplexDouble**)&A->data);
//    }
//  }
//  else if (export_target == MPF_SP_FA)
//  {
//    MPF_SparseCsr *A = &context->fA.csr;  /* @NOTE used for saving, but not for manipulating */
//    if (context->data_type == MPF_REAL)
//    {
//      mpf_sparse_d_export_csr(context->Ainput_handle, &tempf_indexing,
//        &context->m_A, &context->n_A, &A->rs, &A->re, &A->cols,
//        (double**)&A->data);
//    }
//    else if (context->data_type == MPF_COMPF_LEX)
//    {
//      mpf_sparse_z_export_csr(context->Ainput_handle, &tempf_indexing,
//        &context->m_A, &context->n_A, &A->rs, &A->re, &A->cols,
//        (MPF_ComplexDouble**)&A->data);
//    }
//  }
//  else if (export_target == MPF_M)
//  {
//    if (context->solver_interface == MPF_SOLVER_CPU)
//    {
//      /* DO NOT ALLOCATE MEMORY A_csr*/
//      if (context->data_type == MPF_REAL)
//      {
//        MPF_SparseCsr *A = &context->meta_solver.krylov.M.csr;
//        mpf_sparse_d_export_csr(context->A_handle, &tempf_indexing, &A->m,
//          &A->n, &A->rs, &A->re, &(A->cols), (double**)&A->data);
//      }
//      else if (context->data_type == MPF_COMPF_LEX)
//      {
//        MPF_SparseCsr *A = &context->meta_solver.krylov.M.csr;
//        mpf_sparse_z_export_csr(context->A_handle, &tempf_indexing,
//          &context->A.m, &context->A.n, &A->rs, &(A->re),
//          &A->cols, (MPF_ComplexDouble**)&(A->data));
//      }
//      //context->m_A = context->Ainput.coo.m;
//      //context->nz_A = context->Ainput.coo.nz;
//    }
//    //else if (context->solver_interface == MPF_SOLVER_CUDA)
//    //{
//    //  MPF_SparseCsr *A = &context->meta_solver.krylov.M.csr;
//    //  MPF_SparseCsr_Cuda *A_gpu = &context->meta_solver.krylov.M_gpu.csr;
//
//    //  /* @NOTE: DO NOT ALLOCATE MEMORY FOR A_csr (!) */
//    //  /* memory for coo matrix configuration is used */
//    //  if (context->data_type == MPF_REAL)
//    //  {
//    //    context->cuda_buffer = mpf_malloc(sizeof(double)*A->nz);
//    //    mpf_sparse_d_export_csr(context->A_handle, &tempf_indexing,
//    //      &context->m_A, &A->n, &A->rs, &A->re, &A->cols,
//    //      (double**)&A->data);
//    //  }
//    //  else if ((context->data_type == MPF_COMPF_LEX) ||
//    //           (context->data_type == MPF_COMPF_LEX_SYMMETRIC))
//    //  {
//    //    context->cuda_buffer = mpf_malloc(sizeof(cuDoubleComplex)*A->nz);
//    //    mpf_sparse_z_export_csr(context->A_handle, &tempf_indexing, &A->m,
//    //      &context->n_A, &A->rs, &A->re, &A->cols,
//    //      (MPF_ComplexDouble**)&A->data);
//    //  }
//
//    //  /* copies rs array */
//    //  CudaInt *row_buffer = mkl_malloc(sizeof(CudaInt)*(A->m+1), 32); /* @OPTIMIZE: alloc buffers beforehand */
//    //  for (i = 0; i < A->m; i++)
//    //  {
//    //    row_buffer[i] = (CudaInt)A->rs[i];
//    //  }
//    //  row_buffer[A->m+1] = A->nz;
//    //  cudaMemcpy(A_gpu->d_row_pointers, row_buffer, sizeof(CudaInt)*(A->m+1),
//    //    cudaMemcpyHostToDevice);
//    //  mkl_free(row_buffer);
//
//    //  /* copies cols array */
//    //  CudaInt *col_buffer = mkl_malloc(sizeof(CudaInt)*A->nz, 32);  /* @OPTIMIZE: alloc buffers beforehand */
//    //  for (i = 0; i < A->nz; ++i)
//    //  {
//    //    col_buffer[i] = (CudaInt)A->cols[i];
//    //  }
//    //  cudaMemcpy(A_gpu->d_cols, col_buffer, sizeof(CudaInt)*A->nz,
//    //    cudaMemcpyHostToDevice);
//    //  mkl_free(col_buffer);
//    //  if (context->data_type == MPF_REAL)
//    //  {
//    //    cudaMemcpy(A_gpu->d_data, A->data, sizeof(double)*A->nz,
//    //      cudaMemcpyHostToDevice);
//    //  }
//    //  else if (context->data_type == MPF_COMPF_LEX)
//    //  {
//    //    cudaMemcpy(A_gpu->d_data, A->data, sizeof(cuDoubleComplex)*A->nz,
//    //      cudaMemcpyHostToDevice);
//    //    MPF_ComplexDouble *t = mpf_malloc(sizeof(cuDoubleComplex)*A->nz);  /* @OPTIMIZE alloce beforehand */
//    //    cudaMemcpy(t, A_gpu->d_data, sizeof(cuDoubleComplex)*A->nz,
//    //      cudaMemcpyDeviceToHost);
//    //    mpf_free(t);
//    //  }
//    //  //mpf_free(tempf_vec);
//    //}
//  }
//  //else if (export_option == MPF_A_OUTPUT)
//  //{
//  //    MPF_SparseCsr *A_csr = &context->fA.coo;  // used for saving, but not for manipulating
//  //    if (context->data_type == MPF_REAL)
//  //    {
//  //        mpf_matrix_sparse_d_export_csr(context->Ainput_handle, &tempf_indexing, &context->m_A, &context->n_A, &(A_csr->rs),
//  //                               &(A_csr->re), &(A_csr->cols), (double **) &(A_csr->data));
//  //    }
//  //    else if ((context->data_type == MPF_COMPF_LEX) || (context->data_type == MPF_COMPF_LEX_SYMMETRIC))
//  //    {
//  //        mpf_matrix_sparse_z_export_csr(context->Ainput_handle, &tempf_indexing, &context->m_A, &context->n_A, &(A_csr->rs),
//  //                               &(A_csr->re), &(A_csr->cols), (MPF_ComplexDouble **) &(A_csr->data));
//  //    }
//  //}
//}

void mpf_convert_layout_to_sparse
(
  MPF_Layout layout,
  MPF_LayoutSparse *sparse_layout
)
{
  if (layout == MPF_COL_MAJOR)
  {
    *sparse_layout = MPF_SPARSE_COL_MAJOR;
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    *sparse_layout = MPF_SPARSE_ROW_MAJOR;
  }
}

double mpf_sparse_d_dot
(
  MPF_Int m_A,
  MPF_Int m_B,
  MPF_Int *A_ind,
  double *A_data,
  MPF_Int *B_ind,
  double *B_data
)
{
  double c = 0;
  MPF_Int i = 0;
  MPF_Int count = 0;

  while ((i < m_A) && (count < m_B))
  {
    while ((A_ind[i] > B_ind[count]) && (count < m_B))
    {
      count += 1;
    }

    if (A_ind[i] == B_ind[count])
    {
      c += A_data[i]*B_data[count];
      count += 1;
    }

    i += 1;
  }

  return c;
}

void mpf_sparse_d_mm_wrapper
(
  MPF_Solver *solver,
  double *B,
  double *X
)
{
  MPF_Int status = mpf_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0,
    solver->M.handle, solver->M.descr, MPF_SPARSE_COL_MAJOR, (double*)B,
    solver->batch, solver->ld, 0.0, (double*)X, solver->ld);
}

void mpf_sparse_d_csr_alloc
(
  MPF_Sparse *A
)
{
  A->mem.csr.rs = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);
  A->mem.csr.re = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);
  A->mem.csr.cols = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->nz);
  A->mem.csr.data = (MPF_Int*)mpf_malloc(sizeof(double)*A->nz);
}

void mpf_sparse_csr_free
(
  MPF_Sparse *A
)
{
  mpf_free(A->mem.csr.rs);
  mpf_free(A->mem.csr.re);
  mpf_free(A->mem.csr.cols);
  mpf_free(A->mem.csr.data);
}

void mpf_sparse_d_export_csr_mem
(
  MPF_Sparse *A
)
{
  printf("here...\n");
  MPF_Int status = mkl_sparse_d_export_csr(A->handle, &A->index, &A->m, &A->n,
    &A->mem.csr.rs, &A->mem.csr.re, &A->mem.csr.cols,
    (double**)&A->mem.csr.data); 
  mpf_sparse_csr_get_nz(A);
}

void mpf_sparse_z_export_csr_mem
(
  MPF_Sparse *A
)
{
  mkl_sparse_z_export_csr(A->handle, &A->index, &A->m, &A->n,
    &A->mem.csr.rs, &A->mem.csr.re, &A->mem.csr.cols,
    (MPF_ComplexDouble**)&A->mem.csr.data);
  mpf_sparse_csr_get_nz(A);
}

void mpf_sparse_csr_get_nz
(
  MPF_Sparse *A
)
{
  A->nz = 0;
  for (MPF_Int i = 0; i < A->m; ++i)
  {
    A->nz += A->mem.csr.re[i]-A->mem.csr.rs[i];
  }
}

MPF_Int mpf_sparse_csr_get_max_row_nz
(
  MPF_Sparse *A
)
{
  MPF_Int max_nz = 0;
  for (MPF_Int i = 0; i < A->m; ++i)
  {
    if (A->mem.csr.re[i]-A->mem.csr.rs[i] > max_nz)
    {
      max_nz = A->mem.csr.re[i]-A->mem.csr.rs[i];
    }
  }
  return max_nz;
}

void mpf_sparse_copy_meta
(
  MPF_Sparse *A,
  MPF_Sparse *B
)
{
  B->m = A->m;
  B->n = A->n;
  B->descr = A->descr;
  B->nz = A->nz;
  if (B->nz > A->nz_max)
  {
    B->nz = A->nz_max;
  }
  B->data_type = A->data_type;
  B->matrix_type = A->matrix_type;
}

void mpf_sparse_d_copy
(
  MPF_Int start_A,
  MPF_Int end_A,
  MPF_Sparse *A,
  MPF_Int start_B,
  MPF_Int end_B,
  MPF_Sparse *B
)
{
  B->m = A->m;
  B->n = A->n;

  for (MPF_Int ii = start_B; ii < end_B; ++ii)
  {
    /* sets number of nonzeros in current row */
    MPF_Int nz_Ac = A->mem.csr.re[start_A+ii] - A->mem.csr.rs[start_A+ii];

    /* sets rs and re for current row in Pc */
    B->mem.csr.rs[start_B+ii] = 0;
    B->mem.csr.re[start_B+ii] = A->mem.csr.re[start_A+ii] - A->mem.csr.rs[start_A+ii];

    /* copies rows of Ac[0] directly to sparse pattern P */
    MPF_Int index = A->mem.csr.rs[start_A+ii];
    memcpy(&B->mem.csr.cols[start_B], &A->mem.csr.cols[index], sizeof(MPF_Int)*nz_Ac);
    memcpy(&((double*)B->mem.csr.data)[start_B], &((double*)A->mem.csr.data)[index], sizeof(double)*nz_Ac);
  }
}

void mpf_sparse_d_eye
(
  MPF_Sparse *A
)
{
  for (MPF_Int i = 0; i < A->m; ++i)
  {
    A->mem.csr.cols[A->mem.csr.rs[i]] = i;
  }

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    ((double*)A->mem.csr.data)[A->mem.csr.rs[i]] = 1.0;
  }

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    A->mem.csr.re[i] = A->mem.csr.rs[i] + 1;
  }
}

void mpf_convert_csr_sy2ge_triu
(
  MPF_Sparse* A,
  MPF_Sparse* B
)
{
  /*                                                         */
  /* for matrices in uplo (triu) mode, needs a heap for tril */
  /*                                                         */

  MPF_Int* new_nz_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*A->m);

  /* stage-1 (preprocessing )*/

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    new_nz_array[i] = A->mem.csr.re[i] - A->mem.csr.rs[i];
  }

  for (MPF_Int i = 0; i < A->m; ++i)
  {
    for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
    {
      if (A->mem.csr.cols[j] != i)
      {
        new_nz_array[A->mem.csr.cols[j]] += 1;
      }
    }
  }

  B->mem.csr.rs[0] = 0;
  B->mem.csr.re[0] = new_nz_array[0];
  for (MPF_Int i = 1; i < A->m; ++i) 
  {
    B->mem.csr.rs[i] = B->mem.csr.rs[i-1]+new_nz_array[i-1];
  }

  for (MPF_Int i = 1; i < A->m; ++i) 
  {
    B->mem.csr.re[i] = B->mem.csr.re[i-1]+new_nz_array[i-1];
  }

  /* stage-2 (computation) */

  MPF_Int nz = 0;
  if (A->data_type == MPF_REAL)
  {
    for (MPF_Int i = 0; i < A->m; ++i)
    {
      for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
      {
        B->mem.csr.cols[B->mem.csr.rs[i]] = A->mem.csr.cols[A->mem.csr.rs[i]];
        ((double*)B->mem.csr.data)[B->mem.csr.rs[i]] = ((double*)A->mem.csr.data)[j];
        B->mem.csr.rs[i] += 1;
        nz += 1;
        if (A->mem.csr.cols[j] != i)
        {
          B->mem.csr.cols[B->mem.csr.rs[A->mem.csr.cols[j]]] = i;
          ((double*)B->mem.csr.data)[B->mem.csr.rs[A->mem.csr.cols[j]]] = ((double*)A->mem.csr.data)[j];
          B->mem.csr.rs[A->mem.csr.cols[j]] += 1;
          nz += 1;
        }
      }
    }
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    for (MPF_Int i = 0; i < A->m; ++i)
    {
      for (MPF_Int j = A->mem.csr.rs[i]; j < A->mem.csr.re[i]; ++j)
      {
        B->mem.csr.cols[B->mem.csr.rs[i]] = A->mem.csr.cols[j];
        ((MPF_ComplexDouble*)B->mem.csr.data)[B->mem.csr.rs[i]] = ((MPF_ComplexDouble*)A->mem.csr.data)[j];
        B->mem.csr.rs[i] += 1;
        if (A->mem.csr.cols[j] != i)
        {
          B->mem.csr.cols[B->mem.csr.rs[A->mem.csr.cols[j]]] = i;
          B->mem.csr.rs[A->mem.csr.cols[j]] += 1;
        }
      }
    }
  }

  B->mem.csr.rs[0] = 0;
  for (MPF_Int i = 1; i < A->m; ++i)
  {
    B->mem.csr.rs[i] = B->mem.csr.re[i-1];
  }

  // use for debug
  //mpf_sparse_debug_write(B, "test_matrix.mtx");

  mpf_free(new_nz_array);
}

void mpf_convert_coo_sy2ge
(
  MPF_Sparse* A,
  MPF_Sparse* B
)
{
  if (A->data_type == MPF_REAL)
  {
    B->nz = 0;
    for (MPF_Int i = 0; i < A->nz; ++i)
    {
      B->mem.coo.rows[B->nz] = A->mem.coo.rows[i];
      B->mem.coo.cols[B->nz] = A->mem.coo.cols[i];
      ((double*)B->mem.coo.data)[B->nz] = ((double*)A->mem.coo.data)[i];
      B->nz += 1;

      if (A->mem.coo.rows[i] != A->mem.coo.cols[i])
      {
        B->mem.coo.rows[B->nz] = A->mem.coo.cols[i];
        B->mem.coo.cols[B->nz] = A->mem.coo.rows[i];
        ((double*)B->mem.coo.data)[B->nz] = ((double*)A->mem.coo.data)[i];
        B->nz += 1;
      }
    }

    mkl_sparse_d_create_coo(&B->handle, INDEXING, B->m, B->n,
      B->nz, B->mem.coo.rows, B->mem.coo.cols, (double*)B->mem.coo.data);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    B->nz = 0;
    for (MPF_Int i = 0; i < A->nz; ++i)
    {
      B->mem.coo.rows[B->nz] = A->mem.coo.rows[i];
      B->mem.coo.cols[B->nz] = A->mem.coo.cols[i];
      ((MPF_ComplexDouble*)B->mem.coo.data)[B->nz] = ((MPF_ComplexDouble*)A->mem.coo.data)[i];
      B->nz += 1;

      if (A->mem.coo.rows[i] != A->mem.coo.cols[i])
      {
        B->mem.coo.rows[B->nz] = A->mem.coo.cols[i];
        B->mem.coo.cols[B->nz] = A->mem.coo.rows[i];
        ((MPF_ComplexDouble*)B->mem.coo.data)[B->nz] = ((MPF_ComplexDouble*)A->mem.coo.data)[i];
        B->nz += 1;
      }
    }

    mkl_sparse_z_create_coo(&B->handle, INDEXING, B->m, B->n,
      B->nz, B->mem.coo.rows, B->mem.coo.cols, (MPF_ComplexDouble*)B->mem.coo.data);
  }
}

void mpf_sparse_csr_to_coo_convert
(
  MPF_Sparse *P_in,
  MPF_Sparse *P_out
)
{

  P_out->m = P_in->m;
  P_out->nz = 0;

  if (P_in->data_type == MPF_REAL)
  {
    double *data_in = (double*)P_in->mem.csr.data;
    double *data_out = (double*)P_out->mem.coo.data;
    P_out->m = P_in->m;
    P_out->n = P_in->n;
    P_out->nz = 0;
    P_out->nz_max = P_in->nz_max;

    printf("IN POUT\n");
    printf("P_out->m: %d\n", P_out->m);
    for (MPF_Int i = 0; i < P_out->m; ++i)
    {
      //printf("i: %d ==> P_out->nz: %d\n", i, P_out->nz);
      for (MPF_Int j = P_in->mem.csr.rs[i]; j < P_in->mem.csr.re[i]; ++j)
      {
        P_out->mem.coo.rows[P_out->nz] = i;
        P_out->mem.coo.cols[P_out->nz] = P_in->mem.csr.cols[j];
        data_out[P_out->nz] = data_in[j];
        P_out->nz += 1;
      }
    }
  }
  else if (P_in->data_type == MPF_COMPLEX)
  {
    MPF_ComplexDouble *data_in = (MPF_ComplexDouble*)P_in->mem.csr.data;
    MPF_ComplexDouble *data_out = (MPF_ComplexDouble*)P_out->mem.coo.data;

    P_out->m = P_in->m;
    P_out->n = P_in->n;
    P_out->nz = 0;
    P_out->nz_max = P_in->nz_max;

    for (MPF_Int i = 0; i < P_out->m; ++i)
    {
      for (MPF_Int j = P_in->mem.csr.rs[i]; j < P_in->mem.csr.re[i]; ++j)
      {
        //printf("context->fA[0]: %1.2E\n", ((MPF_ComplexDouble*)P_out->mem.csr.data)[0].real);
        /* allocates more memory if required */
        if (P_out->nz == P_out->nz_max)
        {
          P_out->nz_max += P_out->m;
          P_out->mem.coo.cols = (MPF_Int*)mkl_realloc(P_out->mem.coo.cols, sizeof(MPF_Int)*P_out->nz_max);
          P_out->mem.coo.rows = (MPF_Int*)mkl_realloc(P_out->mem.coo.rows, sizeof(MPF_Int)*P_out->nz_max);
        }

        /* updates rows/cols/data and number of nonzeros of P_out */
        P_out->mem.coo.rows[P_out->nz] = i;
        P_out->mem.coo.cols[P_out->nz] = P_in->mem.csr.cols[j];
        data_out[P_out->nz] = data_in[j];
        P_out->nz++;
      }
    }
  }
  else
  {
    return;
  }
  printf("[%li %li]\n", P_in->mem.csr.rs[0], P_in->mem.csr.re[0]);
  printf("P_out->m: %d\n", P_out->m);
  printf("P_out->nz: %d\n", P_out->nz);
}

int mpf_sparse_coo_read
(
  MPF_Sparse *A,
  char *filename_A,
  char *typecode_A
)
{
  FILE *file_handle;
  int ret_code;
  int i;

  if (strcmp(filename_A, "stdin") == 0)
  {
    file_handle = stdin;
  }
  else if ((file_handle = fopen(filename_A, "r")) == NULL)
  {
    return MM_COULD_NOT_READ_FILE;
  }

  //if ((ret_code = mm_read_banner(file_handle, (MM_typecode*)typecode_A)) != 0)
  if ((ret_code = mm_read_banner(file_handle, (MM_typecode*)typecode_A)) != 0)
  {
    return ret_code;
  }

  if (!(mm_is_valid(typecode_A)  &&
        mm_is_sparse(typecode_A) &&
        mm_is_matrix(typecode_A)))
  {
    return MM_UNSUPPORTED_TYPE;
  }

  if ((ret_code = mm_read_mtx_crd_size(file_handle, &A->m, &A->n, &A->nz)) != 0)
  {
    return ret_code;
  }

  if (mm_is_symmetric(typecode_A))
  {
    A->descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    A->matrix_type = MPF_MATRIX_SYMMETRIC;
  }
  else if (mm_is_hermitian(typecode_A))
  {
    A->descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    A->matrix_type = MPF_MATRIX_HERMITIAN;
  }
  else
  {
    //@TODO: find correct type
    A->descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    A->matrix_type = MPF_MATRIX_GENERAL;
  }

  if (mm_is_real(typecode_A))
  {
    for (i = 0; i < A->nz; ++i)
    {
      if (fscanf(file_handle, "%d %d %lf ", &A->mem.coo.rows[i], &A->mem.coo.cols[i],
        &((double*)A->mem.coo.data)[i]) != 3)
      {
        return MM_PREMATURE_EOF;
      }

      /* converts to 0-based indexing */
      --A->mem.coo.rows[i];
      --A->mem.coo.cols[i];
    }
    MPF_Int status = 0;
    #if DEBUG == 1
      printf("m_A: %d, n_A: %d, nz: %d\n", A.m, A.n, A.nz);
    #endif

    status = mkl_sparse_d_create_coo(&A->handle, INDEXING, A->m, A->n,
      A->nz, A->mem.coo.rows, A->mem.coo.cols, (double*)A->mem.coo.data);
  }
  else if (mm_is_complex(typecode_A))
  {
    for (MPF_Int i = 0; i < A->nz; ++i)
    {
      if (fscanf(file_handle, "%d %d %lf %lf ", &A->mem.coo.rows[i], &A->mem.coo.cols[i],
        &((double*)A->mem.coo.data)[2*i], &((double*)A->mem.coo.data)[2*i+1]) != 4)
      {
          return MM_PREMATURE_EOF;
      }

      /* converts to 0-based indexing */
      A->mem.coo.rows[i]--;
      A->mem.coo.cols[i]--;
    }
    mkl_sparse_z_create_coo(&A->handle, INDEXING, A->m, A->n,
      A->nz, A->mem.coo.rows, A->mem.coo.cols, (MPF_ComplexDouble*)A->mem.coo.data);
  }
  else
  {
    return MM_UNSUPPORTED_TYPE;
  }

  if (file_handle != stdin)
  {
    fclose(file_handle);
  }

  return 0;
}

int mpf_sparse_size_read_ext
(
  FILE *file_handle,
  MPF_Int *m,
  MPF_Int *n,
  MPF_Int *nz
)
{
  mm_read_mtx_crd_size(file_handle, m, n, nz);
  return 0;
}

int mpf_sparse_size_read
(
  MPF_Sparse *A,
  char *filename_A
)
{
  FILE *file_handle = NULL;
  if ((file_handle = fopen(filename_A, "r")) == NULL)
  {
    return -1;
  }
  mm_read_mtx_crd_size(file_handle, &A->m, &A->n, &A->nz);

  fclose(file_handle);
  return 0;
}

int mpf_sparse_meta_read
(
  MPF_Sparse *A,
  char *filename_A,
  MM_typecode* typecode_A
)
{
  FILE *file_handle = NULL;
  if ((file_handle = fopen(filename_A, "r")) == NULL)
  {
    return -1;
  }
  mm_read_banner(file_handle, typecode_A);
  mm_read_mtx_crd_size(file_handle, &A->m, &A->n, &A->nz);

  fclose(file_handle);
  return 0;
}

void mpf_sparse_coo_write
(
  MPF_Sparse *A,
  char *filename,
  MM_typecode matrix_code
)
{
  /* write sparse coo matrix */
  printf("testing\n");
  printf("A->m: %li, A->n: %li, A->nz: %li\n", A->m, A->n, A->nz);
  mm_write_mtx_crd(filename, A->m, A->n, A->nz, A->mem.coo.rows, A->mem.coo.cols,
    A->mem.coo.data, matrix_code);
}

void mpf_sparse_coo_z_print
(
  const MPF_Sparse *storage_A
)
{
  MPF_ComplexDouble *data = (MPF_ComplexDouble *) storage_A->mem.coo.data;
  for (MPF_Int i = 0; i < storage_A->nz; ++i)
  {
    if (data[i].imag >= 0)
    {
      printf ("(%d, %d) %1.4E+%1.4Ei \n", storage_A->mem.coo.rows[i],
         storage_A->mem.coo.cols[i], data[i].real, data[i].imag);
    }
    else
    {
      printf ("(%d, %d) %1.4E-%1.4Ei \n", storage_A->mem.coo.rows[i],
        storage_A->mem.csr.cols[i], data[i].real, -data[i].imag);
    }
  }
}

void mpf_sparse_debug_write
(
  MPF_Sparse* A,
  char filename_A[100]
)
{
  MPF_Sparse Acoo;
  printf("Ac.m: %d, Ac.n: %d, Ac.nz: %d\n", A->m, A->n, A->nz);
  Acoo.m = A->m;
  Acoo.n = A->n;
  Acoo.nz = A->nz;
  Acoo.data_type = A->data_type;
  Acoo.matrix_type = A->matrix_type;
  Acoo.descr = A->descr;

printf("alloc\n");
  mpf_sparse_coo_alloc(&Acoo);

printf("before convert\n");
  mpf_sparse_csr_to_coo_convert(A, &Acoo);

printf("after convert\n");

  MM_typecode typecode_A;
  mm_initialize_typecode(&typecode_A);
  mm_set_real(&typecode_A);
  mm_set_matrix(&typecode_A);
  mm_set_coordinate(&typecode_A);
  mm_set_general(&typecode_A);
printf("before coo_write\n");
  mpf_sparse_coo_write(&Acoo, filename_A, typecode_A);

  // should deallocate memory here
}



void mpf_sparse_read
(
  char filename[],
  MPF_Sparse* A
)
{
  MPF_Sparse Acoo;
  MM_typecode matcode;
  mpf_sparse_meta_read(&Acoo, filename, &matcode);

  A->m = Acoo.m;
  A->n = Acoo.n;
  A->nz = Acoo.nz;
  A->export_mem_function = mpf_sparse_export_csr;

  if (mm_is_real(matcode))
  {
    A->data_type = MPF_REAL;
    Acoo.data_type = MPF_REAL;
  }
  else if (mm_is_complex(matcode))
  {
    A->data_type = MPF_REAL;
    Acoo.data_type = MPF_COMPLEX;
  }

  /* read input matrix in coo */
  mpf_sparse_coo_alloc(&Acoo);
  mpf_sparse_coo_read(&Acoo, filename, matcode);
  mpf_sparse_coo_to_csr_convert(&Acoo, A);

  if (A->data_type == MPF_REAL)
  {
    mkl_sparse_d_create_csr(&A->handle, INDEXING, A->m, A->n, A->mem.csr.rs,
      A->mem.csr.re, A->mem.csr.cols, (double*)A->mem.csr.data);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    mkl_sparse_z_create_csr(&A->handle, INDEXING, A->m, A->n, A->mem.csr.rs,
      A->mem.csr.re, A->mem.csr.cols, (MPF_ComplexDouble*)A->mem.csr.data);
  }

}
