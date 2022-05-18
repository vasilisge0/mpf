
void mp_cuda_matrix_d_announce
(
  double *A,
  const CudaInt m_A,
  const CudaInt n_A,
  const CudaInt ld_A,
  char filename[100]
)
{
  printf("\n%s:\n", filename);
  mp_cuda_matrix_d_print(A, m_A, n_A, ld_A);
  printf("\n");
}


void mp_cuda_matrix_z_announce
(
  cuDoubleComplex *A,
  const CudaInt m_A,
  const CudaInt n_A,
  const CudaInt ld_A,
  char filename[100]
)
{
    printf("\n%s:\n", filename);
    mp_cuda_matrix_z_print(A, m_A, n_A, ld_A);
    printf("\n");
}


void mp_matrix_cuda_print
(
  double *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
)
{
  double *A = mp_malloc((sizeof *A)*m_A*n_A);
  cudaMemcpy(A, Acuda, (sizeof *A)*m_A*n_A, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m_A; ++i)
  {
    for (int j = 0; j < n_A; ++j)
    {
      printf ("%1.2E ", A[j*ld_A + i]);
    }
    printf ("\n");
  }
  mp_free(A);
}

void mp_cuda_matrix_d_print
(
  double *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
)
{
  double *A = mp_malloc((sizeof *A)*m_A*n_A);
  cudaMemcpy(A, Acuda, (sizeof *A)*m_A*n_A, cudaMemcpyDeviceToHost);
  mp_matrix_d_print(A, m_A, n_A, ld_A);
  mp_free(A);
}

void mp_cuda_matrix_z_print
(
  cuDoubleComplex *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
)
{
  MPComplexDouble *A = mp_malloc((sizeof *A)*ld_A*n_A);
  cudaMemcpy(A, Acuda, (sizeof *A)*ld_A*n_A, cudaMemcpyDeviceToHost);
  mp_matrix_z_print(A, m_A, n_A, ld_A);
  mp_free(A);
}

void mp_complex_matrix_print
(
  MPComplexDouble *A,
  MPInt m_A,
  MPInt n_A,
  MPInt ld_A
)
{
  for (uint i = 0; i < m_A; ++i)
  {
    for (uint j = 0; j < n_A; ++j)
    {
      if (A[ld_A*j + i].imag >= 0)
      {
          printf ("%1.2E+%1.2Ei  ", A[ld_A*j + i].real, A[ld_A*j + i].imag);
      }
      else
      {
          printf ("%1.2E - %1.2Ei  ", A[ld_A*j + i].real, -A[ld_A*j + i].imag);
      }
    }
    printf ("\n");
  }
}



/* -----------------  cuda solver initialization functions ------------------ */



void mp_cuda_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
)
{
  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_CG;
    context->solver_inner_function = &mp_cuda_dsy_cg;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_CG;
    context->solver_inner_function = &mp_cuda_zsy_cg;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_CG;
    context->solver_inner_function = &mp_cuda_zhe_cg;
  }

  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = 0;
  context->solver_interface = MP_SOLVER_CUDA;

  mp_cuda_cg_memory_get(
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner);
}

void mp_cuda_global_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
)
{
  if
  (
  context->data_type == MP_REAL
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_GLOBAL_CG;
    context->solver_inner_function = &mp_cuda_dsy_global_cg;
    context->solver_interface = MP_SOLVER_CUDA;

    /* krylov parameters */
    context->meta_solver.krylov.iterations = iterations;
    context->meta_solver.krylov.restarts = 0;
    context->meta_solver.krylov.tolerance = tolerance;
  }
  else if ((context->data_type == MP_COMPLEX) &&
           (context->A_type == MP_MATRIX_SYMMETRIC))
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_GLOBAL_CG;
    context->solver_inner_function = &mp_cuda_zsy_global_cg;
    context->solver_interface = MP_SOLVER_CUDA;

    /* krylov parameters */
    context->meta_solver.krylov.iterations = iterations;
    context->meta_solver.krylov.restarts = 0;
    context->meta_solver.krylov.tolerance = tolerance;
  }
  else if ((context->data_type == MP_COMPLEX) &&
           (context->A_type == MP_MATRIX_HERMITIAN))
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_GLOBAL_CG;
    context->solver_inner_function = &mp_cuda_zhe_global_cg;
    context->solver_interface = MP_SOLVER_CUDA;

    /* krylov parameters */
    context->meta_solver.krylov.tolerance = tolerance;
    context->meta_solver.krylov.iterations = iterations;
    context->meta_solver.krylov.restarts = 0;
  }

  /* get the number of bytes for inner memory (cpu/cuda) */
  mp_cuda_global_cg_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}

void mp_cuda_block_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;
  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_BLOCK_LANCZOS;
    context->solver_inner_function = &mp_cuda_dsy_block_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_BLOCK_LANCZOS;
    context->solver_inner_function = &mp_cuda_zsy_block_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_BLOCK_LANCZOS;
    //context->solver_inner_function = &mp_cuda_zhe_block_lanczos;
    mp_cuda_block_lanczos_memory_get(
      context->data_type,
      context->A_struct_type,
      context->meta_solver.krylov,
      context->m_A,
      &context->bytes_inner,
      &context->bytes_cuda_inner);
  }
}

void mp_cuda_global_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;
  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_GLOBAL_LANCZOS;
    context->solver_inner_function = &mp_cuda_dsy_global_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_GLOBAL_LANCZOS;
    context->solver_inner_function = &mp_cuda_zsy_global_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_GLOBAL_LANCZOS;
    context->solver_inner_function = &mp_cuda_zhe_global_lanczos;
  }
  mp_cuda_global_lanczos_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}

void mp_cuda_batch_init
(
MPContext *context
)
{
  context->solver_outer_type = MP_CUDA_BATCH;
  //context->solver_outer_function = &mp_cuda_batch_external_2;
}

void mp_cuda_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;

  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_LANCZOS;
    context->solver_inner_function = &mp_cuda_dsy_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_LANCZOS;
    context->solver_inner_function = &mp_cuda_zsy_lanczos;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_LANCZOS;
    context->solver_inner_function = &mp_cuda_zhe_lanczos;
  }

  mp_cuda_lanczos_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}

void mp_cuda_global_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;

  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_GLOBAL_GMRES;
    context->solver_inner_function = &mp_cuda_dge_global_gmres;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_struct_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_GLOBAL_GMRES;
    context->solver_inner_function = &mp_cuda_zsy_global_gmres_2;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_struct_type == MP_MATRIX_HERMITIAN)
  )

  mp_cuda_global_lanczos_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}


void mp_cuda_block_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;

  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DGE_BLOCK_GMRES;
    context->solver_inner_function = &mp_cuda_dge_block_gmres;
  }
  else if ((context->data_type == MP_COMPLEX) &&
           (context->A_type == MP_MATRIX_SYMMETRIC))
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_BLOCK_GMRES;
    //context->solver_inner_function = &mp_cuda_zsy_block_gmres;
  }
  else if ((context->data_type == MP_COMPLEX) &&
           (context->A_type == MP_MATRIX_HERMITIAN))
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_BLOCK_GMRES;
    context->solver_inner_function = &mp_cuda_zhe_block_gmres;
  }

  mp_cuda_block_gmres_memory_get(
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner);
}

void mp_cuda_block_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
)
{
  context->solver_inner_type = MP_SOLVER_CUDA_DSY_BLOCK_CG;
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->solver_interface = MP_SOLVER_CUDA;
  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DSY_BLOCK_CG;
    context->solver_inner_function = &mp_cuda_dsy_block_cg;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_BLOCK_CG;
    context->solver_inner_function = &mp_cuda_zsy_block_cg;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZHE_BLOCK_CG;
    context->solver_inner_function = &mp_cuda_zhe_block_cg;
  }

  mp_cuda_block_cg_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->meta_solver.krylov,
    context->m_A,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}

void mp_cuda_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
)
{
  context->meta_solver.krylov.tolerance = tolerance;
  context->meta_solver.krylov.iterations = iterations;
  context->meta_solver.krylov.restarts = restarts;
  context->solver_interface = MP_SOLVER_CUDA;

  if (context->data_type == MP_REAL)
  {
    context->solver_inner_type = MP_SOLVER_CUDA_DGE_GMRES;
    context->solver_inner_function = &mp_cuda_dge_gmres;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_SYMMETRIC)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZSY_GMRES;
    context->solver_inner_function = &mp_cuda_zsy_gmres;
  }
  else if
  (
    (context->data_type == MP_COMPLEX) &&
    (context->A_type == MP_MATRIX_HERMITIAN)
  )
  {
    context->solver_inner_type = MP_SOLVER_CUDA_ZGE_GMRES;
    context->solver_inner_function = &mp_cuda_zge_gmres;
  }

  mp_cuda_gmres_memory_get
  (
    context->data_type,
    context->A_struct_type,
    context->m_A,
    context->meta_solver.krylov,
    &context->bytes_inner,
    &context->bytes_cuda_inner
  );
}

void mp_context_gpu_cuda_finish(MPContext* context)
{
  //if (((MPContextGpuCuda *) context->context_gpu)->cublas_handle != NULL)
  //{
  //    cublasDestroy(((MPContextGpuCuda *) context->context_gpu)->cublas_handle);
  //    ((MPContextGpuCuda *) context->context_gpu)->cublas_handle = NULL;
  //}
  //if (((MPContextGpuCuda *) context->context_gpu)->cusparse_handle != NULL)
  //{
  //    cusparseDestroy(((MPContextGpuCuda *) context->context_gpu)->cusparse_handle);
  //    ((MPContextGpuCuda *) context->context_gpu)->cusparse_handle = NULL;
  //}
  //mp_free(context->context_gpu);
  //context->context_gpu = NULL;
  if (context->context_gpu.cublas_handle != NULL)
  {
      cublasDestroy(context->context_gpu.cublas_handle);
      //context->context_gpu.cublas_handle = NULL;
  }
  if (context->context_gpu.cusparse_handle != NULL)
  {
//cusparseDestroy(context->context_gpu.cusparse_handle);
      //context->context_gpu.cusparse_handle = NULL;
      //cusparseDestroy(((MPContextGpuCuda *) context->context_gpu)->cusparse_handle);
      //((MPContextGpuCuda *) context->context_gpu)->cusparse_handle = NULL;
  }
  //mp_free(context->context_gpu);
  //context->context_gpu = NULL;
}

void mp_context_cuda_init(MPContext* context)
{
  //context->context_gpu = mp_malloc(sizeof(MPContextGpuCuda));
  //cublasCreate(&(((MPContextGpuCuda *) context->context_gpu)->cublas_handle));
  //cusparseCreate(&(((MPContextGpuCuda *) context->context_gpu)->cusparse_handle));

  //MPContextGpuCuda *handle =context->context_gpu;
  //handle = mp_malloc(sizeof(MPContextGpuCuda));
  //cublasCreate(&handle->cublas_handle);
  //cusparseCreate(&handle->cusparse_handle);
  //printf("t: %d", t);

  //MPContextGpuCuda *handle = NULL;
  //context->context_gpu = mp_malloc(sizeof(MPContextGpuCuda));
  //handle = (MPContextGpuCuda *) context->context_gpu;
  //int s;
  //cublasHandle_t test_h;
//s = cublasCreate(&test_h);
//cublasDestroy(test_h);
  cublasCreate(&context->context_gpu.cublas_handle);
  //printf("blas: %d\n", s);
  cusparseCreate(&context->context_gpu.cusparse_handle);
}

/*** cusparse sparse matrix manipulation functions ***/

void mp_cuda_sparse_descr_create
(
  //MPContext *context,
  MPDataType data_type,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt nz_A,
  cusparseSpMatDescr_t descr_A,
  MPSparseCsr_Cuda *A
)
{
  if (data_type == MP_REAL)
  {
    cusparseCreateCsr(&descr_A, m_A, n_A, nz_A, A->d_row_pointers, A->d_cols,
      A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  }
  else if (data_type == MP_COMPLEX)
  {
    cusparseCreateCsr(&descr_A, m_A, n_A, nz_A, A->d_row_pointers, A->d_cols,
      A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
  }
}

/*================================================================*/
/* == complex scalar manipulation functions for cuda functions == */
/*================================================================*/

cuDoubleComplex mp_cuda_scalar_z_set
(
  const double variable_real
)
{
  cuDoubleComplex variable_complex;
  variable_complex.x = variable_real;
  variable_complex.y = 0.0;
  return variable_complex;
}

cuDoubleComplex mp_cuda_scalar_z_init
(
  const double real_value,
  const double imag_value
)
{
  cuDoubleComplex gamma;
  gamma.x = real_value;
  gamma.y = imag_value;
  return gamma;
}

cuDoubleComplex mp_cuda_scalar_z_add
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
)
{
  cuDoubleComplex gamma;
  gamma.x = alpha.x + beta.x;
  gamma.y = alpha.y + beta.y;
  return gamma;
}

cuDoubleComplex mp_cuda_scalar_z_divide
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
)
{
  cuDoubleComplex gamma;
  gamma.x = (alpha.x*beta.x + alpha.y*beta.y) / (beta.x*beta.x + beta.y*beta.y);
  gamma.y = (alpha.y*beta.x - alpha.x*beta.y) / (beta.x*beta.x + beta.y*beta.y);
  return gamma;
}

cuDoubleComplex mp_cuda_scalar_z_multiply
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
)
{
  cuDoubleComplex gamma;
  gamma.x = alpha.x*beta.x - alpha.y*beta.y;
  gamma.y = alpha.x*beta.y + alpha.y*beta.x;
  return gamma;
}

cuDoubleComplex mp_cuda_scalar_z_normalize
(
  cuDoubleComplex alpha,
  const double beta
)
{
  alpha.x = alpha.x / beta;
  alpha.y = alpha.y / beta;
  return alpha;
}

cuDoubleComplex mp_cuda_scalar_z_subtract
(
  cuDoubleComplex alpha,
  const cuDoubleComplex beta
)
{
  alpha.x = alpha.x - beta.x;
  alpha.y = alpha.y - beta.y;
  return alpha;
}

cuDoubleComplex mp_cuda_scalar_z_invert_sign
(
  cuDoubleComplex alpha
)
{
  alpha.x = -alpha.x;
  alpha.y = -alpha.y;
  return alpha;
}

cuComplex mp_cuda_scalar_c_init
(
  const float real_value,
  const float imag_value
)
{
  cuComplex gamma;
  gamma.x = real_value;
  gamma.y = imag_value;
  return gamma;
}

cuComplex mp_cuda_scalar_c_add
(
  const cuComplex alpha,
  const cuComplex beta
)
{
  cuComplex gamma;
  gamma.x = alpha.x + beta.x;
  gamma.y = alpha.y + beta.y;
  return gamma;
}

cuComplex mp_cuda_scalar_c_divide
(
  const cuComplex alpha,
  const cuComplex beta
)
{
  cuComplex gamma;
  gamma.x = (alpha.x*beta.x + alpha.y*beta.y) / (beta.x*beta.x + beta.y*beta.y);
  gamma.y = (alpha.y*beta.x - alpha.x*beta.y) / (beta.x*beta.x + beta.y*beta.y);
  return gamma;
}

cuComplex mp_cuda_scalar_c_multiply
(
  const cuComplex alpha,
  const cuComplex beta
)
{
  cuComplex gamma;
  gamma.x = alpha.x*beta.x - alpha.y*beta.y;
  gamma.y = alpha.x*beta.y + alpha.y*beta.x;
  return gamma;
}

cuComplex mp_cuda_scalar_c_normalize
(
  cuComplex alpha,
  const float beta
)
{
  alpha.x = alpha.x / beta;
  alpha.y = alpha.y / beta;
  return alpha;
}

cuComplex mp_cuda_scalar_c_subtract
(
  cuComplex alpha,
  const cuComplex beta
)
{
  alpha.x = alpha.x - beta.x;
  alpha.y = alpha.y - beta.y;
  return alpha;
}

cuComplex mp_cuda_scalar_c_invert_sign
(
  cuComplex alpha
)
{
  alpha.x = -alpha.x;
  alpha.y = -alpha.y;
  return alpha;
}

cuDoubleComplex mp_scalar_cuda_z_init
(
  const double real_value,
  const double imag_value
)
{
  cuDoubleComplex gamma;
  ((double *) &gamma)[0] = 1.0;
  ((double *) &gamma)[1] = 0.0;
  //gamma.real = real_value;
  //gamma.imag = imag_value;
  return gamma;
}
