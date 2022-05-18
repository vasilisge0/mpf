#include "mp.h"
#include "mp_cuda.h"
#include "mp_cuda_auxilliary.h"

void mp_cuda_zsy_global_gmres
(
  /* solver parameters */
  const KrylovMeta meta,

  /* data */
  const MPSparseDescr A_descr,
  const MPSparseHandle A_handle,
  const MPInt n,
  MPComplexDouble *B,
  MPComplexDouble *X,
  MPComplexDouble *memory,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
  MPComplexDouble MINUS_ONE_C = mp_scalar_z_init(-1.0, 0.0);
  /* solver context */
  MPInt k = 0;
  MPInt j = 0;
  MPInt i = 0;
  MPInt t = 0;
  MPComplexDouble B_norm = ZERO_C;
  MPComplexDouble R_norm = ZERO_C;
  double R_norm_abs = 0.0;
  MPComplexDouble temp_complex = ZERO_C;
  MPComplexDouble h_temp = ZERO_C;
  double h_temp_abs = 0.0;
  MPComplexDouble trace = ZERO_C;
  /* solver meta */
  MPInt blk = meta.blk;
  MPInt inner_iterations = meta.iterations;
  MPInt outer_iterations = 1+meta.restarts;
  MPInt m_H = (meta.iterations+1);
  MPInt n_H = meta.iterations;
  const MPInt ld_H = meta.iterations+1;
  MPInt size_V = n*blk*m_H;
  MPInt size_H = m_H*n_H;
  MPInt size_Br = m_H;
  MPInt size_Hblk = blk*blk;
  /* cpu memory */
  MPComplexDouble *V = memory;
  MPComplexDouble *H = &V[size_V];
  MPComplexDouble *Br = &H[size_H];
  MPComplexDouble *Hblk = &Br[size_Br];
  MPComplexDouble *temp_matrix = &Hblk[size_Hblk];
  /* handles on assigned memory */
  MPComplexDouble *W = NULL;
  MPComplexDouble *Vprev = NULL;
  MPComplexDouble *const Vfirst = V;
  MPComplexDouble *const Vlast = &V[(n*blk)*meta.iterations];
  MPComplexDouble *R = Vlast;

  /* computes residual vector-blocks using initial approximation of solution
     block vectors */
  memcpy(V, B, (sizeof *V)*n*blk);
  mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A_handle,
    A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, V, n);
  mp_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
    B, n*blk, B, n*blk, &ZERO_C, &B_norm, 1);
  mp_vectorized_z_sqrt(1, &B_norm, &B_norm);

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    /* first iteration */
    mp_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      V, n*blk, V, n*blk, &ZERO_C, &R_norm, 1);
    mp_vectorized_z_sqrt(1, &R_norm, &R_norm);
    R_norm = mp_scalar_z_divide(R_norm, B_norm);
    mp_vectorized_z_abs(1, &R_norm, &R_norm_abs);
    #if DEBUG == 1
      printf("relative residual frobenious norm: %lf\n", R_norm_abs);
    #endif
    if (R_norm_abs <= meta.tolerance)  /* checks terminating condition */
    {
      return;
    }
    temp_complex = mp_scalar_z_divide(ONE_C, R_norm);
    mp_zscal(n*blk, &temp_complex, V, 1);
    Br[0] = B_norm;
    for (i = 1; i < m_H; ++i)
    {
      Br[i] = ZERO_C;
    }

    /* inner loop */
    for (j = 0; j < inner_iterations; ++j)
    {
      W = &V[(n*blk)*(j+1)];
      Vprev = &V[(n*blk)*j];
      mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A_handle,
        A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, Vprev, blk, n, ZERO_C,
        W, n);
      for (i = 0; i < j + 1; ++i)
      {
        Vprev = &Vfirst[(n*blk)*i];
        mp_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, blk, blk, n,
          &ONE_C, W, n, Vprev, n, &ZERO_C, Hblk, blk);
        trace = ZERO_C;
        for (t = 0; t < blk; ++t)
        {
          trace = mp_scalar_z_add(trace, Hblk[blk*t+t]);
        }
        H[ld_H*j+i] = trace;
        temp_complex = mp_scalar_z_invert_sign(trace);
        mp_zaxpy(n*blk, &temp_complex, Vprev, 1, W, 1);
      }
      mp_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk,
        &ONE_C, W, n*blk, W, n*blk, &ZERO_C, &h_temp, 1);
      mp_vectorized_z_sqrt(1, &h_temp, &h_temp);
      mp_vectorized_z_abs(1, &h_temp, &h_temp_abs);
      if (h_temp_abs <= 1e-12)
      {
        inner_iterations = j+1;
        m_H = (j+1);
        n_H = j;
        break;
      }
      temp_complex = mp_scalar_z_divide(ONE_C, h_temp);
      mp_zscal(n*blk, &temp_complex, W, 1);
      H[ld_H*j+j+1] = h_temp;
    }

    /* solves system of equations using qr decomposition */
    mp_qr_zge_givens(H, Br, m_H, n_H, blk, temp_matrix);
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, Br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      W = &V[n*blk*i];
      mp_zaxpy(n*blk, &Br[i], W, 1, X, 1);
    }
    memcpy(R, B, (sizeof *B)*n*blk);
    mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A_handle,
      A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, X, blk, n, ONE_C, R, n);
    mp_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, n*blk, &ONE_C,
      R, n*blk, R, n*blk, &ZERO_C, &R_norm, 1);
    mp_vectorized_z_sqrt(1, &R_norm, &R_norm);
    R_norm = mp_scalar_z_divide(R_norm, B_norm);
    mp_vectorized_z_abs(1, &R_norm, &R_norm_abs);
    #if DEBUG == 1
      printf("norm_frobenious_residual: %1.4E\n", R_norm_abs);
    #endif
    if ((R_norm_abs <= meta.tolerance) || (j == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      memcpy(V, R, (sizeof *R)*n*blk);
      inner_iterations = meta.iterations;
      m_H = meta.iterations+1;
      n_H = meta.iterations;
    }
  }
  Vprev = NULL;
  W = NULL;
  R = NULL;
}

void mp_cuda_dge_gmres
(
  /* solver parameterrs */
  MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* input/output data */
  MPSparseCsr_Cuda *A,
  const MPInt n,
  double *d_b,
  double *d_x,
  double *memory_cpu,
  double *memory_gpu,

  /* output metadata */
  MPSolverInfo *info
)
{
  /* contstants */
  const double ONE_REAL = 1.0;
  const double MINUS_ONE_REAL = -1.0;
  const double ZERO_REAL = 0.0;
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  MPInt k = 0;
  double b_norm = 0;
  double r_norm = 0;
  double temp_real = 1.0;
  /* solver metadata */
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = 1+(CudaInt)meta.restarts;
  CudaInt ld_H = (CudaInt)meta.iterations+1;
  CudaInt m_H = (CudaInt)meta.iterations+1;
  CudaInt n_H = (CudaInt)meta.iterations;
  /* cpu memory */
  double *H = memory_cpu;
  double *h_br = &H[m_H*n_H];
  double *temp_matrix = &h_br[m_H];
  /* gpu memory */
  double *d_V  = memory_gpu;
  double *d_br = &d_V[n*((CudaInt)meta.iterations+1)];
  /* handles on gpu memory */
  double *d_vfirst = d_V;
  double *d_vlast = &d_V[n*(CudaInt)meta.iterations];
  double *d_vprev = d_vfirst;
  double *d_w = &d_vfirst[n];
  double *d_r = d_vlast;
  /* start descriptors for cusparse */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_vprev;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  /* initialize descriptors and memory */
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  mp_zeros_d_set(MP_COL_MAJOR, m_H, 1, h_br, ld_H);

  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers,
    A->d_cols, A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&descr_b, n, d_b, CUDA_R_64F);
  cusparseCreateDnVec(&descr_x, n, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&descr_vprev, n, d_vprev, CUDA_R_64F);
  cusparseCreateDnVec(&descr_w, n, d_w, CUDA_R_64F);
  cusparseCreateDnVec(&descr_r, n, d_r, CUDA_R_64F);

  /* first iteration */
  cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL,
    descr_r, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
  cudaDeviceSynchronize();
  cublasDnrm2(context_gpu->cublas_handle, n, d_b, 1, &b_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*n, cudaMemcpyDeviceToDevice);
  cublasDnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
  
  /* outer-loop (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_d_set(MP_COL_MAJOR, ld_H, 1, h_br,  ld_H);
    h_br[0] = r_norm;
    temp_real = 1/r_norm;
    cublasDscal(context_gpu->cublas_handle, n, &temp_real, d_V, 1);
    /* inner iterations */
    for (j = 0; j < inner_iterations; j++)
    {
      d_w = &d_V[n*(j+1)];
      d_vprev = &d_V[n*j];
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_vprev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, &ONE_REAL, descr_A,
        descr_vprev, &ZERO_REAL, descr_w, CUDA_R_64F,
        CUSPARSE_CSRMV_ALG1, NULL);

      for (i = 0; i < j+1; ++i)
      {
        d_vprev = &d_V[n*i];
        cusparseDnVecSetValues(descr_vprev, d_vprev);
        cublasDdot(context_gpu->cublas_handle, n, d_w, 1, d_vprev, 1,
          &H[m_H*j+i]);
        temp_real = -H[ld_H*j+i];
        cublasDaxpy(context_gpu->cublas_handle, n, &temp_real, d_vprev, 1,
          d_w, 1);  /* W <-- W - V(:, J)*H(j, j)) */
      }
      cublasDnrm2(context_gpu->cublas_handle, n, d_w, 1, &H[ld_H*j+j+1]);
      if (fabs(H[m_H*j+j+1]) <= 1e-12)
      {
          inner_iterations = j; //@BUG: j+1?
          m_H = inner_iterations;
          n_H = inner_iterations-1;
          break;
      }
      temp_real = 1/H[ld_H*j+j+1];
      cublasDscal(context_gpu->cublas_handle, n, &temp_real, d_w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    mp_qr_givens_dge(n_H, 1, H, m_H, h_br, m_H, temp_matrix);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, h_br, ld_H);
    cudaMemcpy(d_br, h_br, (sizeof *d_br)*m_H, cudaMemcpyHostToDevice);
    cublasDgemv(context_gpu->cublas_handle, CUBLAS_OP_N, n,
      n_H, &ONE_REAL, d_V, n, d_br, 1, &ONE_REAL, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL, descr_r,
      CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cublasDnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("residual: %1.4E\n", r_norm/b_norm);
    #endif
    #if MODE

    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = meta.iterations;
      m_H = meta.iterations+1;
      n_H = meta.iterations;
    }
  }
  d_V = NULL;
  d_br = NULL;
  d_vfirst = NULL;
  d_vlast = NULL;
  d_vprev = NULL;
  d_w = NULL;
  d_r = NULL;
}

void mp_cuda_dge_gmres_2
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  MPSparseCsr_Cuda *A,
  const MPInt n,
  const double *d_b,
  double *d_x,
  double *memory_cpu,
  double *memory_gpu,

  /* solver collected meta */
  MPSolverInfo *info
)
{
  /* constants */
  const double ONE_REAL = 1.0;
  const double MINUS_ONE_REAL = -1.0;
  const double ZERO_REAL = 0.0;
  /* solver context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  double b_norm = 0;
  double r_norm = 0;
  /* mathematical object dimensions */
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = (CudaInt)meta.restarts+1;
  CudaInt ld_m_H = (CudaInt)meta.iterations+1;
  CudaInt m_H = ld_m_H;
  CudaInt n_H = (CudaInt)meta.iterations;
  CudaInt ld_H = m_H;
  /* - cpu memory */
  double *H = memory_cpu;
  double *h_br = &H[m_H*n_H];
  double *temp_matrix = &h_br[m_H];
  /* - gpu memory */
  double *d_V = memory_gpu;
  double *d_br = &d_V[n*((CudaInt)meta.iterations+1)];
  /* - map handles to already accessed gpu memory */
  double *d_vfirst = d_V;
  double *d_vlast = &d_V[n*(CudaInt)meta.iterations];
  double *d_vprev = d_vfirst;
  double *d_w = &d_V[n];
  double *d_r = d_vlast;
  double temp_real = 1.0;
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_vprev;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  /* initialize memory and cuda descriptors */
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  mp_zeros_d_set(MP_COL_MAJOR, m_H, 1, h_br, ld_H);
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnVec(&descr_b, n, (void *) d_b, CUDA_R_64F);
  cusparseCreateDnVec(&descr_x, n, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&descr_r, n, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&descr_w, n, d_w, CUDA_R_64F);
  cusparseCreateDnVec(&descr_vprev, n, d_vprev, CUDA_R_64F);
  /* first iteration */
  cudaDeviceSynchronize();
  cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL, descr_r,
    CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
  cublasDnrm2(context_gpu->cublas_handle, n, d_b, 1, &b_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*n, cudaMemcpyDeviceToDevice);
  cublasDnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);

  /* outer-loop (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_d_set(MP_COL_MAJOR, ld_H, 1, h_br,  ld_H);
    h_br[0] = r_norm;
    temp_real = 1/r_norm;
    cublasDscal(context_gpu->cublas_handle, n, &temp_real, d_V, 1);
    for (j = 0; j < inner_iterations; ++j)
    {
      d_w = &d_V[n*(j+1)];
      d_vprev = &d_V[n*j];
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_vprev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_REAL, descr_A, descr_vprev, &ZERO_REAL, descr_w,
        CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);

      for (i = 0; i < j + 1; ++i)
      {
        d_vprev = &d_V[n*i];
        cusparseDnVecSetValues(descr_vprev, d_vprev);
        cublasDdot(context_gpu->cublas_handle, n, d_w, 1, d_vprev, 1,
          &H[m_H*j+i]);
        temp_real = -H[ld_H*j+i];
        cublasDaxpy(context_gpu->cublas_handle, n, &temp_real, d_vprev, 1, d_w,
          1);
      }
      cublasDnrm2(context_gpu->cublas_handle, n, d_w, 1, &H[ld_H*j+j+1]);
      if (fabs(H[ld_H*j+j+1]) <= 1e-12)
      {
          inner_iterations = j; //@BUG: j+1(?)
          m_H = inner_iterations;
          n_H = inner_iterations-1;
          break;
      }
      temp_real = 1/H[ld_H*j+j+1];
      cublasDscal(context_gpu->cublas_handle, n, &temp_real, d_w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria -- */
    mp_qr_givens_dge(n_H, 1, H, ld_H, h_br, ld_H, temp_matrix); //@BUG: use function with input ld_H
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, h_br, m_H);
    cudaMemcpy(d_br, h_br, (sizeof *d_br)*m_H, cudaMemcpyHostToDevice);
    cublasDgemv(context_gpu->cublas_handle, CUBLAS_OP_N, n, n_H, &ONE_REAL, d_V,
      n, d_br, 1, &ONE_REAL, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL, descr_r,
      CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cublasDnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("norm residual: %1.4E\n", r_norm/b_norm);
    #endif
    #if MODE == MP_PROFILE

    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      inner_iterations = meta.iterations;
      m_H = meta.iterations+1;
      n_H = meta.iterations;
    }
  }
  d_b = NULL;
  d_x = NULL;
  d_V = NULL;
  d_br = NULL;
  d_vfirst = NULL;
  d_vlast = NULL;
  d_vprev = NULL;
  d_w = NULL;
  d_r = NULL;
}

void mp_cuda_zsy_gmres
(
  /* solver parameters */
  MPContextCuda *context_gpu,   /* (input) */
  const KrylovMeta meta,        /* (input) */

  /* data */
  MPSparseCsr_Cuda *A,             /* (input) */
  const MPInt n,                /* (input) */
  const cuDoubleComplex *d_b,   /* (input) */
  cuDoubleComplex *d_x,         /* (output) */
  MPComplexDouble *memory_cpu,  /* (input/output) */
  cuDoubleComplex *memory_gpu,  /* (input/output) */

  /* solver metadata */
  MPSolverInfo *info            /* (output )*/
)
{
  /* constants */
  const MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  const cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  double b_norm = 0;
  double r_norm = 0;
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = 1+(CudaInt)meta.restarts;
  /* mathematical object dimensions */
  CudaInt m_H = (CudaInt)meta.iterations+1;
  CudaInt n_H = (CudaInt)meta.iterations;
  CudaInt ld_H = m_H;
  /* - cpu memory */
  MPComplexDouble *H = memory_cpu;
  MPComplexDouble *h_br = &H[m_H*n_H];
  /* - gpu memory */
  cuDoubleComplex *d_V = memory_gpu;
  cuDoubleComplex *d_br = &d_V[n*((CudaInt)meta.iterations+1)];
  /* - map handles to already accessed gpu memory */
  cuDoubleComplex *d_vfirst = d_V;
  cuDoubleComplex *d_vlast = &d_V[n*(CudaInt)meta.iterations];
  cuDoubleComplex *d_vprev = d_vfirst;
  cuDoubleComplex *d_w = &d_V[n];
  cuDoubleComplex *d_r = d_vlast;
  MPComplexDouble temp_complex = ONE_C_CPU;
  
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_vprev;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  double h_abs = 0.0;
  
  /* initialize memory and cuda descriptors */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  mp_zeros_z_set(MP_COL_MAJOR, m_H, 1, h_br, ld_H);
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, n, (void *) d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, n, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r, n, d_r, CUDA_C_64F);
  cusparseCreateDnVec(&descr_w, n, d_w, CUDA_C_64F);
  cusparseCreateDnVec(&descr_vprev, n, d_vprev, CUDA_C_64F);

  /* first iteration */
  cudaDeviceSynchronize();
  cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
  cusparseDnVecSetValues(descr_r, d_r);
  cudaDeviceSynchronize();
  //er = cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_x, &ONE_C,
  //             descr_r, CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);
  cudaDeviceSynchronize();
  cublasDznrm2(context_gpu->cublas_handle, n, d_b, 1, &b_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*n, cudaMemcpyDeviceToDevice);
  cublasDznrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, 1, h_br, ld_H);
    h_br[0] = mp_scalar_z_set(r_norm);
    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, r_norm);
    cublasZscal(context_gpu->cublas_handle, n, (cuDoubleComplex *)&temp_complex,
      d_V, 1);
    #if DEBUG == 1
      printf("init\n");
      printf("====\n");
      printf("    norm_b: %1.4E\n", b_norm);
      printf("    r_norm: %1.4E\n", r_norm);
      printf("iterations: 1/%d\n", meta.iterations);
    #endif
    for (j = 0; j < inner_iterations; ++j)
    {
      d_w = &d_V[n*(j+1)];
      d_vprev = &d_V[n*j];
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_vprev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_C, descr_A, descr_vprev, &ZERO_C,
        descr_w, CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);

      for (i = 0; i < j + 1; ++i)
      {
        d_vprev = &d_V[n*i];
        cusparseDnVecSetValues(descr_vprev, d_vprev);
        cublasZdotu(context_gpu->cublas_handle, n, d_w, 1, d_vprev, 1,
          (cuDoubleComplex*)&H[m_H*j+i]);
        temp_complex = mp_scalar_z_invert_sign(H[ld_H*j+i]);
        cublasZaxpy(context_gpu->cublas_handle, n,
          (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);
      }
      cublasDznrm2(context_gpu->cublas_handle, n, d_w, 1,
        (double*)&H[ld_H*j+j+1]); //@BUG: this cast might be dangerous
      mp_vectorized_z_abs(1, &H[ld_H*j+j+1], &h_abs);
      if (h_abs <= 1e-12)
      {
        inner_iterations = j;
        m_H = inner_iterations;
        n_H = inner_iterations-1;
        break;
      }
      temp_complex = mp_scalar_z_divide(ONE_C_CPU, H[ld_H*j+j+1]);
      cublasZscal(context_gpu->cublas_handle, n,
        (cuDoubleComplex*)&temp_complex, d_w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    //mp_qr_givens_zsy_factorize(n_H, 1, H, m_H, h_br, m_H, temp_matrix);
    mp_qr_zsy_givens(H, h_br, m_H, n_H, 1); // @BUG: pass ld as input
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, h_br, m_H);
    cudaMemcpy(d_br, h_br, (sizeof *d_br)*m_H, cudaMemcpyHostToDevice);
    cublasZgemv(context_gpu->cublas_handle, CUBLAS_OP_N, n, n_H, &ONE_C,
      d_V, n, d_br, 1, &ONE_C, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_x, &ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("           r_norm: %1.4E\n", r_norm);
      printf("relative residual: %1.4E\n", r_norm/b_norm);
    #endif
    #if MODE == MP_PROFILE

    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = meta.iterations;
      m_H = meta.iterations+1;
      n_H = meta.iterations;
    }
  }

  d_b = NULL;
  d_x = NULL;
  d_V = NULL;
  d_br = NULL;
  d_vfirst = NULL;
  d_vlast = NULL;
  d_vprev = NULL;
  d_w = NULL;
  d_r = NULL;
}

void mp_cuda_zge_gmres
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPSparseCsr_Cuda *A,
  const MPInt n,
  const cuDoubleComplex *d_b,
  cuDoubleComplex *d_x,
  MPComplexDouble *memory_cpu,
  cuDoubleComplex *memory_gpu,

  /* metadata */
  MPSolverInfo *info
)
{
  /* constants */
  const MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  const cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  MPInt k = 0;
  double b_norm = 0;
  double r_norm = 0;
  /* solver meta */
  MPInt inner_iterations = (CudaInt)meta.iterations;
  MPInt outer_iterations = 1+(CudaInt)meta.restarts;
  /* mathematical object dimensions */
  MPInt m_H = (CudaInt)meta.iterations+1;
  MPInt n_H = (CudaInt)meta.iterations;
  MPInt ld_H = m_H;
  /* - cpu memory */
  MPComplexDouble *H = memory_cpu;
  MPComplexDouble *h_br = &H[m_H*n_H];
  MPComplexDouble *temp_matrix = &h_br[m_H];
  /* - gpu memory */
  cuDoubleComplex *d_V = memory_gpu;
  cuDoubleComplex *d_br = &d_V[n*(ld_H+1)];
  /* - map handles to already accessed gpu memory */
  cuDoubleComplex *d_vfirst = d_V;
  cuDoubleComplex *d_vlast = &d_V[n*(CudaInt)meta.iterations];
  cuDoubleComplex *d_vprev = d_vfirst;
  cuDoubleComplex *d_w = &d_V[n];
  cuDoubleComplex *d_r = d_vlast;
  MPComplexDouble temp_complex = ONE_C_CPU;

  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_vprev;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  double h_abs = 0.0;
  /* initialize memory and cuda descriptors */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  mp_zeros_z_set(MP_COL_MAJOR, m_H, 1, h_br, ld_H);
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, n, (void *) d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, n, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r, n, d_r, CUDA_C_64F);
  cusparseCreateDnVec(&descr_w, n, d_w, CUDA_C_64F);
  cusparseCreateDnVec(&descr_vprev, n, d_vprev, CUDA_C_64F);

  /* first iteration */
  cudaDeviceSynchronize();
  cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
  cusparseDnVecSetValues(descr_r, d_r);
  cudaDeviceSynchronize();
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r,
    CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);
  cudaDeviceSynchronize();
  cublasDznrm2(context_gpu->cublas_handle, n, d_b, 1, &b_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*n, cudaMemcpyDeviceToDevice);
  cublasDznrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);

  /* outer-loop (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, 1, h_br,  ld_H);
    h_br[0] = mp_scalar_z_init(r_norm, 0.0);
    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, r_norm);
    cublasZscal(context_gpu->cublas_handle, n, (cuDoubleComplex *)&temp_complex,
      d_V, 1);
    for (j = 0; j < inner_iterations; ++j)
    {
      d_w = &d_V[n*(j+1)];
      d_vprev = &d_V[n*j];
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_vprev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_C, descr_A, descr_vprev, &ZERO_C,
        descr_w, CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);

      for (i = 0; i < j+1; ++i)
      {
        d_vprev = &d_V[n*i];
        cusparseDnVecSetValues(descr_vprev, d_vprev);
        cublasZdotc(context_gpu->cublas_handle, n, d_w, 1, d_vprev, 1,
          (cuDoubleComplex*)&H[m_H*j+i]);
        temp_complex = mp_scalar_z_invert_sign(H[ld_H*j+i]);
        cublasZaxpy(context_gpu->cublas_handle, n,
          (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);
      }
      cublasDznrm2(context_gpu->cublas_handle, n, d_w, 1, &h_abs);
      H[m_H*j+j+1] = mp_scalar_z_init(h_abs, 0.0);
      if (h_abs <= 1e-12)
      {
          inner_iterations = j; //@BUG: this should be j+1
          m_H = inner_iterations;
          n_H = inner_iterations-1;
          break;
      }
      temp_complex = mp_scalar_z_divide(ONE_C_CPU, H[ld_H*j+j+1]);
      cublasZscal(context_gpu->cublas_handle, n,
        (cuDoubleComplex*)&temp_complex, d_w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    //mp_qr_givens_zsy_factorize(n_H, 1, H, m_H, h_br, m_H, temp_matrix);
    //mp_qr_givens_zsy_factorize(H, h_br, m_H, n_H, 1); //original
    mp_qr_zge_givens(H, h_br, m_H, n_H, 1, temp_matrix);//@BUG: should include ld_H
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, h_br, ld_H);
    cudaMemcpy(d_br, h_br, (sizeof *d_br)*m_H, cudaMemcpyHostToDevice);
    cublasZgemv(context_gpu->cublas_handle, CUBLAS_OP_N, n, n_H, &ONE_C,
      d_V, n, d_br, 1, &ONE_C, d_x, 1);

    /* computes residual */
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_x, &ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("r_norm: %1.4E\n", r_norm);
      printf("relative residual: %1.4E\n", r_norm/b_norm);
      printf("iterations: %d/%d\n", n_H, inner_iterations);
    #endif
    #if MODE == MP_PROFILE
    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = (CudaInt)meta.iterations;
      m_H = (CudaInt)meta.iterations+1;
      n_H = (CudaInt)meta.iterations;
    }
  }

  d_b = NULL;
  d_x = NULL;
  d_V = NULL;
  d_br = NULL;
  d_vfirst = NULL;
  d_vlast = NULL;
  d_vprev = NULL;
  d_w = NULL;
  d_r = NULL;
}

void mp_cuda_sge_gmres
(
  /* solver parameters */
  const MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPSparseCsr_Cuda *A,
  const MPInt n,
  float *d_b,
  float *d_x,
  float *memory_cpu,
  float *memory_gpu,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  const float ONE_REAL = 1.0;
  const float MINUS_ONE_REAL = -1.0;
  const float ZERO_REAL = 0.0;
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  MPInt k = 0;
  float norm_b = 0;
  float r_norm = 0;
  float temp_real = 1.0;
  /* solver metadata */
  MPInt inner_iterations = meta.iterations;
  MPInt outer_iterations = 1+meta.restarts;
  /* mathematical object dimensions */
  MPInt m_H = meta.iterations+1;
  MPInt n_H = meta.iterations;
  MPInt ld_H = m_H;
  /* mappings of mathematical objects to allocated memory */
  /* - cpu memory */
  float *H = memory_cpu;
  float *h_br = &H[m_H*n_H];
  float *temp_matrix = &h_br[m_H];
  /* - gpu memory */
  float *d_V = memory_gpu;
  float *d_br = &d_V[n*(meta.iterations+1)];
  /* - other handles on accessed gpu memory */
  float *d_vfirst = d_V;
  float *d_vlast = &d_V[n*meta.iterations];
  float *d_vprev = d_vfirst;
  float *d_w = &d_vfirst[n];
  float *d_r = d_vlast;
  /* mathematical object descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_vprev;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  /* initialization of memory and cuda descriptors */
  mp_zeros_s_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  mp_zeros_s_set(MP_COL_MAJOR, m_H, 1, h_br, ld_H);
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_32F);
  cusparseCreateDnVec(&descr_b, n, d_b, CUDA_R_32F);
  cusparseCreateDnVec(&descr_x, n, d_x, CUDA_R_32F);
  cusparseCreateDnVec(&descr_r, n, d_r, CUDA_R_32F);
  cusparseCreateDnVec(&descr_w, n, d_w, CUDA_R_32F);
  cusparseCreateDnVec(&descr_vprev, n, d_vprev, CUDA_R_32F);

  /* first iteration */
  cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL,
               descr_r, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, NULL);
  cublasSnrm2(context_gpu->cublas_handle, n, d_b, 1, &norm_b);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*n, cudaMemcpyDeviceToDevice);
  cublasSnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);

  /* outer-loop (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_s_set(MP_COL_MAJOR, ld_H, 1, h_br, ld_H);
    h_br[0] = r_norm;
    temp_real = 1/r_norm;
    cublasSscal(context_gpu->cublas_handle, n, &temp_real, d_V, 1);
    for (j = 0; j < inner_iterations; ++j)
    {
      d_w = &d_V[n*(j+1)];
      d_vprev = &d_V[n*j];
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_vprev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_REAL, descr_A, descr_vprev, &ZERO_REAL, descr_w,
        CUDA_R_32F, CUSPARSE_CSRMV_ALG1, NULL);
      for (i = 0; i < j+1; ++i)
      {
        d_vprev = &d_V[n*i];
        cusparseDnVecSetValues(descr_vprev, d_vprev);
        cublasSdot(context_gpu->cublas_handle, n, d_w, 1, d_vprev, 1,
          &H[ld_H*j+i]);
        temp_real = -H[ld_H*j+i];
        cublasSaxpy(context_gpu->cublas_handle, n, &temp_real, d_vprev, 1, d_w,
          1);
      }
      cublasSnrm2(context_gpu->cublas_handle, n, d_w, 1, &H[ld_H*j+j+1]);
      if (fabs(H[m_H*j+j+1]) <= 1e-12)
      {
        inner_iterations = j;
        m_H = inner_iterations;
        n_H = inner_iterations-1;
        break;
      }
      temp_real = 1/H[m_H*j+j+1];
      cublasSscal(context_gpu->cublas_handle, n, &temp_real, d_w, 1);
    }

    /* constructs solution to the linear system of equations and checks
       termination criteria */
    mp_qr_givens_sge(n_H, 1, H, ld_H, h_br, ld_H, temp_matrix);
    mp_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, h_br, m_H);
    cudaMemcpy(d_br, h_br, (sizeof *d_br)*m_H, cudaMemcpyHostToDevice);
    cublasSgemv(context_gpu->cublas_handle, CUBLAS_OP_N, n, n_H, &ONE_REAL,
      d_V, n, d_br, 1, &ONE_REAL, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*n, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_REAL, descr_A, descr_x, &ONE_REAL, descr_r,
      CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cublasSnrm2(context_gpu->cublas_handle, n, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("r_norm: %1.4E\n", r_norm);
      printf("relative residual: %1.4E\n", r_norm/norm_b);
      printf("iterations: %d/%d\n", inner_iterations, meta.iterations);
    #endif
    #if MODE == MP_PROFILE

    #endif
    if (r_norm/norm_b <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      inner_iterations = meta.iterations;
      m_H = meta.iterations+1;
      n_H = meta.iterations;
    }
  }

  d_b = NULL;
  d_x = NULL;
  d_V = NULL;
  d_br = NULL;
  d_vfirst = NULL;
  d_vlast = NULL;
  d_vprev = NULL;
  d_w = NULL;
  d_r = NULL;
}

void mp_cuda_dge_block_gmres
(
  /* solver parameters */
  const MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPSparseCsr_Cuda *A,
  const MPInt n,
  double *d_B,
  double *d_X,
  double *memory_cpu,
  double *memory_gpu,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* initialization */
  MPInt i = 0;
  MPInt j = 0;
  MPInt k = 0;
  /* meta */
  CudaInt blk = (CudaInt)meta.blk;
  CudaInt inner_iterations = meta.iterations;
  CudaInt outer_iterations = 1+meta.restarts;
  CudaInt ld_H = (meta.iterations+1)*blk;
  CudaInt m_H = (meta.iterations+1)*blk;
  CudaInt n_H = meta.iterations * blk;
  /* constants */
  const double MINUS_ONE_REAL = -1.0;
  const double ONE_REAL = 1.0;
  const double ZERO_REAL = 0.0;
  /* memory cpu */
  double *B_norms_array = memory_cpu;
  double *h_H = &B_norms_array[blk];
  double *Br = &h_H[m_H*n_H];
  double *Vtemp = &Br[m_H*blk];
  double *Htemp = &Vtemp[n*blk];
  /* handles on cpu memory */
  double *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  double *Hblk = NULL;
  /* data */
  double *d_V = memory_gpu;
  double *d_H = &d_V[n*m_H];
  double *d_Br = &d_H[m_H*n_H];
  /* handles on data */
  double *d_Vlast = &d_V[n*n_H];
  double *d_R = d_Vlast;
  double *d_Vprev = d_V;
  double *d_W = &d_Vprev[n*blk];
  double *d_Hblk = NULL;
  /* solver context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;
  /* declares descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_r;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  /* initializes descriptors */
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_B, n, blk, n, d_B, CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_X, n, blk, n, d_X, CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_r, n, blk, n, d_R, CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_Vprev, n, blk, n, d_Vprev, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_W, n, blk, n, d_W, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  /* first iteration */

  cudaMemcpy(d_V, d_B, (sizeof *d_V)*n*blk, cudaMemcpyDeviceToDevice);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_REAL, descr_A, descr_X,
    &ONE_REAL, descr_Vprev, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
  for (i = 0; i < blk; ++i)
  {
    cublasDnrm2(context_gpu->cublas_handle, n, &d_B[n*i], 1, &B_norms_array[i]);
    cublasDnrm2(context_gpu->cublas_handle, n, &d_V[n*i], 1, &r_norm);
    r_norm = r_norm/B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }
  #if DEBUG == 1
      //printf("max relative residual: %1.4E\n", r_norm_max);
  #endif
  if (r_norm_max <= meta.tolerance) /* checks terminating condition */
  {
    return;
  }

  /* main loop (outer iterations) */
  for (k = 0; k < outer_iterations; k++)
  {
    mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, h_H, ld_H);
    mp_zeros_d_set(MP_COL_MAJOR, m_H, blk, Br, ld_H);
    d_W = d_V;
    cudaMemcpy(Vtemp, d_V, (sizeof *Vtemp)*n*blk, cudaMemcpyDeviceToHost);
    mp_dgeqrf(LAPACK_COL_MAJOR, n, blk, Vtemp, n, reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, Vtemp, n, Br, m_H);
    mp_dorgqr(LAPACK_COL_MAJOR, n, blk, blk, Vtemp, n, reflectors_array);
    cudaMemcpy(d_W, Vtemp, (sizeof *d_W)*n*blk, cudaMemcpyHostToDevice);

    for (j = 0; j < inner_iterations; ++j)
    {
      d_W = &d_V[n*blk*(j+1)];
      d_Vprev = &d_V[n*blk*j];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_REAL, descr_A, descr_Vprev,
        &ZERO_REAL, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);

      for (i = 0; i < j + 1; ++i)
      {
        d_Hblk = &d_H[m_H*blk*j + blk*i];
        d_Vprev = &d_V[n*blk*i];
        cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
          blk, n, &ONE_REAL, d_W, n, d_Vprev, n, &ZERO_REAL, d_Hblk,
           m_H);
        cudaMemcpy(Htemp, &d_H[m_H*blk*j],
          (sizeof *Htemp)*m_H*blk, cudaMemcpyDeviceToHost);
        cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
          blk, blk, &MINUS_ONE_REAL, d_Vprev, n, d_Hblk, m_H, &ONE_REAL,
          d_W, n);
      }

      cudaMemcpy(Htemp, &d_H[ld_H*blk*j], (sizeof *Htemp)*m_H*blk,
        cudaMemcpyDeviceToHost);
      Hblk = &Htemp[blk*(j+1)];
      cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*n*blk, cudaMemcpyDeviceToHost);
      mp_dgeqrf(LAPACK_COL_MAJOR, n, blk, Vtemp, n, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, blk, Vtemp, n, Hblk, m_H);
      cudaMemcpy(&d_H[ld_H*blk*j], Htemp, (sizeof *d_H)*m_H*blk,
        cudaMemcpyHostToDevice);
      H_IJ_norm = mp_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, m_H);
      #if DEBUG == 1
        //printf("iterations: %d\n", iterations);
        //printf("matrix_norm_frobenious_H_IJ: %1.4E\n", norm_H_IJ);
      #endif
      if ((H_IJ_norm <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = blk*(inner_iterations+1);
        n_H = blk*inner_iterations;
        break;
      }
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, blk, blk, Vtemp, n, reflectors_array);
      cudaMemcpy(d_W, Vtemp, (sizeof *d_W)*n*blk, cudaMemcpyHostToDevice);
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    cudaMemcpy(h_H, d_H, (sizeof *h_H)*ld_H*n_H, cudaMemcpyDeviceToHost);
    mp_block_qr_dge_givens(n_H, blk, h_H, ld_H, Br, ld_H, blk);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, 1.0, h_H, ld_H, Br, ld_H);
    cudaMemcpy(d_Br, Br, (sizeof *d_Br)*ld_H*blk, cudaMemcpyHostToDevice);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, blk,
      n_H, &ONE_REAL, d_V, n, d_Br, ld_H, &ONE_REAL, d_X, n);
    cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk,
      cudaMemcpyDeviceToDevice);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_REAL, descr_A, descr_X,
      &ONE_REAL, descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    r_norm_max = 0.0;
    for (i = 0; i < blk; ++i)
    {
      cublasDnrm2(context_gpu->cublas_handle, n, &d_R[n*i], 1, &r_norm);
      r_norm = r_norm/B_norms_array[i];;
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }
    #if DEBUG == 1
      printf("max_relative residual: %1.4E\n", r_norm_max);
    #endif
    #if MODE == MP_PROFILER

    #endif
    if (r_norm_max <= meta.tolerance)
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
      inner_iterations = meta.iterations;
      m_H = (meta.iterations+1)*blk;
      n_H = (meta.iterations)*blk;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_Vprev);
  cusparseDestroyDnMat(descr_W);
  mp_free(reflectors_array);
  d_B = NULL;
  d_X = NULL;
  d_V = NULL;
  d_H = NULL;
  d_Br = NULL;
  d_Vlast = NULL;
  d_R = NULL;
  d_Vprev = NULL;
  d_W = NULL;
  d_Hblk = NULL;
}

void mp_cuda_zhe_block_gmres
(
  /* solver parameters */
  MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* input/output data */
  const MPSparseCsr_Cuda *A,
  const MPInt n,
  cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  MPComplexDouble *memory_cpu,
  cuDoubleComplex *memory_gpu,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* initialization */
  MPInt i = 0;
  MPInt j = 0;
  MPInt k = 0;
  /* meta */
  
  CudaInt blk = (CudaInt)meta.blk;
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = 1+(CudaInt)meta.restarts;
  CudaInt ld_H = ((CudaInt)meta.iterations+1)*blk;
  CudaInt m_H = ((CudaInt)meta.iterations+1)*blk;
  CudaInt n_H = (CudaInt)meta.iterations*blk;

  /* constants */
  const MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  const cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);

  /* memory cpu */
  double *B_norms_array = mp_malloc(sizeof(double)*blk);
  MPComplexDouble *h_H = memory_cpu;
  MPComplexDouble *Br = &h_H[m_H*n_H];
  MPComplexDouble *Vtemp = &Br[m_H*blk];
  MPComplexDouble *Htemp = &Vtemp[n*blk];
  /* handles on cpu memory */
  MPComplexDouble *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  MPComplexDouble *Hblk = NULL;
  /* data */
  cuDoubleComplex *d_V = memory_gpu;
  cuDoubleComplex *d_H = &d_V[n*m_H];
  cuDoubleComplex *d_Br = &d_H[m_H*n_H];

  /* handles on data */
  cuDoubleComplex *d_Vlast = &d_V[n*n_H];
  cuDoubleComplex *d_R = d_Vlast;
  cuDoubleComplex *d_Vprev = d_V;
  cuDoubleComplex *d_W = &d_Vprev[n*blk];
  cuDoubleComplex *d_Hblk = NULL;

  /* solver context */
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;

  /* declares descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_r;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;

  /* initializes descriptors */
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, n, blk, n, d_B, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, n, blk, n, d_X, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, n, blk, n, d_R, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, n, blk, n, d_Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, n, blk, n, d_W, CUDA_C_64F, CUSPARSE_ORDER_COL);

  /* first iteration */
  cudaMemcpy(d_V, d_B, (sizeof *d_V)*n*blk, cudaMemcpyDeviceToDevice);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
    &ONE_C, descr_Vprev, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  for (i = 0; i < blk; ++i)
  {
    cublasDznrm2(context_gpu->cublas_handle, n, &d_B[n*i], 1,
      &B_norms_array[i]);
    cublasDznrm2(context_gpu->cublas_handle, n, &d_V[n*i], 1, &r_norm);
    r_norm = r_norm/B_norms_array[i];
    if (r_norm > r_norm_max)
    {
      r_norm_max = r_norm;
    }
  }
  #if DEBUG == 1
      //printf("max relative residual: %1.4E\n", r_norm_max);
  #endif
  if (r_norm_max <= meta.tolerance) /* checks terminating condition */
  {
    return;
  }
  /* main loop (outer iterations) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, h_H, ld_H);
    mp_zeros_z_set(MP_COL_MAJOR, m_H, blk, Br, ld_H);
    d_W = d_V;
    cudaMemcpy(Vtemp, d_V, (sizeof *Vtemp)*n*blk, cudaMemcpyDeviceToHost);
    mp_gram_schmidt_zhe(n, blk, Vtemp, Br, ld_H);
    cudaMemcpy(d_W, Vtemp, (sizeof *d_W)*n*blk, cudaMemcpyHostToDevice);

    for (j = 0; j < inner_iterations; ++j)
    {
      d_W = &d_V[n*blk*(j+1)];
      d_Vprev = &d_V[n*blk*j];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
        &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      for (i = 0; i < j + 1; ++i)
      {
        d_Hblk = &d_H[ld_H*blk*j+blk*i];
        d_Vprev = &d_V[n*blk*i];
        cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
          blk, n, &ONE_C, d_W, n, d_Vprev, n, &ZERO_C, d_Hblk, ld_H);
        //cudaMemcpy(Htemp, &d_H[m_H*blk*j], (sizeof *Htemp) * m_H * blk, cudaMemcpyDeviceToHost);
        cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
          blk, blk, &MINUS_ONE_C, d_Vprev, n, d_Hblk, m_H, &ONE_C,
          d_W, n);
      }
      cudaMemcpy(&h_H[ld_H*j], &d_H[ld_H*j], (sizeof *h_H)*ld_H*blk,
        cudaMemcpyDeviceToHost);
      //cudaMemcpy(Htemp, &d_H[m_H*blk*j], (sizeof *Htemp) * m_H * blk, cudaMemcpyDeviceToHost);
      Hblk = &h_H[ld_H*blk*j + blk*(j+1)];
      //Hblk = &Htemp[blk*(j+1)];
      mp_gram_schmidt_zhe(n, blk, Vtemp, Hblk, ld_H);
      cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*n*blk, cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_H[ld_H*j], &d_H[ld_H*j], (sizeof *h_H)*ld_H*blk,
        cudaMemcpyDeviceToHost);
      cudaMemcpy(&d_H[ld_H*blk*j], Htemp, (sizeof *d_H)*ld_H*blk,
        cudaMemcpyHostToDevice);
      H_IJ_norm = mp_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      #if DEBUG == 1
        //printf("iterations: %d\n", iterations);
        //printf("matrix_norm_frobenious_H_IJ: %1.4E\n", H_IJ_norm);
      #endif
      printf("matrix_norm_frobenious_H_IJ: %1.4E\n", H_IJ_norm);
      if ((H_IJ_norm <= 1e-12) || (j == (CudaInt)meta.iterations-1))
      {
        //rintf("matrix_norm_frobenious_H_IJ: %1.4E\n", H_IJ_norm);
        inner_iterations = j+1;
        m_H = blk*(inner_iterations+1);
        n_H = blk*inner_iterations;
        break;
      }
      cudaMemcpy(d_W, Vtemp, (sizeof *d_W)*n*blk, cudaMemcpyHostToDevice);
    }

    /* solves system of equations using qr decomposition and evaluates
       termination criteria */
    //cudaMemcpy(h_H, d_H, (sizeof *h_H) * max_m_H * n_H, cudaMemcpyDeviceToHost);
    mp_block_qr_zge_givens(h_H, Br, m_H, n_H, blk, blk);  //@BUG: ld_H should be passed as input
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, blk, &ONE_C_CPU,  h_H, ld_H, Br, ld_H);
    cudaMemcpy(d_Br, Br, (sizeof *d_Br)*ld_H*blk, cudaMemcpyHostToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, blk,
      n_H, &ONE_C, d_V, n, d_Br, ld_H, &ONE_C, (cuDoubleComplex*)d_X, n);

    cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
      &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    r_norm_max = 0.0;
    for (i = 0; i < blk; ++i)
    {
      cublasDznrm2(context_gpu->cublas_handle, n, &d_R[n*i], 1, &r_norm);
      r_norm = r_norm/B_norms_array[i];;
      if (r_norm > r_norm_max)
      {
          r_norm_max = r_norm;
      }
    }
    #if DEBUG == 1
      printf("max_relative residual: %1.4E\n", r_norm_max);
    #endif
    if (r_norm_max <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
      inner_iterations = (CudaInt)meta.iterations;
      m_H = ((CudaInt)meta.iterations+1)*blk;
      n_H = ((CudaInt)meta.iterations)*blk;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_Vprev);
  cusparseDestroyDnMat(descr_W);
  mp_free(B_norms_array);
  mp_free(reflectors_array);
  d_B = NULL;
  d_X = NULL;
  d_V = NULL;
  d_H = NULL;
  d_Br = NULL;
  d_Vlast = NULL;
  d_R = NULL;
  d_Vprev = NULL;
  d_W = NULL;
  d_Hblk = NULL;
}


void mp_cuda_dge_global_gmres
(
  /* solver parameters */
  MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* input/output data */
  const MPInt n,
  const MPInt blk,
  const MPSparseCsr_Cuda *A,
  const double *d_B,
  double *d_X,
  double *memory_cpu,
  double *memory_gpu,

  /* solver related information */
  MPSolverInfo *info
)
{
  /* context */
  CusparseInt i = 0;
  CusparseInt j = 0;
  CusparseInt k = 0;
  CusparseInt z = 0;
  double B_norm = 0.0;
  double R_norm = 0.0;
  double temp_real = 0.0;
  double trace = 0.0;
  double temp_trace = 0.0;
  size_t size_buffer = 0;
  /* constants */
  const double ONE_REAL = 1.0;
  const double MINUS_ONE_REAL = -1.0;
  const double ZERO_REAL = 0.0;
  /* dimension related stuff*/
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = (CudaInt)meta.restarts+1;
  CudaInt ld_H = (CudaInt)meta.iterations+1;
  CudaInt m_H = (CudaInt)meta.iterations+1;
  CudaInt n_H = (CudaInt)meta.iterations;
  /* descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  CusparseDenseMatrixDescriptor descr_r;
  /* host memory */
  double *h_H = memory_cpu;
  double *br = &h_H[m_H*n_H];
  double *temp_matrix = &br[m_H]; // device memory
  /* device memory */
  double *d_V = memory_gpu;
  double *d_Vfirst = d_V;
  double *d_Vlast = &d_V[n*blk*(CudaInt)meta.iterations];
  double *d_R = d_Vlast;
  double *Vprev = d_Vfirst;
  double *d_W = &d_V[n*blk];

  /* sets descriptors */
  cusparseCreateCsr(&descr_A, n, n, A->nz, A->d_row_pointers, A->d_cols,
    A->d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_B, n, blk, n, (void *) d_B, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_X, n, blk, n, (void *) d_X, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_r, n, blk, n, d_R, CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_W, n, blk, n, d_W, CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_Vprev, n, blk, n, Vprev, CUSPARSE_ORDER_COL,
    CUDA_R_64F);

  cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);

  /* first iteration */
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_REAL, descr_A, descr_X,
    &ONE_REAL, descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);

  cublasDnrm2(context_gpu->cublas_handle, n*blk, d_B, 1, &B_norm);
  cublasDnrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
  cudaMemcpy(d_Vfirst, d_R, (sizeof *d_Vfirst)*n*blk, cudaMemcpyDeviceToDevice);
  temp_real = 1/R_norm;
  cublasDscal(context_gpu->cublas_handle, n*blk, &temp_real, d_Vfirst, 1);
  cudaDeviceSynchronize();
  if (R_norm/B_norm <= meta.tolerance)
  {
    return;
  }

  /* outer iterations */
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, h_H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_d_set(MP_COL_MAJOR, ld_H, 1, br, ld_H);
    br[0] = R_norm;
    /* inner iterations */
    for (j = 0; j < inner_iterations; j++)
    {
      d_W = &d_Vfirst[(n*blk)*(j+1)];
      Vprev = &d_Vfirst[(n * blk)*j];
      cusparseDnMatSetValues(descr_Vprev, Vprev);
      cusparseDnMatSetValues(descr_W, d_W); //@NOTE: the following 2 lines are required
      cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_REAL, descr_A, descr_Vprev, &ZERO_REAL, descr_W, CUDA_R_64F,
        CUSPARSE_CSRMM_ALG1, &size_buffer);

      if (size_buffer > 0)
      {
        void *d_external_buffer = NULL;
        cudaMalloc(d_external_buffer, size_buffer);
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_REAL, descr_A,
          descr_Vprev, &ZERO_REAL, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
          d_external_buffer);
        cudaFree(d_external_buffer);
      }
      else if (size_buffer == 0)
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_REAL, descr_A, descr_Vprev,
          &ZERO_REAL, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      for (i = 0; i < j+1; ++i)
      {
        Vprev = &d_Vfirst[(n*blk)*i];
        trace = 0.0;
        for (z = 0; z < blk; ++z)
        {
          temp_trace = 0.0;
          cublasDdot(context_gpu->cublas_handle, n, &Vprev[n*z], 1, &d_W[n*z],
            1, &temp_trace);
          cudaDeviceSynchronize();
          trace += temp_trace;
          cudaDeviceSynchronize();
        }
        h_H[ld_H*j+i] = trace;
        trace = -trace;
        cublasDaxpy(context_gpu->cublas_handle, n*blk, &trace, Vprev, 1, d_W, 1);
      }
      cublasDnrm2(context_gpu->cublas_handle, n*blk, d_W, 1, &h_H[ld_H*j+j+1]);
      if ((fabs(h_H[ld_H*j+j+1]) <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = (j+1);
        n_H = j;
        break;
      }
      else
      {
        temp_real = 1/h_H[ld_H*j+j+1];
        cublasDscal(context_gpu->cublas_handle, n*blk, &temp_real, d_W, 1);
      }
    }
    /* constructs solution to the linear system of equations */
    mp_qr_givens_dge(n_H, blk, h_H, ld_H, br, ld_H, temp_matrix);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
             n_H, 1, 1.0, h_H, ld_H, br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_Vfirst[n*blk*i];
      cublasDaxpy(context_gpu->cublas_handle, n*blk, &br[i], d_W, 1, d_X, 1);
    }

    cusparseDnMatSetValues(descr_X, d_X);
    cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_REAL, descr_A, descr_X,
      &MINUS_ONE_REAL, descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    cublasDnrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
    #if DEBUG == 1
      printf("relative norm_frobenious_residual: %1.4E\n", R_norm/B_norm);
    #endif
    if ((R_norm/B_norm <= meta.tolerance) || (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V)*n*blk, cudaMemcpyDeviceToDevice);
      temp_real = 1/R_norm;
      cublasDscal(context_gpu->cublas_handle, n*blk, &temp_real, d_V, 1);
      inner_iterations = (CudaInt)meta.iterations;
      m_H = (CudaInt)meta.iterations+1;
      n_H = (CudaInt)meta.iterations;
    }
  }

  /* transfers solution to vectors_block_X */
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_zsy_global_gmres_2
(
  /* solver input parameters */
  MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* input/output data */
  const MPInt n,
  const MPInt blk,
  const MPSparseCsr_Cuda *A,
  const cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  MPComplexDouble *memory_cpu,
  cuDoubleComplex *memory_gpu,

  /* collects solver information */
  MPSolverInfo *info
)
{
  /* context */
  CusparseInt i = 0;
  CusparseInt j = 0;
  CusparseInt k = 0;
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble ZERO_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  double B_norm = 0.0;
  double R_norm = 0.0;
  MPComplexDouble temp_complex = ZERO_C_CPU;
  MPComplexDouble trace = ZERO_C_CPU;
  size_t size_buffer = 0;
  /* dimension related stuff*/
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = (CudaInt)meta.restarts+1;
  CudaInt ld_H = (CudaInt)meta.iterations+1;
  CudaInt m_H = (CudaInt)meta.iterations+1;
  CudaInt n_H = (CudaInt)meta.iterations;
  double h_abs = 0.0;
  /* descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  CusparseDenseMatrixDescriptor descr_r;
  /* host memory */
  MPComplexDouble *h_H = memory_cpu;
  MPComplexDouble *br = &h_H[m_H*n_H];
  MPComplexDouble *temp_matrix = &br[m_H];
  /* device memory */
  cuDoubleComplex *d_V = (cuDoubleComplex *) memory_gpu;
  cuDoubleComplex *d_Vfirst = d_V;
  cuDoubleComplex *d_Vlast = &d_V[n*blk*(CudaInt)meta.iterations];
  cuDoubleComplex *d_R = d_Vlast;
  cuDoubleComplex *Vprev = d_Vfirst;
  cuDoubleComplex *d_W = &d_V[n*blk];
  /* sets descriptors */
  cusparseCreateCsr(&descr_A, n, n, A->nz,
    A->d_row_pointers, A->d_cols, A->d_data, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, n, blk, n, (void *) d_B, CUDA_C_64F,
     CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, n, blk, n, (void *) d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, n, blk, n, d_R, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, n, blk, n, d_W, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, n, blk, n, Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);


  /* first iteration */
  cudaMemcpy(d_R, d_B, (sizeof *d_R) * n * blk, cudaMemcpyDeviceToDevice);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
    &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);

  cublasDznrm2(context_gpu->cublas_handle, n*blk, d_B, 1, &B_norm);
  cublasDznrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
  cudaMemcpy(d_Vfirst, d_R, (sizeof *d_Vfirst)*n*blk, cudaMemcpyDeviceToDevice);
  temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
  cublasZscal(context_gpu->cublas_handle, n*blk,
    (cuDoubleComplex*)&temp_complex, d_Vfirst, 1);
  cudaDeviceSynchronize();
  if (R_norm/B_norm <= meta.tolerance)
  {
    return;
  }
  printf("norm_frobenious_residual: %1.4E\n", R_norm);
  /* outer iterations */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, h_H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, 1, br, ld_H);
    br[0] = mp_scalar_z_init(R_norm, 0.0);
    /* inner iterations */
    for (j = 0; j < inner_iterations; ++j)
    {
      d_W = &d_Vfirst[(n*blk)*(j+1)];
      Vprev = &d_Vfirst[(n*blk)*j];
      cusparseDnMatSetValues(descr_Vprev, Vprev);
      cusparseDnMatSetValues(descr_W, d_W); // the following 2 lines are required
      cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_Vprev, &ZERO_C, descr_W, CUDA_C_64F,
        CUSPARSE_CSRMM_ALG1, &size_buffer);

      if (size_buffer > 0)
      {
        void *d_external_buffer = NULL;
        cudaMalloc(d_external_buffer, size_buffer);
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
          d_external_buffer);

        cudaFree(d_external_buffer);
      }
      else if (size_buffer == 0)
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      for (i = 0; i < j+1; ++i)
      {
        Vprev = &d_Vfirst[(n*blk)*i];
        cudaDeviceSynchronize();
        cublasZdotu(context_gpu->cublas_handle, n*blk, Vprev, 1, d_W, 1,
          (cuDoubleComplex *)&trace);
        cudaDeviceSynchronize();
        h_H[ld_H*j+i] = trace;
        trace = mp_scalar_z_invert_sign(trace);
        cublasZaxpy(context_gpu->cublas_handle, n*blk, (cuDoubleComplex*)&trace,
          Vprev, 1, d_W, 1);
      }
      cublasDznrm2(context_gpu->cublas_handle, n*blk, d_W, 1, (double*)&h_H[ld_H*j+j+1]); //@BUG: this casting is kind of dangerous but should work
      mp_vectorized_z_abs(1, &h_H[ld_H*j+j+1], &h_abs);
      if ((h_abs <= 1e-12) || (j == inner_iterations-1))  /* checks termination condition */
      {
        inner_iterations = j+1;
        m_H = (inner_iterations+1);
        n_H = inner_iterations;
        break;
      }
      else
      {
        temp_complex = mp_scalar_z_divide(ONE_C_CPU, h_H[ld_H*j+j+1]);
        cublasZscal(context_gpu->cublas_handle, n*blk,
          (cuDoubleComplex*)&temp_complex, d_W, 1);
        inner_iterations = (CudaInt)meta.iterations;
        m_H = (CudaInt)meta.iterations+1;
        n_H = (CudaInt)meta.iterations;
      }
    }
    /* constructs solution to the linear system of equations */
    //mp_qr_givens_zge_factorize(h_H, br, m_H, n_H, 1, temp_matrix);
    mp_qr_givens_zge_2(h_H, ld_H, br, ld_H, m_H, n_H, 1, temp_matrix);    //@BUG: set input ld_H
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, h_H, ld_H, br, ld_H);

    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_Vfirst[n*blk*i];
      cublasZaxpy(context_gpu->cublas_handle, n*blk, (cuDoubleComplex*)&br[i],
        d_W, 1, d_X, 1);
    }

    cusparseDnMatSetValues(descr_X, d_X);
    cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_X,
      &MINUS_ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
    #if DEBUG == 1
      printf(">relative norm_frobenious_residual: %1.4E\n", R_norm/B_norm);
    #endif
    if ((R_norm/B_norm <= meta.tolerance) || (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V)*n*blk, cudaMemcpyDeviceToDevice);
      temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
      cublasZscal(context_gpu->cublas_handle, n*blk,
        (cuDoubleComplex*)&temp_complex, d_V, 1);
      inner_iterations = (CudaInt)meta.iterations;
      m_H = (CudaInt)meta.iterations+1;
      n_H = (CudaInt)meta.iterations;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_zhe_global_gmres_2
(
  /* solver parameters */
  const MPContextCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPSparseCsr_Cuda *A_csr,
  const MPInt n,
  const cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  MPComplexDouble *memory_cpu,
  cuDoubleComplex *memory_gpu,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* context */
  CusparseInt i = 0;
  CusparseInt j = 0;
  CusparseInt k = 0;
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble ZERO_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  double B_norm = 0.0;
  double R_norm = 0.0;
  MPComplexDouble temp_complex = ZERO_C_CPU;
  MPComplexDouble trace = ZERO_C_CPU;
  /* dimension related stuff*/
  CudaInt blk = (CudaInt)meta.blk;
  CudaInt inner_iterations = (CudaInt)meta.iterations;
  CudaInt outer_iterations = (CudaInt)meta.restarts+1;
  CudaInt ld_H = (CudaInt)meta.iterations+1;
  CudaInt m_H = (CudaInt)meta.iterations+1;
  CudaInt n_H = (CudaInt)meta.iterations;
  size_t size_buffer;
  double h_abs = 0.0;
  /* descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  CusparseDenseMatrixDescriptor descr_r;
  /* host memory */
  MPComplexDouble *h_H = memory_cpu;
  MPComplexDouble *br = &h_H[m_H*n_H];
  MPComplexDouble *temp_matrix = &br[m_H]; // device memory
  /* device memory */
  cuDoubleComplex *d_V = memory_gpu;
  cuDoubleComplex *d_Vfirst = d_V;
  cuDoubleComplex *d_Vlast = &d_V[n*blk*(CudaInt)meta.iterations];
  cuDoubleComplex *d_R = d_Vlast;
  cuDoubleComplex *Vprev = d_Vfirst;
  cuDoubleComplex *d_W = &d_V[n*blk];
  /* sets descriptors */
  cusparseCreateCsr(&descr_A, n, n, A_csr->nz,
    A_csr->d_row_pointers, A_csr->d_cols, A_csr->d_data, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, n, blk, n, (void *) d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, n, blk, n, (void *) d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, n, blk, n, d_R, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, n, blk, n, d_W, CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, n, blk, n, Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);

  /* first iteration */
  cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
    &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);

  cublasDznrm2(context_gpu->cublas_handle, n*blk, d_B, 1, &B_norm);
  cublasDznrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
  cudaMemcpy(d_Vfirst, d_R, (sizeof *d_Vfirst) * n * blk,
    cudaMemcpyDeviceToDevice);
  temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
  cublasZscal(context_gpu->cublas_handle, n*blk,
    (cuDoubleComplex*)&temp_complex, d_Vfirst, 1);
  cudaDeviceSynchronize();
  if (R_norm/B_norm <= meta.tolerance)
  {
    return;
  }
  /* outer iterations */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, h_H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, m_H, 1, br, ld_H);
    br[0] = mp_scalar_z_init(R_norm, 0.0);
    /* inner iterations */
    for (j = 0; j < inner_iterations; j++)
    {
      d_W = &d_Vfirst[(n*blk)*(j+1)];
      Vprev = &d_Vfirst[(n*blk)*j];
      cusparseDnMatSetValues(descr_Vprev, Vprev);
      cusparseDnMatSetValues(descr_W, d_W);
      cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_Vprev, &ZERO_C, descr_W, CUDA_C_64F,
        CUSPARSE_CSRMM_ALG1, &size_buffer);
      if (j == 0)
      {
        MPComplexDouble *t = mp_malloc(sizeof(MPComplexDouble)*n*blk);
        cudaMemcpy(t, d_W, sizeof(MPComplexDouble)*n*blk,
          cudaMemcpyDeviceToHost);
        mp_free(t);
      }
      if (size_buffer > 0)
      {
        void *d_external_buffer = NULL;
        cudaMalloc(d_external_buffer, size_buffer);
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
        cudaFree(d_external_buffer);
      }
      else if (size_buffer == 0)
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      for (i = 0; i < j+1; ++i)
      {
        Vprev = &d_Vfirst[(n*blk)*i];
        cudaDeviceSynchronize();
        cublasZdotu(context_gpu->cublas_handle, n*blk, Vprev, 1, d_W,
          1, (cuDoubleComplex*)&trace);
        cudaDeviceSynchronize();
        h_H[ld_H*j+i] = trace;
        trace = mp_scalar_z_invert_sign(trace);
        cublasZaxpy(context_gpu->cublas_handle, n*blk,
          (cuDoubleComplex*)&trace, Vprev, 1, d_W, 1);
      }
      cublasDznrm2(context_gpu->cublas_handle, n*blk, d_W, 1,
        (double*)&h_H[ld_H*j+j+1]);
      mp_vectorized_z_abs(1, &h_H[ld_H*j+j+1], &h_abs);
      if ((h_abs <= 1e-12) || (j == inner_iterations-1))
      {
        //max_iterations = j+1;
        m_H = (j+2);
        n_H = m_H-1;
        break;
      }
      else
      {
        temp_complex = mp_scalar_z_divide(ONE_C_CPU, h_H[ld_H*j+j+1]);
        cublasZscal(context_gpu->cublas_handle, n*blk,
          (cuDoubleComplex*)&temp_complex, d_W, 1);
        inner_iterations = j;
        m_H = inner_iterations+1;
        n_H = inner_iterations;
      }
    }

    /* constructs solution to the linear system of equations */
    //mp_qr_givens_zge_factorize(h_H, br, m_H, n_H, 1, temp_matrix);
    // use that in order to avoid problems where max_m_H != m_H
    mp_qr_zge_givens_3(h_H, ld_H, br, ld_H, m_H, n_H, 1, temp_matrix);
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, h_H, ld_H, br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_Vfirst[n*blk*i];
      cublasZaxpy(context_gpu->cublas_handle, n*blk, (cuDoubleComplex*)&br[i],
        d_W, 1, d_X, 1);
    }

    /* computes residual */
    cusparseDnMatSetValues(descr_X, d_X);
    cudaMemcpy(d_R, d_B, (sizeof *d_R)*n*blk, cudaMemcpyDeviceToDevice);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_X, &MINUS_ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, n*blk, d_R, 1, &R_norm);
    #if DEBUG == 1
      printf(">relative norm_frobenious_residual: %1.4E\n", R_norm/B_norm);
    #endif
    if ((R_norm/B_norm <= meta.tolerance) || (k == outer_iterations-1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V)*n*blk, cudaMemcpyDeviceToDevice);
      temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
      cublasZscal(context_gpu->cublas_handle, n*blk,
        (cuDoubleComplex *)&temp_complex, d_V, 1);
      inner_iterations = (CudaInt)meta.iterations;
      m_H = (CudaInt)meta.iterations;
      n_H = (CudaInt)meta.iterations;
    }
  }

  /* transfers solution to vectors_block_X */
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  MPInt n,
  KrylovMeta meta,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
)
{
  MPInt m_H = meta.blk*(meta.iterations+1);
  MPInt n_H = meta.blk*meta.iterations;
  if (data_type == MP_REAL)
  {
    *memory_bytes_cpu = sizeof(double)*
      (m_H*n_H  /* size_H */
      +m_H      /* size_br */
      +n_H*2);  /* size_temp_matrix */
    *memory_bytes_gpu = sizeof(double)*
      (n*m_H     /* size_V */
      +m_H);     /* size_br */
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      (m_H*n_H    /* size_H */
      +m_H        /* size_br */
      +n_H*2);    /* size_temp_matrix */
    *memory_bytes_gpu = sizeof(cuDoubleComplex)*
      (n*m_H    /* size_V */
      +n);      /* d_br*/
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      (m_H*n_H    /* size_H */
      +m_H        /* size_br */
      +n_H*2);    /* size_temp_matrix */

    *memory_bytes_gpu = sizeof(cuDoubleComplex)*
      (n*m_H    /* size_V */
      +n);      /* d_br*/
  }
}

void mp_cuda_block_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
)
{
  MPInt blk = meta.blk;
  MPInt iterations = meta.iterations;
  MPInt m_H = blk*(iterations+1);
  MPInt n_H = blk*(iterations);
  if (data_type == MP_REAL)
  {
    *memory_bytes_cpu = sizeof(double)*
      (blk        /* size_B_norms */
      +m_H*n_H    /* size_H */
      +m_H*blk    /* size_Br */
      +n*blk      /* size_Vtemp */
      +m_H*blk);  /* size_Htemp */

    *memory_bytes_gpu = sizeof(double)*
      (n*m_H*blk  /* size_V */
      +m_H*n_H    /* size_H */
      +m_H*blk);  /* size_Br */
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      (blk        /* size_B_norms */
      +m_H*n_H    /* size_H */
      +m_H*blk    /* size_Br */
      +n*blk      /* size_Vtemp */
      +m_H*blk);  /* size_Htemp */

    *memory_bytes_gpu = sizeof(MPComplexDouble)*
      (n*m_H*blk  /* size_V */
      +m_H*n_H    /* size_H */
      +m_H*blk);  /* size_Br */
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      (blk        /* size_B_norms */
      +m_H*n_H    /* size_H */
      +m_H*blk    /* size_Br */
      +n*blk      /* size_Vtemp */
      +m_H*blk);  /* size_Htemp */

    *memory_bytes_gpu = sizeof(MPComplexDouble)*
      (n*m_H*blk  /* size_V */
      +m_H*n_H    /* size_H */
      +m_H*blk);  /* size_Br */
  }
}

void mp_cuda_global_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  MPInt n,
  KrylovMeta meta,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
)
{
  MPInt iterations = meta.iterations;
  MPInt blk = meta.blk;
  if (data_type == MP_REAL)
  {
    *memory_bytes_cpu = sizeof(double)*
      ((iterations+1)*iterations   /* size_H */
      +iterations+1                /* size_br */
      +iterations*2);              /* size_temp */

    *memory_bytes_gpu = sizeof(double)*
      (n*blk*(iterations+1));  /* size_V */
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      ((iterations+1)*iterations  /* size_H */
      +iterations+1               /* size_br */
      +iterations*2);             /* size_temp */

    *memory_bytes_gpu = sizeof(cuDoubleComplex)*
      (n*blk*(iterations+1)); /* size_V */
  }
  else if ((data_type == MP_COMPLEX) && (structure_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes_cpu = sizeof(MPComplexDouble)*
      ((iterations+1)*iterations  /* size_H */
      +iterations+1               /* size_br */
      +iterations*2);             /* size_temp */

    *memory_bytes_gpu = sizeof(cuDoubleComplex)*
      (n*blk*(iterations+1)); /* size_V */
  }
}
