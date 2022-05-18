#include "mp.h"
#include "mp_cuda.h"
#include "mp_cuda_solve.h"
#include "mp_cuda_auxilliary.h"

void mp_cuda_dsy_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,

  const MPSparseCsr_Cuda A,

  double *d_b,
  double *d_x,
  double *memory,
  double *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  const double ONE_R = 1.0;
  const double MINUS_ONE_R = -1.0;
  const double ZERO_R = 0.0;
  /* solver context */
  MPInt k = 0;
  MPInt j = 0;
  double b_norm = 0.0;
  double r_norm = 0.0;
  double h_temp = 0.0;
  double temp_real = 1.0;
  /* meta */
  MPInt m_B = m_A;
  MPInt outer_iterations = meta.restarts+1;
  MPInt inner_iterations = meta.iterations;
  MPInt ld_H = meta.iterations;
  MPInt m_H = meta.iterations;
  MPInt n_H = meta.iterations;
  /* cpu memory */
  double *H = memory;
  double *br = &H[m_H*n_H];
  /* gpu memory */
  double *d_V = (double *) memory_cuda;
  double *d_r = &d_V[m_B*(inner_iterations)];
  double *d_br = &d_r[m_B];
  /* handles on gpu memory */
  double *d_w = &d_V[2*m_B];
  double *d_vprev = d_V;
  double *d_vcurr = &d_V[m_B];
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_v_prev;
  CusparseDenseVectorDescriptor descr_v_curr;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  /* initialize memory and descriptors */
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, H, m_H);
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnVec(&descr_b, m_B, d_b, CUDA_R_64F);
  cusparseCreateDnVec(&descr_x, m_B, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&descr_r, m_B, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&descr_w, m_B, d_w, CUDA_R_64F);
  cusparseCreateDnVec(&descr_v_prev, m_B, d_vprev, CUDA_R_64F);
  cusparseCreateDnVec(&descr_v_curr, m_B, d_vcurr, CUDA_R_64F);
  /* adds first krylov vector */
  cudaMemcpy(d_r, d_b, (sizeof *d_b)*m_B, cudaMemcpyDeviceToDevice);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_R, descr_A, descr_x, &ONE_R, descr_r, CUDA_R_64F,
    CUSPARSE_CSRMV_ALG1, NULL);
  cublasDnrm2(context_gpu->cublas_handle, m_B, d_b, 1, &b_norm);
  cublasDnrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);

  /* outer iterations */
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    temp_real = 1/r_norm;
    cublasDscal(context_gpu->cublas_handle, m_B, &temp_real, d_V, 1);
    mp_zeros_d_set(MP_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = r_norm;
    d_vprev = d_V;
    d_w = &d_V[m_B];
    cusparseDnVecSetValues(descr_v_prev, d_vprev);
    cusparseDnVecSetValues(descr_w, d_w);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_R, descr_A, descr_v_prev, &ZERO_R, descr_w, CUDA_R_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cublasDdot(context_gpu->cublas_handle, m_B, d_w, 1, d_vprev, 1, &H[0]);
    cudaDeviceSynchronize();

    temp_real = -H[0];
    cublasDaxpy(context_gpu->cublas_handle, m_B, &temp_real, d_vprev, 1,
      d_w, 1);
    cublasDnrm2(context_gpu->cublas_handle, m_B, d_w, 1, &h_temp);
    if (h_temp < 1e-12)
    {
      inner_iterations = 1;
      break;
    }
    H[1] = h_temp;

    temp_real = 1/h_temp;
    cublasDscal(context_gpu->cublas_handle, m_B, &temp_real, d_w, 1);
    for (j = 1; j < inner_iterations; ++j)
    {
      d_w = &d_V[m_B*(j+1)];
      d_vcurr = &d_V[m_B*j];
      d_vprev = &d_V[m_B*(j-1)];
      cusparseDnVecSetValues(descr_v_curr, d_vcurr);
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_v_prev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_R, descr_A, descr_v_curr, &ZERO_R, descr_w, CUDA_R_64F,
        CUSPARSE_CSRMV_ALG1, NULL);

      cublasDdot(context_gpu->cublas_handle, m_B, d_w, 1, d_vcurr, 1,
        &H[m_H*j+j]);
      cudaDeviceSynchronize();
      temp_real = -H[m_H*j+j];
      cublasDaxpy(context_gpu->cublas_handle, m_B, &temp_real, d_vcurr, 1, d_w,
        1);

      H[m_H*j+j-1] = H[m_H*(j-1)+j];
      temp_real = -H[m_H*j+j-1];
      cublasDaxpy(context_gpu->cublas_handle, m_B, &temp_real, d_vprev, 1, d_w,
        1);
      cublasDnrm2(context_gpu->cublas_handle, m_B, d_w, 1, &h_temp);
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      H[m_H*j+j+1] = h_temp;
      h_temp = 1/h_temp;
      cublasDscal(context_gpu->cublas_handle, m_B, &h_temp, d_w, 1);
    }

    /* solves linear system of equations and checks termination condition */
    mp_qr_dsy_givens(m_H, n_H, 1, H, ld_H, br);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, 1.0, H, n_H, br, m_H);
    cudaMemcpy(d_br, br, (sizeof *d_br), cudaMemcpyHostToDevice);
    cudaMemcpy(d_br, br, (sizeof *d_br)*n_H, cudaMemcpyHostToDevice);
    cublasDgemv(context_gpu->cublas_handle, CUBLAS_OP_N, m_B, n_H, &ONE_R, d_V,
      m_B, d_br, 1, &ONE_R, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*m_B, cudaMemcpyDeviceToDevice);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_R, descr_A, descr_x, &ONE_R, descr_r, CUDA_R_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cublasDnrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf("relative residual: %1.4E\n", r_norm/b_norm);
    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);
    }
  }
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnVec(descr_b);
  cusparseDestroyDnVec(descr_x);
  cusparseDestroyDnVec(descr_r);
  cusparseDestroyDnVec(descr_v_prev);
  cusparseDestroyDnVec(descr_v_curr);
  cusparseDestroyDnVec(descr_w);
}

void mp_cuda_zsy_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  cuDoubleComplex *d_b,
  cuDoubleComplex *d_x,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  MPInt k = 0;
  MPInt j = 0;
  double b_norm = 0.0;
  double r_norm = 0.0;
  double h_temp = 0.0;
  MPComplexDouble temp_complex = ONE_C_CPU;
  /* meta */
  MPInt m_B = m_A;
  MPInt outer_iterations = meta.restarts+1;
  MPInt inner_iterations = meta.iterations;
  MPInt ld_H = meta.iterations;
  MPInt m_H = meta.iterations;
  MPInt n_H = meta.iterations;
  /* cpu memory */
  MPComplexDouble *H = (MPComplexDouble *) memory;
  MPComplexDouble *br = &H[m_H*n_H];
  /* gpu memory */
  cuDoubleComplex *d_V = memory_cuda;
  cuDoubleComplex *d_r = &d_V[m_B*(inner_iterations)];
  cuDoubleComplex *d_br       = &d_r[m_B];
  /* handles on gpu memory */
  cuDoubleComplex *d_w = &d_V[2*m_B];
  cuDoubleComplex *d_vprev = d_V;
  cuDoubleComplex *d_vcurr = d_V + m_B;
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_v_prev;
  CusparseDenseVectorDescriptor descr_v_curr;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;
  /* initialize memory and descriptors */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, m_H);
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, m_B, d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, m_B, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r, m_B, d_r, CUDA_C_64F);
  cusparseCreateDnVec(&descr_w, m_B, d_w, CUDA_C_64F);
  cusparseCreateDnVec(&descr_v_prev, m_B, d_vprev, CUDA_C_64F);
  cusparseCreateDnVec(&descr_v_curr, m_B, d_vcurr, CUDA_C_64F);
  /* adds first krylov vector */
  cudaMemcpy(d_r, d_b, (sizeof *d_b)*m_B, cudaMemcpyDeviceToDevice);
  cusparseDnVecSetValues(descr_r, d_r);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r, CUDA_C_64F,
    CUSPARSE_CSRMV_ALG1, NULL);

  cudaDeviceSynchronize();
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_b, 1, &b_norm);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);

  /* outer iterrations */

  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, r_norm);
    cublasZscal(context_gpu->cublas_handle, m_B,
      (cuDoubleComplex*)&temp_complex, d_V, 1);
    mp_zeros_z_set(MP_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = mp_scalar_z_init(r_norm, 0.0);

    d_vprev = d_V;
    d_w = &d_V[m_B];
    cusparseDnVecSetValues(descr_v_prev, d_vprev);
    cusparseDnVecSetValues(descr_w, d_w);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_C, descr_A, descr_v_prev, &ZERO_C, descr_w, CUDA_C_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cublasZdotu(context_gpu->cublas_handle, m_B, d_w, 1, d_vprev, 1,
      (cuDoubleComplex*)&H[0]);
    cudaDeviceSynchronize();

    temp_complex = mp_scalar_z_invert_sign(H[0]);
    cublasZaxpy(context_gpu->cublas_handle, m_B,
      (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);
    cublasDznrm2(context_gpu->cublas_handle, m_B, d_w, 1, &h_temp);
    if (h_temp < 1e-12)
    {
      inner_iterations = 1;
      break;
    }
    H[1] = mp_scalar_z_init(h_temp, 0.0);

    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, h_temp);
    cublasZscal(context_gpu->cublas_handle, m_B,
      (cuDoubleComplex*)&temp_complex, d_w, 1);
    for (j = 1; j < inner_iterations; ++j)
    {
      d_w = &d_V[m_B*(j+1)];
      d_vcurr = &d_V[m_B*j];
      d_vprev = &d_V[m_B*(j-1)];
      cusparseDnVecSetValues(descr_v_curr, d_vcurr);
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_v_prev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_C, descr_A, descr_v_curr, &ZERO_C, descr_w, CUDA_C_64F,
        CUSPARSE_CSRMV_ALG1, NULL);

      cublasZdotu(context_gpu->cublas_handle, m_B, d_w, 1, d_vcurr, 1,
        (cuDoubleComplex*)&H[m_H*j+j]);
      cudaDeviceSynchronize();
      temp_complex = mp_scalar_z_invert_sign(H[ld_H*j+j]);
      cublasZaxpy(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_vcurr, 1, d_w, 1);

      H[ld_H*j+j-1] = H[ld_H*(j-1)+j];
      temp_complex = H[m_H*j+j-1];
      temp_complex = mp_scalar_z_invert_sign(temp_complex);
      cublasZaxpy(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);

      cublasZdotu(context_gpu->cublas_handle, m_B, d_w, 1, d_w, 1,
        (cuDoubleComplex*)&temp_complex);
      mp_vectorized_z_sqrt(1, &temp_complex, &temp_complex);
      mp_vectorized_z_abs(1, &temp_complex, &h_temp);
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      H[m_H*j+j+1] = temp_complex;
      temp_complex = mp_scalar_z_normalize(ONE_C_CPU, h_temp);
      cublasZscal(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_w, 1);
    }

    /* solves linear system of equations and checks termination condition */
    mp_qr_zsy_givens_2(H, br, ld_H, n_H, 1);
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, &ONE_C, H, n_H, br, m_H);
    cudaMemcpy(d_br, br, (sizeof *d_br)*n_H, cudaMemcpyHostToDevice);
    cublasZgemv(context_gpu->cublas_handle, CUBLAS_OP_N, m_B, n_H, &ONE_C, d_V,
      m_B, d_br, 1, &ONE_C, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*m_B, cudaMemcpyDeviceToDevice);

    cusparseDnVecSetValues(descr_x, d_x);
    cusparseDnVecSetValues(descr_r, d_r);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r, CUDA_C_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
    #if DEBUG == 1
      printf(">> relative residual: %1.4E\n", r_norm/b_norm);
    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k;
      break;
    }
    else
    {
      /* restart */
      cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnVec(descr_b);
  cusparseDestroyDnVec(descr_x);
  cusparseDestroyDnVec(descr_r);
  cusparseDestroyDnVec(descr_v_prev);
  cusparseDestroyDnVec(descr_v_curr);
  cusparseDestroyDnVec(descr_w);
}

void mp_cuda_zhe_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  cuDoubleComplex *d_b,
  cuDoubleComplex *d_x,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  MPInt k = 0;
  MPInt j = 0;
  double b_norm = 0.0;
  double r_norm = 0.0;
  double h_temp = 0.0;
  MPComplexDouble temp_complex = ONE_C_CPU;
  /* meta */
  MPInt m_B = m_A;
  MPInt outer_iterations = meta.restarts+1;
  MPInt inner_iterations = meta.iterations;
  MPInt ld_H = meta.iterations;
  MPInt m_H = meta.iterations;
  MPInt n_H = meta.iterations;
  /* cpu memory */
  MPComplexDouble *H = memory;
  MPComplexDouble *br = &H[m_H*n_H];
  /* gpu memory */
  cuDoubleComplex *d_V = memory_cuda;
  cuDoubleComplex *d_r = &d_V[m_B*(inner_iterations)];
  cuDoubleComplex *d_br = &d_r[m_B];
  /* handles on gpu memory */
  cuDoubleComplex *d_w = &d_V[2*m_B];
  cuDoubleComplex *d_vprev = d_V;
  cuDoubleComplex *d_vcurr = d_V + m_B;
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_v_prev;
  CusparseDenseVectorDescriptor descr_v_curr;
  CusparseDenseVectorDescriptor descr_w;
  CusparseDenseVectorDescriptor descr_r;

  /* initialize memory and descriptors */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, m_H);
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, m_B, d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, m_B, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r, m_B, d_r, CUDA_C_64F);
  cusparseCreateDnVec(&descr_w, m_B, d_w, CUDA_C_64F);
  cusparseCreateDnVec(&descr_v_prev, m_B, d_vprev, CUDA_C_64F);
  cusparseCreateDnVec(&descr_v_curr, m_B, d_vcurr, CUDA_C_64F);
  /* adds first krylov vector */
  cudaMemcpy(d_r, d_b, (sizeof *d_b)*m_B, cudaMemcpyDeviceToDevice);
  cusparseDnVecSetValues(descr_r, d_r);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r, CUDA_C_64F,
    CUSPARSE_CSRMV_ALG1, NULL);
  cudaDeviceSynchronize();

  cublasDznrm2(context_gpu->cublas_handle, m_B, d_b, 1, &b_norm);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
  cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);

  /* outer iterations */
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, ld_H);
  for (k = 0; k < outer_iterations; ++k)
  {
    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, r_norm);
    cublasZscal(context_gpu->cublas_handle, m_B,
     (cuDoubleComplex*)&temp_complex, d_V, 1);
    mp_zeros_z_set(MP_COL_MAJOR, m_H, 1, br, m_H);
    br[0] = mp_scalar_z_init(r_norm, 0.0);
    d_vprev = d_V;
    d_w = &d_V[m_B];
    cusparseDnVecSetValues(descr_v_prev, d_vprev);
    cusparseDnVecSetValues(descr_w, d_w);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_C, descr_A, descr_v_prev, &ZERO_C, descr_w, CUDA_C_64F,
       CUSPARSE_CSRMV_ALG1, NULL);
    cublasZdotc(context_gpu->cublas_handle, m_B, d_w, 1, d_vprev, 1,
      (cuDoubleComplex*)&H[0]);
    cudaDeviceSynchronize();
    temp_complex = mp_scalar_z_invert_sign(H[0]);
    cublasZaxpy(context_gpu->cublas_handle, m_B,
      (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);
    cublasDznrm2(context_gpu->cublas_handle, m_B, d_w, 1, &h_temp);
    if (h_temp < 1e-12)
    {
      inner_iterations = 1;
      break;
    }
    H[1] = mp_scalar_z_init(h_temp, 0.0);

    printf("\n\n >> last error: %d\n\n", cudaGetLastError());

    temp_complex = mp_scalar_z_normalize(ONE_C_CPU, h_temp);
    cublasZscal(context_gpu->cublas_handle, m_B,
      (cuDoubleComplex*)&temp_complex, d_w, 1);
    for (j = 1; j < inner_iterations; ++j)
    {

      d_w = &d_V[m_B*(j+1)];
      d_vcurr = &d_V[m_B*j];
      d_vprev = &d_V[m_B*(j-1)];
      cusparseDnVecSetValues(descr_v_curr, d_vcurr);
      cusparseDnVecSetValues(descr_w, d_w);
      cusparseDnVecSetValues(descr_v_prev, d_vprev);
      cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        &ONE_C, descr_A, descr_v_curr, &ZERO_C, descr_w, CUDA_C_64F,
        CUSPARSE_CSRMV_ALG1, NULL);

      cublasZdotc(context_gpu->cublas_handle, m_B, d_w, 1, d_vcurr, 1,
        (cuDoubleComplex*)&H[m_H*j+j]);
      cudaDeviceSynchronize();
      temp_complex = mp_scalar_z_invert_sign(H[ld_H*j+j]);
      cublasZaxpy(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_vcurr, 1, d_w, 1);

      H[ld_H*j+j-1] = H[ld_H*(j-1)+j];
      temp_complex = H[m_H*j+j-1];
      temp_complex = mp_scalar_z_invert_sign(temp_complex);
      cublasZaxpy(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_vprev, 1, d_w, 1);
      cublasDznrm2(context_gpu->cublas_handle, m_B, d_w, 1, &h_temp);
      if ((h_temp <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = inner_iterations;
        n_H = inner_iterations;
        break;
      }
      H[m_H*j+j+1] = mp_scalar_z_init(h_temp, 0.0);
      temp_complex = mp_scalar_z_normalize(ONE_C_CPU, h_temp);
      cublasZscal(context_gpu->cublas_handle, m_B,
        (cuDoubleComplex*)&temp_complex, d_w, 1);
    }

    /* solves linear system of equations and checks termination condition */
    mp_qr_zsy_givens_2(H, br, ld_H, n_H, 1);
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, 1, &ONE_C, H, n_H, br, m_H);
    cudaMemcpy(d_br, br, (sizeof *d_br)*n_H, cudaMemcpyHostToDevice);
    cublasZgemv(context_gpu->cublas_handle, CUBLAS_OP_N, m_B, n_H, &ONE_C, d_V,
      m_B, d_br, 1, &ONE_C, d_x, 1);
    cudaMemcpy(d_r, d_b, (sizeof *d_r)*m_B, cudaMemcpyDeviceToDevice);

    cusparseDnVecSetValues(descr_x, d_x);
    cusparseDnVecSetValues(descr_r, d_r);
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r, CUDA_C_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cublasDznrm2(context_gpu->cublas_handle, m_B, d_r, 1, &r_norm);
    cudaDeviceSynchronize();
    #if DEBUG == 1
      printf(">> relative residual: %1.4E\n", r_norm/b_norm);
    #endif
    if (r_norm/b_norm <= meta.tolerance)
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      /* restart */
      cudaMemcpy(d_V, d_r, (sizeof *d_V)*m_B, cudaMemcpyDeviceToDevice);
    }
  }
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnVec(descr_b);
  cusparseDestroyDnVec(descr_x);
  cusparseDestroyDnVec(descr_r);
  cusparseDestroyDnVec(descr_v_prev);
  cusparseDestroyDnVec(descr_v_curr);
  cusparseDestroyDnVec(descr_w);
}

void mp_cuda_dsy_block_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  double *d_B,
  double *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  const double ONE_R = 1.0;
  const double MINUS_ONE_R = -1.0;
  const double ZERO_R = 0.0;
  /* solver context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  double H_IJ_norm = 0.0;
  double r_norm_new = 0.0;
  double r_norm_max = 0.0;
  /* meta */
  CudaInt outer_iterations = 1+meta.restarts;
  CudaInt inner_iterations = meta.iterations;
  double tolerance = meta.tolerance;
  CudaInt m_B = m_A;
  CudaInt blk = meta.blk;
  CudaInt ld_H = meta.iterations*blk;
  CudaInt m_H = ld_H;
  CudaInt n_H = ld_H;
  /* host memory */
  double *B_norms_array = memory;
  double *H = &B_norms_array[blk];
  double *Br = &H[m_H*n_H];
  double *Vtemp = &Br[m_H*blk];
  double *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  double *Hblk = NULL;
  /* device memory */
  double *d_V = memory_cuda;
  double *d_Br = &d_V[m_B*blk*(meta.iterations+1)];
  /* handles on device memory */
  double *d_Vlast = &d_V[m_B*m_H];
  double *d_R = d_Vlast;
  double *d_Vprev = d_V;
  double *d_W = &d_V[m_B*blk];
  double *d_Hblk_temp = NULL;
  double *Hblk_dest = NULL;
  /* temporary memory (host and device respectively) */
  double *h_Hblk_temp = mp_malloc((sizeof *h_Hblk_temp)*blk*blk);
  cudaMalloc((void **) &d_Hblk_temp, (sizeof *d_Hblk_temp)*blk*blk);
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  CusparseDenseMatrixDescriptor descr_r;
  /* initialize descriptor */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_r, m_B, blk, m_B, d_R, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_W, m_B, blk, m_B, d_W, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_Vprev, m_B, blk, m_B, d_Vprev,
    CUSPARSE_ORDER_COL, CUDA_R_64F);
  /* initialize krylov method */
  cudaMemcpy(d_R, d_B, (sizeof *d_R) * m_B * blk, cudaMemcpyDeviceToDevice);
  mp_zeros_d_set(MP_COL_MAJOR, m_H, n_H, H, m_H);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A, descr_X, &ONE_R,
    descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
  for (i = 0; i < blk; ++i)
  {
    cublasDnrm2(context_gpu->cublas_handle, m_B, &d_B[m_B*i], 1,
      &B_norms_array[i]);
    cublasDnrm2(context_gpu->cublas_handle, m_B, &d_R[m_B*i], 1, &r_norm_new);
    r_norm_new = r_norm_new/B_norms_array[i];
    if (r_norm_new > r_norm_max)
    {
      r_norm_max = r_norm_new;
    }
  }
  #if DEBUG == 1
    printf("[first] max relative residual: %1.4E\n", r_norm_max);
  #endif
  if (r_norm_max <= meta.tolerance)
  {
    return;
  }

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_d_set(MP_COL_MAJOR, m_H, blk, Br, m_H);
    cudaMemcpy(Vtemp, d_R, (sizeof *Vtemp)*m_B*blk, cudaMemcpyDeviceToHost);
    mp_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, Vtemp, m_B, reflectors_array);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m_B, blk, Vtemp, m_B, Br, ld_H);
    mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Vtemp, m_B, reflectors_array);
    cudaMemcpy(d_V, Vtemp, (sizeof *d_V)*m_B*blk, cudaMemcpyHostToDevice);

    d_Vprev = d_V;
    d_W = &d_V[(m_B*blk)];
    cusparseDnMatSetValues(descr_Vprev, d_Vprev);
    cusparseDnMatSetValues(descr_W, d_W);

    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_Vprev, &ZERO_R,
      descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);

    /* computation of H(1, 1) block */
    Hblk = H;
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
      blk, m_B, &ONE_R, d_Vprev, m_B, d_W, m_B, &ZERO_R, d_Hblk_temp, blk);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &MINUS_ONE_R, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_R, d_W, m_B);
    cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp) * m_B * blk, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Hblk_temp, d_Hblk_temp,
      (sizeof *h_Hblk_temp)*blk*blk, cudaMemcpyDeviceToHost);
    mp_domatcopy('C', 'N', blk, blk, 1.0, h_Hblk_temp, blk, Hblk, ld_H);
    cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp) * m_B * blk, cudaMemcpyDeviceToHost);

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, Vtemp, m_B, reflectors_array);
    Hblk = &H[blk];
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, Vtemp, m_B, Hblk, ld_H);
    mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Vtemp, m_B, reflectors_array);
    cudaMemcpy(d_W, Vtemp, (sizeof *d_W) * m_B * blk, cudaMemcpyHostToDevice);
    H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, m_H);

    /* inner iterations */
    for (j = 1; j < inner_iterations-1; ++j)
    {
      /* configure Hblk(j, i) */
      Hblk = &H[ld_H*blk*j + blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      d_Vprev = &d_V[m_B*blk*j];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);

      /* update W */
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_Vprev, &ZERO_R,
        descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
        blk, m_B, &ONE_R, d_Vprev, m_B, d_W, m_B, &ZERO_R, d_Hblk_temp, blk);
      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
        blk, blk, &MINUS_ONE_R, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_R, d_W,
        m_B);
      cudaMemcpy(h_Hblk_temp, d_Hblk_temp, (sizeof *h_Hblk_temp)*blk*blk,
        cudaMemcpyDeviceToHost);
      mp_domatcopy('C', 'N', blk, blk, 1.0, h_Hblk_temp, blk, Hblk, ld_H);

      Hblk = &H[ld_H*blk*(j-1) + blk*j];
      mp_domatcopy('C', 'N', blk, blk, 1.0, Hblk, m_H, h_Hblk_temp, blk);
      cudaMemcpy(d_Hblk_temp, h_Hblk_temp, (sizeof *d_Hblk_temp)*blk*blk,
        cudaMemcpyHostToDevice);

      d_Vprev = &d_V[m_B*blk*(j-1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m_B,
        blk, blk, &MINUS_ONE_R, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_R,
        d_W, m_B);

      Hblk = &H[m_H*blk*(j-1) + blk*j];
      Hblk_dest = &H[m_H*blk*j + blk*(j-1)];
      mp_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[ld_H*blk*j + blk*(j+1)];
      cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*m_B*blk, cudaMemcpyDeviceToHost);
      mp_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, Vtemp, m_B, reflectors_array);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, Vtemp, m_B, Hblk, ld_H);

      H_IJ_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk*(inner_iterations+1);
        n_H = blk*inner_iterations;
        break;
      }
      mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Vtemp, m_B, reflectors_array);
      cudaMemcpy(d_W, Vtemp, (sizeof *d_W) * m_B * blk, cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
    if (H_IJ_norm > 1e-12)
    {
      j = inner_iterations-1;
      Hblk = &H[ld_H*blk*j + blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      d_Vprev = &d_V[(m_B * blk)*j];

      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_Vprev, &ZERO_R,
        descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);

      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
        blk, m_B, &ONE_R, d_Vprev, m_B, d_W, m_B, &ZERO_R, d_Hblk_temp, blk);
      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
        blk, blk, &MINUS_ONE_R, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_R,
        d_W, m_B);
      cudaMemcpy(h_Hblk_temp, d_Hblk_temp, (sizeof *h_Hblk_temp)*blk*blk,
        cudaMemcpyDeviceToHost);
      mp_domatcopy('C', 'N', blk, blk, 1.0, h_Hblk_temp, blk, Hblk, ld_H);

      Hblk = &H[ld_H*blk*(j-1) + blk*j];
      d_Vprev = &d_V[m_B*blk*(j-1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m_B,
        blk, blk, &MINUS_ONE_R, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_R, d_W,
        m_B);

      Hblk = &H[ld_H*blk*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mp_domatcopy ('C', 'T', blk, blk, 1.0, Hblk, ld_H, Hblk_dest, ld_H);
    }

    /* solves system of equations and evaluates termination criteria */
    cudaDeviceSynchronize();
    mp_block_qr_dsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, 1.0, H, ld_H, Br, ld_H);
    cudaMemcpy(d_Br, Br, (sizeof *d_Br)*ld_H*blk, cudaMemcpyHostToDevice);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      n_H, &ONE_R, d_V, m_B, d_Br, ld_H, &ONE_R, d_X, m_B);

    cudaMemcpy(d_R, d_B, (sizeof *d_R)*m_B*blk, cudaMemcpyDeviceToDevice);
    cusparseDnMatSetValues(descr_r, d_R);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A, descr_X, &ONE_R,
      descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);

    r_norm_max = 0.0;
    for (i = 0; i < blk; ++i)
    {
      cublasDnrm2(context_gpu->cublas_handle, m_B, &d_R[m_B*i], 1, &r_norm_new);
      r_norm_new = r_norm_new/B_norms_array[i];;
      if (r_norm_new > r_norm_max)
      {
          r_norm_max = r_norm_new;
      }
    }
    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_max, k);
    #endif
    if ((r_norm_max <= tolerance) || (k == meta.restarts))
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      //cudaMemcpy(Vtemp, d_R, (sizeof *Vtemp) * m_B * blk, cudaMemcpyDeviceToHost);
      //mp_dgeqrf(LAPACK_COL_MAJOR, blk, blk, Vtemp, m_B, reflectors_array);
      //mp_zeros_d_set(MP_COL_MAJOR, m_H, blk, Br, m_H);
      //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, Vtemp, m_B, Br, ld_H);
      //mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Vtemp, m_B, reflectors_array);
      //cudaMemcpy(d_V, Vtemp, (sizeof *d_V) * m_B * blk, cudaMemcpyHostToDevice);
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_Vprev);
  cusparseDestroyDnMat(descr_W);
  mp_free(reflectors_array);
  mp_free(h_Hblk_temp);
  cudaFree(d_Hblk_temp);
}

void mp_cuda_zsy_block_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt n_A,
  const MPInt nz_A,
  const MPSparseDescr A_descr,
  const MPSparseCsr_Cuda A,
  cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  const cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  const cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  double H_IJ_norm = 0.0;
  double r_norm = 0.0;
  double r_norm_max = 0.0;
  /* meta */
  CudaInt outer_iterations = 1+meta.restarts;
  CudaInt inner_iterations = meta.iterations;
  CudaInt m_B = n_A;
  CudaInt blk = meta.blk;
  CudaInt ld_H = meta.iterations*blk;
  CudaInt m_H = ld_H;
  CudaInt n_H = ld_H;
  /* host memory */
  double *B_norms_array = memory;
  MPComplexDouble *H = &((MPComplexDouble *)B_norms_array)[blk];
  MPComplexDouble *Br = &H[m_H*n_H];
  MPComplexDouble *Vtemp = &Br[m_H*blk+blk];
  MPComplexDouble *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  MPComplexDouble *Hblk = NULL;
  /* device memory */
  cuDoubleComplex *d_V  = memory_cuda;
  cuDoubleComplex *d_Br = &d_V[m_B*blk*(meta.iterations+1)];
  /* handles on device memory */
  cuDoubleComplex *d_Vlast = &d_V[m_B*m_H];
  cuDoubleComplex *d_R = d_Vlast;
  cuDoubleComplex *d_Vprev = d_V;
  cuDoubleComplex *d_W = &d_V[m_B*blk];
  cuDoubleComplex *d_Hblk_temp = NULL;
  MPComplexDouble *Hblk_dest = NULL;
  /* temporary memory (host and device respectively) */
  MPComplexDouble *h_Hblk_temp = mp_malloc((sizeof *h_Hblk_temp)*blk*blk);
  cudaMalloc((void **) &d_Hblk_temp, (sizeof *d_Hblk_temp)*blk*blk);
  /* cuda descriptors */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_Vprev;
  CusparseDenseMatrixDescriptor descr_W;
  CusparseDenseMatrixDescriptor descr_r;
  /* initialize descriptor */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, m_B, blk, m_B, d_R, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, m_B, blk, m_B, d_W, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, m_B, blk, m_B, d_Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  /* initialize krylov method */
  cudaMemcpy(d_R, d_B, (sizeof *d_R)*m_B*blk, cudaMemcpyDeviceToDevice);
  mp_zeros_z_set(MP_COL_MAJOR, m_H, n_H, H, m_H);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
    descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  for (i = 0; i < blk; ++i)
  {
    cublasDznrm2(context_gpu->cublas_handle, m_B, &d_B[m_B*i], 1,
      &B_norms_array[i]);
    cublasDznrm2(context_gpu->cublas_handle, m_B, &d_R[m_B*i], 1, &r_norm);
    r_norm = r_norm/B_norms_array[i];
    if (r_norm > r_norm_max)
    {
        r_norm_max = r_norm;
    }
  }
  #if DEBUG == 1
    printf("[first] max relative residual: %1.4E\n", r_norm_max);
  #endif
  if (r_norm_max <= meta.tolerance)
  {
    printf("RETURNING\n");
    return;
  }

  /* outer-loop (restarts) */

  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, m_H, blk, Br, m_H);
    cudaMemcpy(Vtemp, d_R, (sizeof *Vtemp) * m_B * blk, cudaMemcpyDeviceToHost);
    mp_gram_schmidt_zge(m_B, blk, Vtemp, Br, m_H);
    cudaMemcpy(d_V, Vtemp, (sizeof *d_V)*m_B*blk, cudaMemcpyHostToDevice);

    d_Vprev = d_V;
    d_W = &d_V[(m_B*blk)];
    cusparseDnMatSetValues(descr_Vprev, d_Vprev);
    cusparseDnMatSetValues(descr_W, d_W);

    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
      descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);

    /* computation of H(1, 1) block */
    Hblk = H;
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_C, d_Vprev, m_B, d_W, m_B, &ZERO_C, d_Hblk_temp, blk);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &MINUS_ONE_C, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_C, d_W, m_B);
    cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*m_B*blk, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Hblk_temp, d_Hblk_temp, (sizeof *h_Hblk_temp)*blk*blk,
      cudaMemcpyDeviceToHost);
    mp_zomatcopy('C', 'N', blk, blk, ONE_C_CPU, h_Hblk_temp, blk, Hblk, ld_H);

    cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*m_B*blk, cudaMemcpyDeviceToHost);
    Hblk = &H[blk];
    mp_gram_schmidt_zge(m_B, blk, Vtemp, Hblk, m_H);
    cudaMemcpy(d_W, Vtemp, (sizeof *d_W)*m_B*blk, cudaMemcpyHostToDevice);
    H_IJ_norm = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, m_H);

    /* inner iterations */
    for (j = 1; j < inner_iterations-1; ++j)
    {
      /* configure Hblk(j, i) */
      Hblk = &H[ld_H*blk*j + blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      d_Vprev = &d_V[m_B*blk*j];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);

      /* update W */
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
        blk, m_B, &ONE_C, d_Vprev, m_B, d_W, m_B, &ZERO_C, d_Hblk_temp, blk);
      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
        blk, blk, &MINUS_ONE_C, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_C, d_W,
        m_B);
      cudaMemcpy(h_Hblk_temp, d_Hblk_temp, (sizeof *h_Hblk_temp)*blk*blk,
        cudaMemcpyDeviceToHost);
      mp_zomatcopy('C', 'N', blk, blk, ONE_C_CPU, h_Hblk_temp, blk, Hblk, ld_H);

      Hblk = &H[ld_H*blk*(j-1)+blk*j];
      mp_zomatcopy('C', 'N', blk, blk, ONE_C_CPU, Hblk, m_H, h_Hblk_temp, blk);
      cudaMemcpy(d_Hblk_temp, h_Hblk_temp, (sizeof *d_Hblk_temp)*blk*blk,
        cudaMemcpyHostToDevice);

      d_Vprev = &d_V[m_B*blk*(j-1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m_B,
        blk, blk, &MINUS_ONE_C, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_C, d_W,
        m_B);

      Hblk = &H[m_H*blk*(j-1) + blk*j];
      Hblk_dest = &H[m_H*blk*j + blk*(j-1)];
      mp_zomatcopy ('C', 'T', blk, blk, ONE_C_CPU, Hblk, ld_H, Hblk_dest, ld_H);

      Hblk = &H[ld_H*blk*j + blk*(j+1)];
      cudaMemcpy(Vtemp, d_W, (sizeof *Vtemp)*m_B*blk, cudaMemcpyDeviceToHost);
      mp_gram_schmidt_zge(m_B, blk, Vtemp, Hblk, m_H);

      H_IJ_norm = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', blk, blk, Hblk, ld_H);
      if (H_IJ_norm <= 1e-12)
      {
        inner_iterations = j;
        m_H = blk*(j+1);
        n_H = blk*j;
        break;
      }
      cudaMemcpy(d_W, Vtemp, (sizeof *d_W) * m_B * blk, cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
    if (H_IJ_norm > 1e-12)
    {
      j = inner_iterations-1;
      Hblk = &H[ld_H*blk*j + blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      d_Vprev = &d_V[(m_B * blk)*j];

      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);

      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
        blk, m_B, &ONE_C, d_Vprev, m_B, d_W, m_B, &ZERO_C, d_Hblk_temp, blk);
      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
        blk, blk, &MINUS_ONE_C, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_C, d_W,
        m_B);
      cudaMemcpy(h_Hblk_temp, d_Hblk_temp, (sizeof *h_Hblk_temp)*blk*blk,
        cudaMemcpyDeviceToHost);
      mp_zomatcopy('C', 'N', blk, blk, ONE_C_CPU, h_Hblk_temp, blk, Hblk, ld_H);

      Hblk = &H[ld_H*blk*(j-1) + blk*j];
      d_Vprev = &d_V[m_B*blk*(j-1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m_B,
        blk, blk, &MINUS_ONE_C, d_Vprev, m_B, d_Hblk_temp, blk, &ONE_C, d_W,
        m_B);

      Hblk = &H[ld_H*blk*(j-1) + blk*j];
      Hblk_dest = &H[(ld_H * blk)*j + blk*(j-1)];
      mp_zomatcopy('C', 'T', blk, blk, ONE_C_CPU, Hblk, ld_H,
        (MPComplexDouble*)Hblk_dest, ld_H);
    }

    /* solves system of equations and evaluates termination criteria */
    mp_block_qr_zsy_givens(n_H, blk, blk, H, ld_H, Br, ld_H);
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      m_H, blk, &ONE_C, H, ld_H, Br, ld_H);

    cudaMemcpy(d_Br, Br, (sizeof *d_Br)*ld_H*blk, cudaMemcpyHostToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      n_H, &ONE_C, d_V, m_B, d_Br, ld_H, &ONE_C, d_X, m_B);

    cudaMemcpy(d_R, d_B, (sizeof *d_R) * m_B * blk, cudaMemcpyDeviceToDevice);
    cusparseDnMatSetValues(descr_r, d_R);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);

    r_norm_max = 0.0;
    for (i = 0; i < blk; ++i)
    {
      cublasDznrm2(context_gpu->cublas_handle, m_B,
       (cuDoubleComplex*)&d_R[m_B*i], 1, &r_norm);
      r_norm = r_norm/B_norms_array[i];;
      if (r_norm > r_norm_max)
      {
        r_norm_max = r_norm;
      }
    }
    #if DEBUG == 1
      printf("max_relative residual: %1.4E -- (restart %d)\n", r_norm_max, k);
    #endif
    if ((r_norm_max <= meta.tolerance) || (k == meta.restarts))
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      //cudaMemcpy(Vtemp, d_R, (sizeof *Vtemp) * m_B * blk, cudaMemcpyDeviceToHost);
      //mp_dgeqrf(LAPACK_COL_MAJOR, blk, blk, Vtemp, m_B, reflectors_array);
      //mp_zeros_d_set(MP_COL_MAJOR, m_H, blk, Br, m_H);
      //LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', blk, blk, Vtemp, m_B, Br, ld_H);
      //mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Vtemp, m_B, reflectors_array);
      //cudaMemcpy(d_V, Vtemp, (sizeof *d_V) * m_B * blk, cudaMemcpyHostToDevice);
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_Vprev);
  cusparseDestroyDnMat(descr_W);
  mp_free(reflectors_array);
  mp_free(h_Hblk_temp);
  cudaFree(d_Hblk_temp);
}

void mp_cuda_dsy_global_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  double *d_B,
  double *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constant */
  const double ONE_C = 1.0;
  const double MINUS_ONE_C = -1.0;
  const double ZERO_R = 0.0;
  /* context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  CudaInt z = 0;
  size_t size_buffer = 0;
  double B_norm = 0.0;
  double R_norm = 0.0;
  double temp_real = 0.0;
  double trace = 0.0;
  double temp_trace = 0.0;
  double Hlast = 0.0;
  /* meta */
  CudaInt inner_iterations = meta.iterations;
  CudaInt outer_iterations = meta.restarts+1;
  CudaInt ld_H = inner_iterations;
  CudaInt m_H = ld_H;
  CudaInt n_H = ld_H;
  CudaInt blk = meta.blk;
  CudaInt m_B = m_A;
  /* host memory */
  double *H = memory;
  double *br = &H[m_H*n_H];
  /* device memory */
  double *d_R = memory_cuda;
  double *d_V = &d_R[m_B*blk];
  /* handes on device memory */
  double *d_Vprev = d_V;
  double *d_W = &d_V[m_B*blk];
  /* setup descriptors */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_Vprev;
  MPCusparseDenseMatrixDescriptor descr_W;
  MPCusparseDenseMatrixDescriptor descr_r;
  /* initialize descriptor */
  cusparseCreateCsr(&descr_A, m_A, m_A, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_r, m_B, blk, m_B, d_R, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_W, m_B, blk, m_B, d_W, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_Vprev, m_B, blk, m_B, d_Vprev, CUSPARSE_ORDER_COL,
    CUDA_R_64F);

  /* first iteration */
  cudaMemcpy(d_V, d_B, (sizeof *d_V)*m_B*blk, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_Vprev, d_B, (sizeof *d_Vprev)*m_B*blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C,
    descr_A, descr_X, &ONE_C, descr_Vprev, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
    &size_buffer);
  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc(d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_Vprev, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_Vprev, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, NULL);
  }

  cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, &R_norm);
  temp_real = 1/R_norm;
  cublasDscal(context_gpu->cublas_handle, m_B*blk, &temp_real, d_Vprev, 1);
  if (R_norm/B_norm <= meta.tolerance)    /* checks terminating condition */
  {
    return;
  }

  /* outer iterations (restarts)*/
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_d_set(MP_COL_MAJOR, ld_H, ld_H, H, ld_H);
    mp_zeros_d_set(MP_COL_MAJOR, ld_H, 1, br, ld_H);
    br[0] = R_norm;

    /* computes new block krylov component */
    d_W = &d_V[m_B*blk];
    d_Vprev = d_V;
    cusparseDnMatSetValues(descr_W, d_W);
    cusparseDnMatSetValues(descr_Vprev, d_Vprev);

    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C,
      descr_A, descr_Vprev, &ZERO_R, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
      &size_buffer);

    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_R,
        descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_R,
        descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    /* computes H(1, 1) */
    trace = 0.0;
    temp_trace = 0.0;
    for (z = 0; z < blk; ++z)
    {
      cublasDdot(context_gpu->cublas_handle, m_B, &d_Vprev[m_B*z], 1,
        &d_W[m_B*z], 1, &temp_trace);
      cudaDeviceSynchronize();
      trace += temp_trace;
      cudaDeviceSynchronize();
    }
    H[0] = trace;

    temp_real = -trace;
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &temp_real, d_Vprev, 1,
      d_W, 1);
    cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_W, 1, &temp_real);
    if (inner_iterations > 1)
    {
      H[1] = temp_real;
    }

    temp_real = 1/temp_real;
    cublasDscal(context_gpu->cublas_handle, m_B*blk, &temp_real, d_W, 1);

    for (j = 1; j < inner_iterations; ++j)
    {
      /* computes new block krylov component */
      d_Vprev = &d_V[m_B*blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);

      cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_Vprev, &ZERO_R, descr_W, CUDA_R_64F,
        CUSPARSE_CSRMM_ALG1, &size_buffer);
      if (size_buffer > 0)
      {
        void *d_external_buffer = NULL;
        cudaMalloc(d_external_buffer, size_buffer);
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_R, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
        cudaFree(d_external_buffer);
      }
      else
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_R, descr_W, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      /* computes H(j-1, j), H(j, j) and H(j+1, j) */
      trace = 0.0;
      for (z = 0; z < blk; ++z)
      {
        cublasDdot(context_gpu->cublas_handle, m_B, &d_Vprev[m_B*z], 1,
          &d_W[m_B*z], 1, &temp_trace);
        cudaDeviceSynchronize();
        trace += temp_trace;
        cudaDeviceSynchronize();
      }
      H[ld_H*j+j] = trace;
      temp_real = -trace;
      cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &temp_real, d_Vprev, 1,
        d_W, 1);
      d_Vprev = &d_V[(m_B*blk)*(j-1)];
      temp_real = - H[ld_H*(j-1)+j];
      cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &temp_real, d_Vprev, 1,
        d_W, 1);
      H[ld_H*j+j-1] = H[ld_H*(j-1)+j];
      cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_W, 1, &Hlast);

      /* checks termination condition */
      if ((Hlast <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      else
      {
        H[m_H*j+j+1] = Hlast;
        temp_real = 1/H[ld_H*j+j+1];
        cublasDscal(context_gpu->cublas_handle, m_B*blk, &temp_real, d_W, 1);
      }
    }

    /* solves system of equation */
    mp_qr_dsy_givens(m_H, n_H, 1, H, ld_H, br);
    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, 1.0, H, ld_H, br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_V[m_B*blk*i];
      cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &br[i], d_W, 1, d_X, 1);
    }

    /* computes residual */
    cudaMemcpy(d_R, d_B, (sizeof *d_R) * m_B*blk, cudaMemcpyDeviceToDevice);
    cusparseDnMatSetValues(descr_X, d_X);
    cusparseDnMatSetValues(descr_r, d_R);
    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
        &ONE_C, descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
        &ONE_C, descr_r, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }
    cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_R, 1, &R_norm);
    #if DEBUG == 1
      printf("relative R_norm: %1.4E\n", R_norm/B_norm);
    #endif
    if ((R_norm/B_norm <= meta.tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V)*m_B*blk, cudaMemcpyDeviceToDevice);
      temp_real = 1/R_norm;
      cublasDscal(context_gpu->cublas_handle, m_B*blk, &temp_real, d_V, 1);
      inner_iterations = meta.iterations;
      m_H = ld_H;
      n_H = ld_H;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_zsy_global_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt n_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constant */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble ZERO_C_CPU = mp_scalar_z_init(0.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  double B_norm = 0.0;
  MPComplexDouble R_norm = ZERO_C_CPU;
  double temp_real = 0.0;
  MPComplexDouble temp_complex = ZERO_C_CPU;
  double r_norm_abs = 0.0;
  MPComplexDouble trace = ZERO_C_CPU;
  MPComplexDouble Hlast = ZERO_C_CPU;
  size_t size_buffer;
  /* meta */
  CudaInt inner_iterations = meta.iterations;
  CudaInt outer_iterations = meta.restarts+1;
  CudaInt ld_H = meta.iterations;
  CudaInt m_H = ld_H;
  CudaInt n_H = ld_H;
  CudaInt blk = meta.blk;
  CudaInt m_B = n_A;
  /* host memory */
  MPComplexDouble *H = memory;
  MPComplexDouble *br = &H[m_H*n_H];
  /* device memory */
  cuDoubleComplex *d_R = memory_cuda;
  cuDoubleComplex *d_V = &d_R[m_B*blk];
  /* handes on device memory */
  cuDoubleComplex *d_Vprev = d_V;
  cuDoubleComplex *d_W = &d_V[m_B*blk];
  /* setup descriptors */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_Vprev;
  MPCusparseDenseMatrixDescriptor descr_W;
  MPCusparseDenseMatrixDescriptor descr_r;
  /* initialize descriptor */
  cusparseCreateCsr(&descr_A, n_A, n_A, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, m_B, blk, m_B, d_R, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, m_B, blk, m_B, d_W, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, m_B, blk, m_B, d_Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  /* first iteration */
  cudaMemcpy(d_V, d_B, (sizeof *d_V)*m_B*blk, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_Vprev, d_B, (sizeof *d_Vprev)*m_B*blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_Vprev, CUDA_C_64F,
    CUSPARSE_CSRMM_ALG1, &size_buffer);
  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc(d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_Vprev, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_Vprev, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  }
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, d_Vprev, 1,
    (cuDoubleComplex*)&temp_complex);
  mp_vectorized_z_sqrt(1, &temp_complex, &R_norm);
  mp_vectorized_z_abs(1, &R_norm, &r_norm_abs);
  #if DEBUG == 1
    printf("B_norm: %1.4E\n", B_norm);
    printf("R_norm: %1.4E\n", r_norm_abs);
  #endif
  temp_complex = mp_scalar_z_divide(ONE_C_CPU, R_norm);
  cublasZscal(context_gpu->cublas_handle, m_B*blk,
    (cuDoubleComplex*)&temp_complex, d_Vprev, 1);
  if (r_norm_abs/B_norm <= meta.tolerance)
  {
    return;
  }

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, ld_H, H, ld_H);
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, 1, br, ld_H);
    br[0] = R_norm;

    /* computes new block krylov component */
    d_W = &d_V[m_B*blk];
    d_Vprev = d_V;
    cusparseDnMatSetValues(descr_W, d_W);
    cusparseDnMatSetValues(descr_Vprev, d_Vprev);

    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C,
      descr_A, descr_Vprev, &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
      &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    /* computes H(1, 1) */
    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, d_W, 1,
      (cuDoubleComplex*)&trace);
    H[0] = trace;

    temp_complex = mp_scalar_z_invert_sign(trace);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
      (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);
    cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_W, 1,
      &temp_complex.real);
    temp_complex.imag = 0.0;

    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_W, 1, d_W, 1,
      (cuDoubleComplex*)&temp_complex);
    mp_vectorized_z_sqrt(1, &temp_complex, &Hlast);
    if (inner_iterations > 1)
    {
      H[1] = Hlast;
    }
    temp_complex = mp_scalar_z_divide(ONE_C_CPU, Hlast);
    cublasZscal(context_gpu->cublas_handle, m_B*blk,
      (cuDoubleComplex*)&temp_complex, d_W, 1);

    for (j = 1; j < inner_iterations; ++j)
    {
      /* computes new block krylov component */
      d_Vprev = &d_V[m_B*blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);

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
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
        cudaFree(d_external_buffer);
      }
      else
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      /* computes H(j-1, j), H(j, j) and H(j+1, j) */
      cudaDeviceSynchronize();
      cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, d_W, 1,
        (cuDoubleComplex*)&trace);
      cudaDeviceSynchronize();
      H[ld_H*j+j] = trace;
      temp_complex = mp_scalar_z_invert_sign(trace);
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);
      d_Vprev = &d_V[(m_B*blk)*(j-1)];
      temp_complex = mp_scalar_z_invert_sign(H[ld_H*(j-1)+j]);
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);

      H[ld_H*j+j-1] = H[ld_H*(j-1)+j];
      cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_W, 1, d_W, 1,
        (cuDoubleComplex*)&temp_complex);
      mp_vectorized_z_sqrt(1, &temp_complex, &Hlast);
      mp_vectorized_z_abs(1, &Hlast, &temp_real);

      /* checks termination condition */
      if ((temp_real <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      else
      {
        H[m_H*j+j+1] = Hlast;
        temp_complex = mp_scalar_z_divide(ONE_C_CPU, H[ld_H*j+j+1]);
        cublasZscal(context_gpu->cublas_handle, m_B*blk,
          (cuDoubleComplex*)&temp_complex, d_W, 1);
      }
    }

    #if DEBUG == 1
        //mp_matrix_dense_print(H, 4, 4, ld_H);
    #endif

    /* solves system of equation */
    mp_qr_zsy_givens_2(H, br, ld_H, n_H, 1);  // use this in order to avoid problems when ld_H != m_H
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_V[m_B*blk*i];
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&br[i], d_W, 1, d_X, 1);
    }

    /* computes residual */
    cudaMemcpy(d_R, d_B, (sizeof *d_R) * m_B*blk, cudaMemcpyDeviceToDevice);
    cusparseDnMatSetValues(descr_X, d_X);
    cusparseDnMatSetValues(descr_r, d_R);
    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r, CUDA_C_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
        descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
        descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_R, 1, d_R, 1,
      (cuDoubleComplex*)&R_norm);
    mp_vectorized_z_sqrt(1, &R_norm, &temp_complex);
    mp_vectorized_z_abs(1, &temp_complex, &r_norm_abs);

    #if DEBUG == 1
      printf("relative r_norm: %1.4E\n", r_norm_abs/B_norm);
    #endif
    if ((r_norm_abs/B_norm <= meta.tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V) * m_B * blk, cudaMemcpyDeviceToDevice);
      temp_complex = mp_scalar_z_divide(ONE_C_CPU, R_norm);
      cublasZscal(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_V, 1);
      inner_iterations = meta.iterations;
      m_H = ld_H;
      n_H = ld_H;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_zhe_global_lanczos
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  cuDoubleComplex *d_B,
  cuDoubleComplex *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constant */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble ZERO_C_CPU = mp_scalar_z_init(0.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* context */
  CudaInt i = 0;
  CudaInt j = 0;
  CudaInt k = 0;
  size_t size_buffer = 0;
  double B_norm = 0.0;
  double R_norm = 0.0;
  double temp_real = 0.0;
  MPComplexDouble temp_complex = ZERO_C_CPU;
  MPComplexDouble trace = ZERO_C_CPU;
  MPComplexDouble Hlast = ZERO_C_CPU;
  /* meta */
  CudaInt inner_iterations = meta.iterations;
  CudaInt outer_iterations = meta.restarts+1;
  CudaInt ld_H = inner_iterations;
  CudaInt m_H = ld_H;
  CudaInt n_H = ld_H;
  CudaInt blk = meta.blk;
  CudaInt m_B = m_A;
  /* host memory */
  MPComplexDouble *H = memory;
  MPComplexDouble *br = &H[m_H*n_H];
  /* device memory */
  cuDoubleComplex *d_R = memory_cuda;
  cuDoubleComplex *d_V = &d_R[m_B*blk];
  /* handes on device memory */
  cuDoubleComplex *d_Vprev = d_V;
  cuDoubleComplex *d_W     = &d_V[m_B*blk];
  /* setup descriptors */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_Vprev;
  MPCusparseDenseMatrixDescriptor descr_W;
  MPCusparseDenseMatrixDescriptor descr_r;
  /* initialize descriptor */
  cusparseCreateCsr(&descr_A, m_A, m_A, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r, m_B, blk, m_B, d_R, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_W, m_B, blk, m_B, d_W, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_Vprev, m_B, blk, m_B, d_Vprev, CUDA_C_64F,
    CUSPARSE_ORDER_COL);

  /* first iteration */
  cudaMemcpy(d_V, d_B, (sizeof *d_V) * m_B * blk, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_R, d_B, (sizeof *d_Vprev) * m_B * blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C,
     descr_A, descr_X, &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
     &size_buffer);

  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc(d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X, &ONE_C,
      descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  }

  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_R, 1, &R_norm);
  #if DEBUG == 1
    //printf("B_norm: %1.4E\n", B_norm);
    //printf("R_norm: %1.4E\n", R_norm);
  #endif
  temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
  cublasZscal(context_gpu->cublas_handle, m_B*blk,
    (cuDoubleComplex*)&temp_complex, d_R, 1);
  if (R_norm/B_norm <= meta.tolerance)    /* checks terminating condition */
  {
    return;
  }
  cudaMemcpy(d_Vprev, d_R, (sizeof *d_Vprev)*m_B*blk, cudaMemcpyDeviceToDevice);

  /* outer iterations (restarts) */
  for (k = 0; k < outer_iterations; ++k)
  {
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, ld_H, H, ld_H);
    mp_zeros_z_set(MP_COL_MAJOR, ld_H, 1, br, ld_H);
    br[0] = mp_scalar_z_init(R_norm, 0.0);

    /* computes new block krylov component */
    d_W = &d_V[m_B*blk];
    d_Vprev = d_V;
    cusparseDnMatSetValues(descr_W, d_W);
    cusparseDnMatSetValues(descr_Vprev, d_Vprev);

    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C,
      descr_A, descr_Vprev, &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
      &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev, &ZERO_C,
        descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    /* computes H(1, 1) */
    cublasZdotc(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, d_W, 1,
      (cuDoubleComplex*)&trace);
    H[0] = trace;

    temp_complex = mp_scalar_z_invert_sign(trace);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
      (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);
    cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_W, 1, &temp_real);
    Hlast = mp_scalar_z_init(temp_real, 0.0);
    if (inner_iterations > 1)
    {
      H[1] = Hlast;
    }
    temp_complex = mp_scalar_z_divide(ONE_C_CPU, Hlast);
    cublasZscal(context_gpu->cublas_handle, m_B*blk,
      (cuDoubleComplex*)&temp_complex, d_W, 1);

    for (j = 1; j < inner_iterations; ++j)
    {
      /* computes new block krylov component */
      d_Vprev = &d_V[m_B*blk*j];
      d_W = &d_V[m_B*blk*(j+1)];
      cusparseDnMatSetValues(descr_Vprev, d_Vprev);
      cusparseDnMatSetValues(descr_W, d_W);

      cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C,
        descr_A, descr_Vprev, &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
        &size_buffer);
      if (size_buffer > 0)
      {
        void *d_external_buffer = NULL;
        cudaMalloc(d_external_buffer, size_buffer);
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
        cudaFree(d_external_buffer);
      }
      else
      {
        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_Vprev,
          &ZERO_C, descr_W, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
      }

      /* computes H(j-1, j), H(j, j) and H(j+1, j) */
      cudaDeviceSynchronize();
      cublasZdotc(context_gpu->cublas_handle, m_B*blk, d_Vprev, 1, d_W, 1,
        (cuDoubleComplex*)&trace);
      cudaDeviceSynchronize();
      H[ld_H*j+j] = trace;
      temp_complex = mp_scalar_z_invert_sign(trace);
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);
      d_Vprev = &d_V[(m_B*blk)*(j-1)];
      temp_complex = mp_scalar_z_invert_sign(H[ld_H*(j-1)+j]);
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_Vprev, 1, d_W, 1);

      H[ld_H*j+j-1] = H[ld_H*(j-1)+j];
      cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_W, 1, &temp_real);
      Hlast = mp_scalar_z_init(temp_real, 0.0);
      /* checks termination condition */
      if ((temp_real <= 1e-12) || (j == inner_iterations-1))
      {
        inner_iterations = j+1;
        m_H = j;
        n_H = j;
        break;
      }
      else
      {
        H[m_H*j+j+1] = Hlast;
        temp_complex = mp_scalar_z_divide(ONE_C_CPU, H[ld_H*j+j+1]);
        cublasZscal(context_gpu->cublas_handle, m_B*blk,
          (cuDoubleComplex*)&temp_complex, d_W, 1);
      }
    }

    /* solves system of equation */
    mp_qr_zsy_givens_2(H, br, ld_H, n_H, 1);  // use this in order to avoid problems when ld_H != m_H
    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      n_H, 1, &ONE_C, H, ld_H, br, ld_H);
    for (i = 0; i < n_H; ++i)
    {
      d_W = &d_V[m_B*blk*i];
      cublasZaxpy(context_gpu->cublas_handle, m_B*blk, (cuDoubleComplex*)&br[i],
        d_W, 1, d_X, 1);
    }

    /* computes residual */
    cudaMemcpy(d_R, d_B, (sizeof *d_R) * m_B*blk, cudaMemcpyDeviceToDevice);
    cusparseDnMatSetValues(descr_X, d_X);
    cusparseDnMatSetValues(descr_r, d_R);
    cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r, CUDA_C_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);

    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
        &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
        &ONE_C, descr_r, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    cublasDznrm2(context_gpu->cublas_handle, m_B, d_R, 1, &R_norm);
    #if DEBUG == 1
      printf("relative r_norm/b_norm: %1.4E\n", R_norm/B_norm);
    #endif
    if ((R_norm/B_norm <= meta.tolerance) || (k == outer_iterations - 1))
    {
      outer_iterations = k+1;
      break;
    }
    else
    {
      cudaMemcpy(d_V, d_R, (sizeof *d_V) * m_B * blk, cudaMemcpyDeviceToDevice);
      temp_complex = mp_scalar_z_normalize(ONE_C_CPU, R_norm);
      cublasZscal(context_gpu->cublas_handle, m_B*blk,
        (cuDoubleComplex*)&temp_complex, d_V, 1);
      m_H = ld_H;
      n_H = ld_H;
      inner_iterations = meta.iterations;
    }
  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r);
  cusparseDestroyDnMat(descr_W);
  cusparseDestroyDnMat(descr_Vprev);
}

void mp_cuda_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt iterations = meta.iterations;
  MPInt m_B = n;

  if (data_type == MP_REAL)
  {
    MPInt iterations = meta.iterations;
    *memory_bytes = sizeof(double)*
      (iterations*iterations   /* size_H */
      +iterations);            /* size_br */

    *memory_cuda_bytes = sizeof(double)*
      (n*(iterations+1)
      +iterations+1);           /* size_V */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes = sizeof(MPComplexDouble)*
      (m_B*(iterations+1)       /* size_V */
      +iterations*iterations    /* size_H */
      +iterations               /* size_br */
      +m_B);                    /* size_residual;*/

    *memory_cuda_bytes = 0;
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes = sizeof(MPComplexDouble)*
      (m_B*(iterations+1)        /* size_V */
      +iterations*iterations     /* size_H */
      +iterations                /* size_br */
      +m_B);                     /* size_residual;*/

    *memory_cuda_bytes = 0;
  }
}

void mp_cuda_block_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt iterations = meta.iterations;
  MPInt blk = meta.blk;
  MPInt m_H = iterations*blk;
  MPInt n_H = iterations*blk;

  if (data_type == MP_REAL)
  {
    // may be required + size_Hblk
    *memory_bytes = sizeof(double) * 
      (blk         /* size_Bnorms */
      +m_H*n_H     /* size_H */
      +m_H*blk     /* size_Br */
      +n*blk       /* size_Vtemp */
      +blk*blk);   /* size_Htemp */
    //more may be needed (size_Hblk)

    *memory_cuda_bytes = sizeof(double)*
      (n*m_H      /* size_V */
      +m_H*blk);   /* size_Br */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    // may be required + size_Hblk
    *memory_bytes = sizeof(MPComplexDouble)*
      (blk         /* size_Bnorms */
      +m_H*n_H     /* size_H */
      +m_H*blk     /* size_Br */
      +n*blk       /* size_Vtemp */
      +blk*blk);   /* size_Htemp */

    //more may be needed (size_Hblk)
    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (n*m_H       /* size_V */
      +blk*blk     /* size_Htemp */
      +m_H*blk);   /* size_Br */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    // may be required + size_Hblk
    *memory_bytes = sizeof(MPComplexDouble) *
      (blk         /* size_Bnorms */
      +m_H*n_H     /* size_H */
      +m_H*blk     /* size_Br */
      +n*blk       /* size_Vtemp */
      +blk*blk);   /* size_Htemp */

    //more may be needed (size_Hblk)
    *memory_cuda_bytes = sizeof(cuDoubleComplex) *
      (n*m_H       /* size_V */
      +blk*blk     /* size_Htemp */
      +m_H*blk);   /* size_Br */
  }
}


void mp_cuda_global_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt iterations = meta.iterations;
  MPInt blk = meta.blk;

  if (data_type == MP_REAL)
  {
    // memory_aligment problem
    *memory_bytes = sizeof(double)*
      (iterations*iterations   /* size_H */
      +iterations);            /* size_br */

    *memory_cuda_bytes = sizeof(double)*
      (n*blk*(iterations+1)    /* size_V */
      +n*blk);                 /* size_residual */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes = sizeof(MPComplexDouble)*
       (iterations*iterations  /* size_H */
       +iterations);           /* size_br */

    *memory_cuda_bytes = sizeof(MPComplexDouble)*
      (n*blk*(iterations+1)    /* size_V */
      +n*blk);                 /* size_residual */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes = sizeof(MPComplexDouble)*
      (iterations*iterations   /* size_H */
      +iterations);            /* size_br */

    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (n*blk*(iterations+1)   /* size_V */
      +n*blk);                /* size_residual */
  }
}
