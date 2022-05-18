#include "mp.h"
#include "mp_cuda.h"
#include "mp_cuda_solve.h"
#include "mp_cuda_auxilliary.h"

/* ----------------------------- real solvers ------------------------------- */

void mp_cuda_dsy_cg
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
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  const double ZERO_R = 0.0;
  const double ONE_R = 1.0;
  const double MINUS_ONE_R = 1.0;
  /* solver context */
  MPInt i = 0;
  double norm_b = 0.0;
  double r_norm = 0.0;
  double alpha = 0.0;
  double beta  = 0.0;
  double temp_real = 1.0;
  MPInt m_B = m_A;
  /* memory cpu */
  double *d_r_new = memory_cuda;
  double *d_r_old = &d_r_new[m_B];
  double *d_dvec = &d_r_old[m_B];
  double *d_temp_vector = NULL;
  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_r_old;
  CusparseDenseVectorDescriptor descr_r_new;
  CusparseDenseVectorDescriptor descr_direction;
  /* initialize descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnVec(&descr_b, m_B, (void *) d_b, CUDA_R_64F);
  cusparseCreateDnVec(&descr_x, m_B, (void *) d_x, CUDA_R_64F);
  cusparseCreateDnVec(&descr_r_old, m_B, d_r_old, CUDA_R_64F);
  cusparseCreateDnVec(&descr_r_new, m_B, d_r_new, CUDA_R_64F);
  cusparseCreateDnVec(&descr_direction, m_B, d_dvec, CUDA_R_64F);

  /* cg initialization */
  cudaMemcpy(d_r_old, d_b, (sizeof *d_r_old)*m_B, cudaMemcpyDeviceToDevice);
  cublasDnrm2(context_gpu->cublas_handle, m_B, d_b, 1, &norm_b);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_R, descr_A, descr_x, &ONE_R, descr_r_old,
    CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
  cublasDnrm2(context_gpu->cublas_handle, m_B, d_r_old, 1, &r_norm);
  cudaMemcpy(d_dvec, d_r_old, (sizeof *d_r_old)*m_B, cudaMemcpyDeviceToDevice);
  #if DEBUG == 1
    printf("[init] relative residual: %1.4E\n", r_norm/norm_b);
  #endif

  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm/norm_b > meta.tolerance))
  {
    cudaDeviceSynchronize();
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_R, descr_A, descr_direction, &ZERO_R, descr_r_new,
      CUDA_R_64F, CUSPARSE_CSRMV_ALG1, NULL);
    cudaDeviceSynchronize();

    cublasDdot(context_gpu->cublas_handle, m_B, d_r_new, 1, d_dvec, 1, &alpha);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();

    cublasDdot(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_real);
    cudaDeviceSynchronize();
    alpha = temp_real/alpha;
    cublasDaxpy(context_gpu->cublas_handle, m_B, &alpha, d_dvec, 1, d_x, 1);

    temp_real = -alpha;
    cublasDscal(context_gpu->cublas_handle, m_B, &temp_real, d_r_new, 1);
    cudaDeviceSynchronize();
    cublasDaxpy(context_gpu->cublas_handle, m_B, &ONE_R, d_r_old, 1, d_r_new,
      1);

    cudaDeviceSynchronize();
    cublasDdot(context_gpu->cublas_handle, m_B, d_r_new, 1, d_r_new, 1, &beta);
    cudaDeviceSynchronize();
    cublasDdot(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_real);
    cudaDeviceSynchronize();
    beta = beta/temp_real;
    cublasDscal(context_gpu->cublas_handle, m_B, &beta, d_dvec, 1);
    cublasDaxpy(context_gpu->cublas_handle, m_B, &ONE_R, d_r_new, 1, d_dvec,
      1);

    cublasDnrm2(context_gpu->cublas_handle, m_B, d_r_new, 1, &r_norm);
    cudaDeviceSynchronize();

    d_temp_vector = d_r_old;
    d_r_old = d_r_new;
    d_r_new = d_temp_vector;
    cusparseDnVecSetValues(descr_r_old, d_r_old);
    cusparseDnVecSetValues(descr_r_new, d_r_new);

    i += 1;
  }
  #if DEBUG == 1
    //printf("relative residual: %1.4E\n", r_norm / norm_b);
    //printf("iterations completed: %d\n", i);
    printf("r_norm: %1.4E\n", r_norm/norm_b);
  #endif

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnVec(descr_b);
  cusparseDestroyDnVec(descr_x);
  cusparseDestroyDnVec(descr_r_old);
  cusparseDestroyDnVec(descr_r_new);
  cusparseDestroyDnVec(descr_direction);
  d_r_new = NULL;
  d_r_old = NULL;
  d_temp_vector = NULL;
}

/*====================*/
/*== cuda functions ==*/
/*====================*/

//void mp_solver_cuda_dsy_global_cg_call(MPContext *context, MPMatrixSparseHandle A_handle, const double *d_B, double *d_X)
//{
//    /* constants */
//    MPContextGpuCuda *context_gpu = context->context_gpu;
//    const double ONE_R = 1.0;
//    const double MINUS_ONE_R = -1.0;
//    const double ZERO_R = 0.0;
//    /* --- initialization --- */
//    MPInt i = 0;
//    MPInt j = 0;
//    MPInt m_B = context->rhs_num_rows;
//    MPInt blk = context->solver_inner_blk;
//    double alpha = 0.0;
//    double beta = 0.0;
//    double gamma = 0.0;
//    double B_norm = 0.0;
//    double r_norm = 1.0;
//    double trace_r_old = 0.0;
//    double trace_r_new = 0.0;
//    MPLayout layout = layout;
//    MPLayoutSparse sparse_layout;
//    /* memory cpu*/
//    double *d_Rold = (double *) context->solver_inner_memory_gpu;
//    double *d_Rnew = &d_Rold[m_B*blk];
//    double *d_Dvec = &d_Rnew[m_B*blk];
//    double *d_Dvec_temp = &d_Dvec[m_B*blk];
//    double *d_residual_temp = NULL;
//    /* assign descrs */
//    CusparseCsrMatrixDescriptor descr_A = context_gpu->sparse_deoscriptor_array[0];
//    CusparseDenseVectorDescriptor descr_B = context_gpu->dense_descr_array[0];
//    CusparseDenseVectorDescriptor descr_X = context_gpu->dense_descr_array[1];
//    CusparseDenseVectorDescriptor descr_r_old = context_gpu->dense_descr_array[2];
//    CusparseDenseVectorDescriptor descr_r_new = context_gpu->dense_descr_array[3];
//    CusparseDenseVectorDescriptor descr_r_temp = context_gpu->dense_descr_array[4];
//    CusparseDenseVectorDescriptor descr_direction = context_gpu->dense_descr_array[5];
//    CusparseDenseVectorDescriptor descr_dvec_temp = context_gpu->dense_descr_array[6];
//
//    cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
//    /* first iteration */
//    mp_convert_layout_to_sparse(layout, &sparse_layout);
//    cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold) * m_B * blk, cudaMemcpyDeviceToDevice);
//    //mp_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A_handle, A_descr, sparse_layout, X,
//    //               blk, m_B, 1.0, Rold, m_B);
//    // (1) initialize descrs before call once only for each thread
//    // (2) also perhaps set the multiplication algorithm beforehand
//    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R,
//                 descr_A, descr_X, &ONE_R, descr_r_old, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
//
//    cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, &r_norm);
//    //r_norm = mp_dlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rold, m_B)/B_norm;
//    cudaMemcpy(d_Dvec, d_Rold, (sizeof *d_Dvec)*m_B*blk, cudaMemcpyDeviceToDevice);
//    /* --- main loop --- */
//    while ((i < context->solver_inner_num_iterations) && (r_norm > context->solver_inner_tolerance))
//    {
//        /* computes alpha and gamma scalars */
//        cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R,
//                     descr_A, descr_X, &ONE_R, descr_r_new, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
//        trace_r_old = 0.0;
//        gamma = 0.0;
//        for (j = 0; j < blk; j++)
//        {
//            cublasDdot(context_gpu->cublas_handle, m_B, &d_Dvec_temp[m_B*j], 1,
//                       &d_Dvec[m_B*j], 1, &gamma);
//            cublasDdot(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
//                       &d_Rold[m_B*j], 1, &trace_r_old);
//        }
//        alpha = trace_r_old/gamma;
//        /* updates X (solution) and residual vecblk */
//        //cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &trace_r_old, d_Vprev_vecblk, 1, d_W_vecblk, 1);  /* W <-- W - V(:, J)*H(j, j)) */
//        cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew), cudaMemcpyDeviceToDevice);
//        cublasDaxpy(context_gpu->cublas_handle, m_B * blk, &alpha, d_Dvec_temp, 1, d_Rnew, 1);  /* W <-- W - V(:, J)*H(j, j)) */
//        trace_r_new = 0.0;
//        for (j = 0; j < blk; j++)
//        {
//            cublasDdot(context_gpu->cublas_handle, m_B, &d_Rnew[m_B*j], 1,
//                       &d_Rnew[m_B*j], 1, &trace_r_new);
//        }
//        beta = trace_r_new/trace_r_old;
//        /* updates direction vecblk */
//        cublasDscal(context_gpu->cublas_handle, m_B * blk, &beta, d_Dvec, 1);
//        cublasDaxpy(context_gpu->cublas_handle, m_B * blk, &alpha, d_Rnew, 1, d_Dvec, 1);  /* W <-- W - V(:, J)*H(j, j)) */
//        cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, &r_norm);
//        /* swaps old with new residual vecblks */
//        d_residual_temp = d_Rnew;
//        d_Rnew = d_Rold;
//        d_Rold = d_residual_temp;
//        i++;
//    }
//    if (context->solver_outer_collect_meta)
//    {
//        //context->solver_outer_meta_iterations_array[context->solver_inner_current_block] = i;
//        //context->solver_outer_meta_residual_array[context->solver_inner_current_block] = r_norm;
//    }
//    #if DEBUG == 1
//        if (context->printout == MP_PRINTOUT_ON)
//        {
//            //printf("iterations: %d/%d, r_norm: %1.4E\n", i, context->solver_inner_num_iterations, r_norm);
//        }
//    #endif
//}


void mp_cuda_dsy_block_cg
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
  const double MINUS_ONE_R = -1.0;
  const double ONE_R = 1.0;
  const double ZERO_R = 0.0;
  /* solver context */
  CudaInt i = 0;
  CudaInt m_B = m_A;
  CudaInt blk = meta.blk;
  double B_norm = 0.0;
  double R_norm = 0.0;
  size_t size_buffer;
  /* memory cpu */
  double *R = memory;
  double *alpha_matrix = &R[m_B*blk];
  double *beta_matrix = &alpha_matrix[blk*blk];
  double *zeta_matrix = &beta_matrix[blk*blk];
  double *Dvec = &zeta_matrix[blk*blk];
  /* memory gpu */
  double *d_Rold = memory_cuda;
  double *d_Rnew = &d_Rold[m_B*blk];
  double *d_Dvec = &d_Rnew[m_B*blk];
  double *d_Dvec_new = &d_Dvec[m_B*blk];
  double *d_Dvec_temp = &d_Dvec_new[m_B*blk];
  double *d_R = &d_Dvec_temp[m_B*blk];
  double *d_alpha_matrix = &d_R[m_B*blk];
  double *d_beta_matrix = &d_alpha_matrix[blk*blk];
  double *d_zeta_matrix = &d_beta_matrix[blk*blk];
  double *d_temp_vecblk = NULL;

  //@OPTIMIZE: should be allocated beforehand
  double *reflectors_array = mp_malloc((sizeof *reflectors_array) * blk);
  MPInt *pivots_array = mp_malloc((sizeof *pivots_array) * blk);

  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_r_old;
  CusparseDenseMatrixDescriptor descr_r_new;
  CusparseDenseMatrixDescriptor descr_direction;
  CusparseDenseMatrixDescriptor descr_direction_new;
  CusparseDenseMatrixDescriptor descr_dvec_temp;

  /* initializes cuda descrs */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, (void *) d_B, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, (void *) d_X, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction_new, m_B, blk, m_B, d_Dvec_new,
    CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUDA_R_64F, CUSPARSE_ORDER_COL);

  /* krylov initialization */
  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &MINUS_ONE_R, descr_A, descr_X, &ONE_R, descr_r_old, CUDA_R_64F,
    CUSPARSE_CSRMM_ALG1, &size_buffer);

  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc((void **) &d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A, descr_X, &ONE_R,
      descr_r_old, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A, descr_X, &ONE_R,
      descr_r_old, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
  }
  cublasDnrm2(context_gpu->cublas_handle, m_B * blk, d_B, 1, &B_norm);
  cublasDnrm2(context_gpu->cublas_handle, m_B * blk, d_Rold, 1, &R_norm);
  #if DEBUG == 1
    printf("B_norm: %1.4E\n", B_norm);
    printf("R_norm: %1.4E\n", R_norm);
  #endif
  cudaMemcpy(R, d_Rold, (sizeof *R)*m_B*blk, cudaMemcpyDeviceToHost);
  mp_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, R, m_B, reflectors_array);
  cudaMemcpy(d_R, R, (sizeof *d_R)*m_B*blk, cudaMemcpyHostToDevice);
  memcpy(Dvec, R, (sizeof *Dvec)*m_B*blk);
  mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Dvec, m_B, reflectors_array);
  cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec)*m_B*blk, cudaMemcpyHostToDevice);

  /* main loop */
  while ((i < meta.iterations) && (R_norm/B_norm > meta.tolerance))
  {
    /* computes alpha_matrix */
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R,
      descr_A, descr_direction, &ZERO_R, descr_direction_new, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_direction,
        &ZERO_R, descr_direction_new, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
        d_external_buffer); cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_direction,
        &ZERO_R, descr_direction_new, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_R, d_Dvec, m_B, d_Dvec_new, m_B, &ZERO_R, d_beta_matrix, blk);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_R, d_Rold, m_B, d_Rold, m_B, &ZERO_R, d_alpha_matrix, blk);
    cudaMemcpy(d_zeta_matrix, d_alpha_matrix, (sizeof *d_zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(alpha_matrix, d_alpha_matrix, (sizeof *alpha_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
      blk, blk, 1.0, R, m_B, alpha_matrix, blk);
    cudaMemcpy(beta_matrix, d_beta_matrix, (sizeof *beta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);
    LAPACKE_dgesv(CblasColMajor, blk, blk, beta_matrix, blk, pivots_array,
      alpha_matrix, blk);

    /* updates solution X and residual */
    cudaMemcpy(d_alpha_matrix, alpha_matrix, (sizeof *d_alpha_matrix)*blk*blk,
      cudaMemcpyHostToDevice);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &ONE_R, d_Dvec, m_B, d_alpha_matrix, blk, &ONE_R, d_X, m_B);

    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &ONE_R, d_Dvec, m_B, d_alpha_matrix, blk, &ZERO_R, d_Dvec_temp, m_B);

    cusparseDnMatSetValues(descr_dvec_temp, d_Dvec_temp);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_R, descr_A, descr_dvec_temp, &ONE_R, descr_r_new,
      CUDA_R_64F, CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &MINUS_ONE_R, descr_A,
        descr_dvec_temp, &ONE_R, descr_r_new, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
        d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A,
        descr_dvec_temp, &ONE_R, descr_r_new, CUDA_R_64F, CUSPARSE_CSRMM_ALG1,
        NULL);
    }

    /* computes beta parameters */
    mp_dimatcopy('C', 'T', blk, blk, 1.0, R, m_B, blk);
    cudaMemcpy(zeta_matrix, d_zeta_matrix, (sizeof *zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    mp_dgesv(CblasColMajor, blk, blk, zeta_matrix, blk, pivots_array, R, blk);
    mp_dimatcopy('C', 'T', blk, blk, 1.0, R, blk, m_B);
    cudaMemcpy(d_R, R, (sizeof *d_R)*m_B*blk, cudaMemcpyHostToDevice);

    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_R, d_Rnew, m_B, d_Rnew, m_B, &ZERO_R, d_beta_matrix, blk);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, blk, blk,
     blk, &ONE_R, d_R, m_B, d_beta_matrix, blk, &ZERO_R, d_beta_matrix, blk);

    /* compute zeta (reorthogonalize) */
    cudaMemcpy(d_Rold, d_Rnew, (sizeof *d_Rold)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasDgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &ONE_R, d_Dvec, m_B, d_beta_matrix, blk, &ONE_R, d_Rold, m_B);
    cudaMemcpy(d_Dvec, d_Rold, (sizeof *Dvec)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(Dvec, d_Rold, (sizeof *Dvec)*m_B*blk, cudaMemcpyDeviceToHost);
    mp_dgeqrf(LAPACK_COL_MAJOR, m_B, blk, Dvec, m_B, reflectors_array);
    memcpy(R, Dvec, (sizeof *R)*m_B*blk);
    mp_dorgqr(LAPACK_COL_MAJOR, m_B, blk, blk, Dvec, m_B, reflectors_array);
    cublasDnrm2(context_gpu->cublas_handle, m_B * blk, d_Rnew, 1, &R_norm);
    cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec)*m_B*blk, cudaMemcpyHostToDevice);
    d_temp_vecblk = d_Rold;
    d_Rold = d_Rnew;
    d_Rnew = d_temp_vecblk;
    i += 1;
  }
  #if DEBUG == 1
    printf("R_norm: %1.4E\n", R_norm/B_norm);
    printf("iterations: %d/%d\n", i, meta.iterations);
  #endif

  /* memory deallocation */
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r_old);
  cusparseDestroyDnMat(descr_r_new);
  cusparseDestroyDnMat(descr_direction);
  cusparseDestroyDnMat(descr_direction_new);
  cusparseDestroyDnMat(descr_dvec_temp);
  mp_free(reflectors_array);  //@OPTIMIZE
  mp_free(pivots_array);      //@OPTIMIZE
}

void mp_cuda_dsy_global_cg
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
  double MINUS_ONE_R = -1.0;
  double ONE_R = 1.0;
  double ZERO_R = 0.0;
  /* solver context */
  MPInt i = 0;
  MPInt m_B = m_A;
  MPInt blk = meta.blk;
  double B_norm = 0.0;
  double r_norm = 1.0;
  double alpha, beta, gamma;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;
  /* memory cpu */
  double *d_Rold = memory;
  double *d_Rnew = &d_Rold[m_B*blk];
  double *d_Dvec = &d_Rnew[m_B*blk];
  double *d_Dvec_temp = &d_Dvec[m_B*blk];
  double *d_residual_temp = NULL;
  /* descrs */
  CusparseCsrMatrixDescriptor descr_A;
  //MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_r_old;
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;
  /* Initialize cuda descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, (void*)d_B, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, (void*)d_X, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_R_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp, CUDA_R_64F,
    CUSPARSE_ORDER_COL); // dont use this

  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_r_old, d_Rold);
  cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cudaDeviceSynchronize();
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_R, descr_A, descr_X, &ONE_R,
    descr_r_old, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
  cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, &r_norm);

  cudaMemcpy(d_Dvec, d_Rold, (sizeof *d_Dvec)*m_B*blk,
    cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_direction, d_Dvec);
  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm/B_norm > meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_direction,
      &ZERO_R, descr_dvec_temp, CUDA_R_64F, CUSPARSE_CSRMM_ALG1, NULL);
    cudaDeviceSynchronize();
    cublasDdot(context_gpu->cublas_handle, m_B*blk, d_Dvec_temp, 1, d_Dvec, 1,
      &gamma);
    cudaDeviceSynchronize();
    cublasDdot(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, d_Rold, 1,
      &trace_r_old);
    cudaDeviceSynchronize();
    alpha = trace_r_old/gamma;

    /* updates X and residual vecblk */
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    alpha = -alpha;
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);

    /* computes beta */
    cudaDeviceSynchronize();
    cublasDdot(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, d_Rnew, 1,
      &trace_r_new);
    beta = trace_r_new/trace_r_old;

    /* update direction vecblk */
    cublasDscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &ONE_R, d_Rnew, 1,
      d_Dvec, 1);
    cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, &r_norm);
    r_norm = r_norm/B_norm;

    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
    cusparseDnMatSetValues(descr_r_old, d_Rold);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    i += 1;
  }
}

void mp_cuda_dsy_global_cg_2
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
  double ONE_R = 1.0;
  double ZERO_R = 0.0;
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  double temp_trace = 0.0;
  MPInt m_B = m_A;
  MPInt blk = meta.blk;
  double B_norm = 0.0;
  double r_norm = 1.0;
  double alpha, beta, gamma;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;
  /* memory cpu */
  double *d_Rold = memory_cuda;
  double *d_Rnew = &d_Rold[m_B*blk];
  double *d_Dvec = &d_Rnew[m_B*blk];
  double *d_Dvec_temp = &d_Dvec[m_B*blk];
  double *d_residual_temp = NULL;

  /* descrs */
  MPCusparseCsrMatrixDescriptor *descr_A = &context_gpu->sparse_descriptors[0];
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;
  cusparseCreateCsr(descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUSPARSE_ORDER_COL,
    CUDA_R_64F);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec,
    CUSPARSE_ORDER_COL, CUDA_R_64F);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUSPARSE_ORDER_COL, CUDA_R_64F);
  cublasDnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);

  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm > meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &ONE_R, *descr_A, descr_direction, &ZERO_R, descr_dvec_temp, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, NULL);

    for (j = 0; j < blk; ++j)
    {
      cublasDdot(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
        &d_Rold[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      gamma += temp_trace;
      cudaDeviceSynchronize();
    }
    alpha = trace_r_old/gamma;
    /* updates X and residual vecblk */
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk, cudaMemcpyDeviceToDevice);
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);
    /* computes beta */
    trace_r_new = 0.0;
    for (j = 0; j < blk; j++)
    {
      cublasDdot(context_gpu->cublas_handle, m_B, &d_Rnew[m_B*j], 1,
        &d_Rnew[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      trace_r_new += temp_trace;
      cudaDeviceSynchronize();
    }
    beta = trace_r_new/trace_r_old;
    /* update direction vecblk */
    cublasDscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasDaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Rnew, 1,
      d_Dvec, 1);
    cublasDnrm2(context_gpu->cublas_handle, m_B, d_Rnew, 1, &r_norm);
    r_norm = r_norm/B_norm;
    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
  }
  printf("iterations: %d/%d, r_norm: %1.4E\n", i, meta.iterations, r_norm);
}

void mp_cuda_ssy_global_cg
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt m_A,
  const MPInt nz_A,

  const MPSparseCsr_Cuda A,

  float *d_B,
  float *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  float ONE_R = 1.0;
  float ZERO_R = 0.0;
  float temp_trace = 0.0;
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  MPInt m_B = m_A;
  MPInt blk = meta.blk;
  float B_norm = 0.0;
  float r_norm = 1.0;
  float alpha, beta, gamma;
  float trace_r_old = 0.0;
  float trace_r_new = 0.0;
  /* memory cpu */
  float *d_Rold = memory;
  float *d_Rnew = &d_Rold[m_B*blk];
  float *d_residual_temp = &d_Rnew[m_B*blk];
  float *d_Dvec = &d_Rnew[m_B*blk];
  float *d_Dvec_temp = NULL;
  /* cuda descrs */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_r_old;
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_r_temp;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;
  /* Initializes descrs */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers,
    A.d_cols, A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_temp, m_B, blk, m_B, d_residual_temp,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cublasSnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);

  /* main loop */
  cublasSnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  while ((i < meta.iterations) && (r_norm < meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_R, descr_A, descr_direction,
      &ZERO_R, descr_r_new, CUDA_R_32F, CUSPARSE_CSRMM_ALG1, NULL);
    trace_r_old = 0.0;
    gamma = 0.0;
    for (j = 0; j < blk; ++j)
    {
      cublasSdot(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
        &d_Rold[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      gamma += temp_trace;
      cudaDeviceSynchronize();
    }
    alpha = trace_r_old/gamma;

    /* updates X and residual vecblk */
    cublasSaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasSaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);

    /* computes beta */
    trace_r_new = 0.0;
    for (j = 0; j < blk; ++j)
    {
      cublasSdot(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
        &d_Rnew[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      gamma += temp_trace;
      trace_r_new += temp_trace;
      cudaDeviceSynchronize();
    }
    beta = trace_r_new/trace_r_old;

    /* update direction vecblk */
    cublasSscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasSaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Rnew, 1,
      d_Dvec, 1);
    cublasSnrm2(context_gpu->cublas_handle, m_B, d_Rnew, 1, &r_norm);
    r_norm = r_norm/B_norm;

    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
  }
}

void mp_zsy_global_cg
(
  /* solver parameters */
  const KrylovMeta meta,

  /* data */
  const MPSparseDescr A_descr,
  const MPSparseHandle A_handle,
  const MPInt n,
  const MPComplexDouble *B,
  MPComplexDouble *X,
  MPComplexDouble *memory,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* initialization */
  MPComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
  MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble MINUS_ONE_C = mp_scalar_z_init(-1.0, 0.0);
  MPInt i = 0;
  MPInt m_B = n;
  MPInt blk = meta.blk;
  MPComplexDouble alpha = ZERO_C;
  MPComplexDouble beta = ZERO_C;
  MPComplexDouble gamma = ZERO_C;
  double B_norm_abs = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, B, m_B);
  double r_norm_abs = 0.0;
  MPComplexDouble trace_r_old = ZERO_C;
  MPComplexDouble trace_r_new = ZERO_C;
  MPLayout layout = MP_COL_MAJOR;
  MPLayoutSparse sparse_layout;
  /* memory cpu*/
  MPComplexDouble *Rold = memory;
  MPComplexDouble *Rnew = &Rold[m_B*blk];
  MPComplexDouble *Dvec = &Rnew[m_B*blk];
  MPComplexDouble *Dvec_temp = &Dvec[m_B*blk];
  MPComplexDouble *Rtemp = NULL;

  /* first iteration */
  mp_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold) * m_B * blk);
  mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A_handle, A_descr,
    sparse_layout, X, blk, m_B, ONE_C, Rold, m_B);
  r_norm_abs = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rold, m_B)/B_norm_abs;
  memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);

  /* main loop */
  while ((i < meta.iterations) && (r_norm_abs > meta.tolerance))
  {
    /* computes alpha and gamma scalars */
    mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A_handle,
      A_descr, sparse_layout, Dvec, blk, m_B, ZERO_C, Dvec_temp, m_B);
    trace_r_old = ZERO_C;
    gamma = ZERO_C;

    mp_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Dvec,
             m_B*blk, Dvec_temp, m_B*blk, &ZERO_C, &gamma, 1);
    mp_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Rold,
             m_B*blk, Rold, m_B*blk, &ZERO_C, &trace_r_old, 1);
    alpha = mp_scalar_z_divide(trace_r_old, gamma);

    /* updates X (solution) and residual vecblk */
    mp_zaxpy(m_B*blk, &alpha, Dvec, 1, X, 1);
    memcpy(Rnew, Rold, (sizeof *Rnew)*m_B*blk);
    alpha = mp_scalar_z_invert_sign(alpha);
    mp_zaxpy(m_B*blk, &alpha, Dvec_temp, 1, Rnew, 1);
    /* computes beta scalar */
    trace_r_new = ZERO_C;
    mp_zgemm(layout, CblasTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Rnew,
             m_B*blk, Rnew, m_B*blk, &ZERO_C, &trace_r_new, 1);
    beta = mp_scalar_z_divide(trace_r_new, trace_r_old);

    /* updates direction vecblk */
    mp_zscal(m_B*blk, &beta, Dvec, 1);
    mp_zaxpy(m_B*blk, &ONE_C, Rnew, 1, Dvec, 1);
    r_norm_abs = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rnew, m_B);
    r_norm_abs = r_norm_abs/B_norm_abs;

    /* swaps old with new residual vecblks */
    Rtemp = Rnew;
    Rnew = Rold;
    Rold = Rtemp;
    i += 1;
  }
  #if defined(COLLECT_META) && (COLLECT_META)
      printf("STATUS: %d, MP_EXPERIMENT: %d\n", STATUS, MP_EXPERIMENT);
  #elif defined(DEBUG) && (DEBUG)
      printf("r_norm: %1.4E\n", r_norm_abs);
  #endif
}

void mp_zhe_global_cg
(
  /* solver parameters */
  const KrylovMeta meta,

  /* data */
  const MPSparseDescr A_descr,
  const MPSparseHandle A_handle,
  const MPInt n,
  const MPComplexDouble *B,
  MPComplexDouble *X,
  void *memory,

  /* collected metadata */
  MPSolverInfo *info
)
{
  MPComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
  MPComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
  MPComplexDouble MINUS_ONE_C = mp_scalar_z_init(-1.0, 0.0);
  MPInt i = 0;
  MPInt m_B = n;
  MPInt blk = meta.blk;
  MPComplexDouble alpha = ZERO_C;
  MPComplexDouble beta = ZERO_C;
  MPComplexDouble gamma = ZERO_C;

  double B_norm = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, B, m_B);
  double r_norm = 0.0;

  MPComplexDouble trace_r_old = ZERO_C;
  MPComplexDouble trace_r_new = ZERO_C;
  MPLayout layout = MP_COL_MAJOR;
  MPLayoutSparse sparse_layout;
  /* memory cpu*/
  MPComplexDouble *Rold = memory;
  MPComplexDouble *Rnew = &Rold[m_B*blk];
  MPComplexDouble *Dvec = &Rnew[m_B*blk];
  MPComplexDouble *Dvec_temp = &Dvec[m_B*blk];
  MPComplexDouble *Rtemp  = NULL;
  /* first iteration */
  mp_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(Rold, B, (sizeof *Rold) * m_B * blk);
  mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, MINUS_ONE_C, A_handle, A_descr,
    sparse_layout, X, blk, m_B, ONE_C, Rold, m_B);

  r_norm = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rold, m_B);
  memcpy(Dvec, Rold, (sizeof *Dvec)*m_B*blk);

  /* main loop */
  while ((i < meta.iterations) && (r_norm/B_norm > meta.tolerance))
  {
    /* computes alpha and gamma scalars */
    mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, A_handle,
      A_descr, sparse_layout, Dvec, blk, m_B, ZERO_C, Dvec_temp, m_B);
    trace_r_old = ZERO_C;
    gamma = ZERO_C;

    mp_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Dvec,
             m_B*blk, Dvec_temp, m_B*blk, &ZERO_C, &gamma, 1);
    mp_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Rold,
             m_B*blk, Rold, m_B*blk, &ZERO_C, &trace_r_old, 1);
    alpha = mp_scalar_z_divide(trace_r_old, gamma);

    /* updates X (solution) and residual vecblk */
    mp_zaxpy(m_B*blk, &alpha, Dvec, 1, X, 1);
    memcpy(Rnew, Rold, (sizeof *Rnew)*m_B*blk);
    alpha = mp_scalar_z_invert_sign(alpha);
    mp_zaxpy(m_B*blk, &alpha, Dvec_temp, 1, Rnew, 1);
    /* computes beta scalar */
    trace_r_new = ZERO_C;
    mp_zgemm(layout, CblasConjTrans, CblasNoTrans, 1, 1, m_B*blk, &ONE_C, Rnew,
             m_B*blk, Rnew, m_B*blk, &ZERO_C, &trace_r_new, 1);
    beta = mp_scalar_z_divide(trace_r_new, trace_r_old);

    /* updates direction vecblk */
    mp_zscal(m_B*blk, &beta, Dvec, 1);
    mp_zaxpy(m_B*blk, &ONE_C, Rnew, 1, Dvec, 1);
    r_norm = mp_zlange(LAPACK_COL_MAJOR, 'F', m_B, blk, Rnew, m_B);
    printf("        * r_norm/B_norm: %1.4E\n", r_norm/B_norm);
    //#if (COLLECT_META)
    //    /* decide first on the threading synchronization constructs scheme that will be used */
    //    //context->solver_current_block
    //    //context->solver_inner_meta_residual_array[] = r_norm;
    //#endif
    /* swaps old with new residual vecblks */
    Rtemp = Rnew;
    Rnew = Rold;
    Rold = Rtemp;
    i++;
  }
  #if defined(COLLECT_META) && (COLLECT_META)
      printf("STATUS: %d, MP_EXPERIMENT: %d\n", STATUS, MP_EXPERIMENT);
      //context->solver_outer_meta_iterations_array[] = i;
      //context->solver_outer_meta_restarts_array[] = r_norm;
  #elif defined(DEBUG) && (DEBUG)
      printf("r_norm/norm_b: %1.4E\n", r_norm/B_norm);
  #endif
}

/*===========================================================*/
/*== global_cg_cuda version for complex symmetric matrices ==*/
/*===========================================================*/

void mp_cuda_zsy_global_cg
( /* solver parameters */
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
  /* constants */
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);

  /* solver context */
  MPInt i = 0;

  MPInt m_B = n_A;
  MPInt blk = meta.blk;
  double B_norm = 0.0;
  double r_norm = 1.0;
  cuDoubleComplex alpha = ZERO_C;
  cuDoubleComplex beta = ZERO_C;
  cuDoubleComplex gamma = ZERO_C;
  cuDoubleComplex trace_r_old = ZERO_C;
  cuDoubleComplex trace_r_new = ZERO_C;
  /* memory cpu */
  cuDoubleComplex *d_Rold = memory_cuda;
  cuDoubleComplex *d_Rnew = &d_Rold[m_B*blk];
  cuDoubleComplex *d_Dvec = &d_Rnew[m_B*blk];
  cuDoubleComplex *d_Dvec_temp = &d_Dvec[m_B*blk];
  cuDoubleComplex *d_residual_temp  = NULL;

  /* descrs */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_r_old;
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;

  /* Initialize cuda descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, (void *) d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, (void *) d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp, CUDA_C_64F,
    CUSPARSE_ORDER_COL); // dont use this

  cudaDeviceSynchronize();
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cudaDeviceSynchronize();

  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_r_old, d_Rold);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
    &ONE_C, descr_r_old, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, &r_norm);

  cudaMemcpy(d_Dvec, d_Rold, (sizeof *d_Dvec)*m_B*blk,
    cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_direction, d_Dvec);

  /* -- main loop (iterations) -- */
  while ((i < meta.iterations) && (r_norm/B_norm > meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &ONE_C,
                 descr_A,
                 descr_direction,
                 &ZERO_C,
                 descr_dvec_temp,
                 CUDA_C_64F,
                 CUSPARSE_CSRMM_ALG1,
                 NULL);

    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Dvec_temp, 1, d_Dvec, 1,
      &gamma);
    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, d_Rold, 1,
      &trace_r_old);
    cudaDeviceSynchronize();
    alpha = mp_cuda_scalar_z_divide(trace_r_old, gamma);

    /* updates X and residual vecblk */
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cusparseDnMatSetValues(descr_direction, d_Dvec);

    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    alpha = mp_cuda_scalar_z_invert_sign(alpha);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);

    /* computes beta */
    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, d_Rnew, 1,
      &trace_r_new);
    cudaDeviceSynchronize();
    beta = mp_cuda_scalar_z_divide(trace_r_new, trace_r_old);

    /* update direction vecblk */
    cublasZscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &ONE_C, d_Rnew, 1,
      d_Dvec, 1);

    cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, &r_norm);

    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
    cusparseDnMatSetValues(descr_r_old, d_Rold);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    i += 1;

  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r_old);
  cusparseDestroyDnMat(descr_r_new);
  cusparseDestroyDnMat(descr_direction);
  cusparseDestroyDnMat(descr_dvec_temp);
}

/*
    global_cg_cuda version for complex symmetric matrices.
*/
void mp_cuda_zhe_global_cg
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
  /* constants */
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);

  /* solver context */
  MPInt i = 0;

  MPInt m_B = n_A;
  MPInt blk = meta.blk;
  double B_norm = 0.0;
  double r_norm = 1.0;

  cuDoubleComplex alpha = ZERO_C;
  cuDoubleComplex beta = ZERO_C;
  cuDoubleComplex gamma = ZERO_C;
  cuDoubleComplex trace_r_old = ZERO_C;
  cuDoubleComplex trace_r_new = ZERO_C;

  /* memory cpu */
  cuDoubleComplex *d_Rold = memory_cuda;
  cuDoubleComplex *d_Rnew = &d_Rold[m_B*blk];
  cuDoubleComplex *d_Dvec = &d_Rnew[m_B*blk];
  cuDoubleComplex *d_Dvec_temp = &d_Dvec[m_B*blk];
  cuDoubleComplex *d_residual_temp  = NULL;

  /* descrs */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_r_old;
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;

  /* Initialize cuda descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, (void *) d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, (void *) d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp, CUDA_C_64F,
    CUSPARSE_ORDER_COL); // dont use this

  cudaDeviceSynchronize();
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  cudaDeviceSynchronize();

  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_r_old, d_Rold);
  cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
    &ONE_C, descr_r_old, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, &r_norm);

  cudaMemcpy(d_Dvec, d_Rold, (sizeof *d_Dvec)*m_B*blk,
    cudaMemcpyDeviceToDevice);
  cusparseDnMatSetValues(descr_direction, d_Dvec);

  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm/B_norm > meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &ONE_C,
                 descr_A,
                 descr_direction,
                 &ZERO_C,
                 descr_dvec_temp,
                 CUDA_C_64F,
                 CUSPARSE_CSRMM_ALG1,
                 NULL);
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B*blk, d_Dvec_temp, 1, d_Dvec, 1,
      &gamma);
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B*blk, d_Rold, 1, d_Rold, 1,
      &trace_r_old);
    cudaDeviceSynchronize();
    alpha = mp_cuda_scalar_z_divide(trace_r_old, gamma);

    /* updates X and residual vecblk */
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cusparseDnMatSetValues(descr_direction, d_Dvec);

    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    alpha = mp_cuda_scalar_z_invert_sign(alpha);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);

    /* computes beta */
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, d_Rnew, 1,
      &trace_r_new);
    cudaDeviceSynchronize();
    beta = mp_cuda_scalar_z_divide(trace_r_new, trace_r_old);

    /* update direction vecblk */
    cublasZscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B*blk, &ONE_C, d_Rnew, 1,
      d_Dvec, 1);
    cublasDznrm2(context_gpu->cublas_handle, m_B*blk, d_Rnew, 1, &r_norm);

    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
    cusparseDnMatSetValues(descr_r_old, d_Rold);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    i += 1;

  }

  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r_old);
  cusparseDestroyDnMat(descr_r_new);
  cusparseDestroyDnMat(descr_direction);
  cusparseDestroyDnMat(descr_dvec_temp);
}

void mp_cuda_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt m_B = n;
  if (data_type == MP_REAL)
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(double)*
      (m_B   /* size_residual_new */
      +m_B   /* size_residual_old */
      +m_B); /* size_direction */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
     (m_B    /* size_residual_new */
     +m_B    /* size_residual_old */
     +m_B);  /* size_direction */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (m_B           /* size_residual_new */
      +m_B           /* size_residual_old */
      +m_B);         /* size_direction */
  }
  else if (data_type == MP_COMPLEX_32)
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(cuComplex)*
      (m_B           /* size_residual_new */
      +m_B           /* size_residual_old */
      +m_B);         /* size_direction */
  }
}

void mp_cuda_block_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt m_B = n;
  MPInt blk = meta.blk;

  if (data_type == MP_REAL)
  {
    *memory_bytes = sizeof(double)*
      (m_B*blk       /* size_residual_old */
      +m_B*blk       /* size_residual_new */
      +m_B*blk       /* size_direction */
      +m_B*blk       /* size_direction_new */
      +m_B*blk       /* size_direction_temp */
      +m_B*blk       /* size_R */
      +blk*blk       /* size_alpha */
      +blk*blk       /* size_beta */
      +blk*blk       /* size_zeta */
      +blk);         /* size_reflectors */

    *memory_cuda_bytes = sizeof(double)*
      (m_B*blk       /* size_residual_old */
      +m_B*blk       /* size_residual_new */
      +m_B*blk       /* size_direction */
      +m_B*blk       /* size_direction_new */
      +m_B*blk       /* size_direction_temp */
      +m_B*blk       /* size_R */
      +blk*blk       /* size_alpha */
      +blk*blk       /* size_beta */
      +blk*blk);     /* size_zeta */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    MPInt m_B = n;
    MPInt blk = meta.blk;
    *memory_bytes = sizeof(MPComplexDouble)*
      (m_B*blk    /* size_residual_old */
      +m_B*blk    /* size_residual_new */
      +m_B*blk    /* size_direction */
      +m_B*blk    /* size_direction_new */
      +m_B*blk    /* size_direction_temp */
      +m_B*blk    /* size_R */
      +blk*blk    /* size_alpha */
      +blk*blk    /* size_beta */
      +blk*blk    /* size_zeta */
      +blk);      /* size_reflectors */

    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (m_B*blk      /* size_residual_old */
      +m_B*blk      /* size_residual_new */
      +m_B*blk      /* size_direction */
      +m_B*blk      /* size_direction_new */
      +m_B*blk      /* size_direction_temp */
      +m_B*blk      /* size_R */
      +blk*blk      /* size_alpha */
      +blk*blk      /* size_beta */
      +blk*blk);    /* size_zeta */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes = sizeof(MPComplexDouble)*
      (m_B*blk    /* size_residual_old */
      +m_B*blk    /* size_residual_new */
      +m_B*blk    /* size_direction */
      +m_B*blk    /* size_direction_new */
      +m_B*blk    /* size_direction_temp */
      +m_B*blk    /* size_R */
      +blk*blk    /* size_alpha */
      +blk*blk    /* size_beta */
      +blk*blk    /* size_zeta */
      +blk);      /* size_reflectors */

    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (m_B*blk      /* size_residual_old */
      +m_B*blk      /* size_residual_new */
      +m_B*blk      /* size_direction */
      +m_B*blk      /* size_direction_new */
      +m_B*blk      /* size_direction_temp */
      +m_B*blk      /* size_R */
      +blk*blk      /* size_alpha */
      +blk*blk      /* size_beta */
      +blk*blk);    /* size_zeta */
  }
}

void mp_cuda_global_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
)
{
  MPInt m_B =n;
  MPInt blk = meta.blk;

  if (data_type == MP_REAL)
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(MPComplexDouble)*
      (m_B*blk      /* size_d_residual_old */
      +m_B*blk      /* size_d_residual_new */
      +m_B*blk      /* size_d_direction */
      +m_B*blk);    /* size_d_direction_new */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_SYMMETRIC))
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (m_B*blk      /* size_d_residual_old */
      +m_B*blk      /* size_d_residual_new */
      +m_B*blk      /* size_d_direction */
      +m_B*blk);    /* size_d_direction_temp */
  }
  else if ((data_type == MP_COMPLEX) && (struct_type == MP_MATRIX_HERMITIAN))
  {
    *memory_bytes = 0;
    *memory_cuda_bytes = sizeof(cuDoubleComplex)*
      (m_B*blk      /* size_d_residual_old */
      +m_B*blk      /* size_d_residual_new */
      +m_B*blk      /* size_d_direction */
      +m_B*blk);    /* size_d_direction_temp */
  }
}

/* COMPLEX */

void mp_cuda_zsy_cg
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const CudaInt n_A,
  const CudaInt nz_A,
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
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  /* solver context */
  MPInt i = 0;
  double norm_b = 0.0;
  double r_norm = 0.0;
  cuDoubleComplex alpha = ZERO_C;
  cuDoubleComplex beta  = ZERO_C;
  cuDoubleComplex temp_complex = ONE_C;
  MPInt m_B = n_A;
  /* memory cpu */
  cuDoubleComplex *d_r_new = memory_cuda;
  cuDoubleComplex *d_r_old = &d_r_new[m_B];
  cuDoubleComplex *d_dvec  = &d_r_old[m_B];
  cuDoubleComplex *d_temp_vector = NULL;

  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_r_old;
  CusparseDenseVectorDescriptor descr_r_new;
  CusparseDenseVectorDescriptor descr_direction;
  /* initialize descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, m_B, d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, m_B, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r_old, m_B, d_r_old, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r_new, m_B, d_r_new, CUDA_C_64F);
  cusparseCreateDnVec(&descr_direction, m_B, d_dvec, CUDA_C_64F);
  /* cg initialization */
  cudaMemcpy(d_r_old, d_b, (sizeof *d_r_old)*m_B, cudaMemcpyDeviceToDevice);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_b, 1, &norm_b);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r_old, CUDA_C_64F,
    CUSPARSE_CSRMV_ALG1, NULL);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_r_old, 1, &r_norm);
  cudaMemcpy(d_dvec, d_r_old, (sizeof *d_dvec)*m_B, cudaMemcpyDeviceToDevice);
  #if DEBUG == 1
    printf("relative residual: %1.4E\n", r_norm/norm_b);
  #endif

  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm/norm_b > meta.tolerance))
  {
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_C, descr_A, descr_direction, &ZERO_C, descr_r_new, CUDA_C_64F,
      CUSPARSE_CSRMV_ALG1, NULL);

    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B, d_r_new, 1, d_dvec, 1, &alpha);
    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_complex);
    cudaDeviceSynchronize();
    alpha = mp_cuda_scalar_z_divide(temp_complex, alpha);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &alpha, d_dvec, 1, d_x, 1);  /* W <-- W - V(:, J)*H(j, j)) */

    alpha = mp_cuda_scalar_z_invert_sign(alpha);
    cublasZscal(context_gpu->cublas_handle, m_B, &alpha, d_r_new, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &ONE_C, d_r_old, 1, d_r_new, 1);  /* W <-- W - V(:, J)*H(j, j)) */
    cusparseDnVecSetValues(descr_r_new, d_r_new);

    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B, d_r_new, 1, d_r_new, 1,
      &beta);
    cudaDeviceSynchronize();
    cublasZdotu(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_complex);
    cudaDeviceSynchronize();
    beta = mp_cuda_scalar_z_divide(beta, temp_complex);
    cublasZscal(context_gpu->cublas_handle, m_B, &beta, d_dvec, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &ONE_C, d_r_new, 1, d_dvec, 1);  /* W <-- W - V(:, J)*H(j, j)) */
    cusparseDnVecSetValues(descr_direction, d_dvec);

    cublasDznrm2(context_gpu->cublas_handle, m_B, d_r_new, 1, &r_norm);     /* computes || residual ||_F */
    d_temp_vector = d_r_old;
    d_r_old = d_r_new;
    d_r_new = d_temp_vector;
    cusparseDnVecSetValues(descr_r_old, d_r_old);
    cusparseDnVecSetValues(descr_r_new, d_r_new);
    i = i + 1;
  }
  d_r_new = NULL;
  d_r_old = NULL;
  d_temp_vector = NULL;
}

void mp_cuda_zhe_cg
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const CudaInt n_A,
  const CudaInt nz_A,
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
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  /* solver context */
  MPInt i = 0;
  double norm_b = 0.0;
  double r_norm = 0.0;
  cuDoubleComplex alpha = ZERO_C;
  cuDoubleComplex beta  = ZERO_C;
  cuDoubleComplex temp_complex = ONE_C;
  MPInt m_B = n_A;

  /* memory cpu */
  cuDoubleComplex *d_r_new = memory;
  cuDoubleComplex *d_r_old = &d_r_new[m_B];
  cuDoubleComplex *d_dvec  = &d_r_old[m_B];
  cuDoubleComplex *d_temp_vector = NULL;

  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseVectorDescriptor descr_b;
  CusparseDenseVectorDescriptor descr_x;
  CusparseDenseVectorDescriptor descr_r_old;
  CusparseDenseVectorDescriptor descr_r_new;
  CusparseDenseVectorDescriptor descr_direction;
  /* initialize descrs and memory */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnVec(&descr_b, m_B, d_b, CUDA_C_64F);
  cusparseCreateDnVec(&descr_x, m_B, d_x, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r_old, m_B, d_r_old, CUDA_C_64F);
  cusparseCreateDnVec(&descr_r_new, m_B, d_r_new, CUDA_C_64F);
  cusparseCreateDnVec(&descr_direction, m_B, d_dvec, CUDA_C_64F);
  /* cg initialization */
  cudaMemcpy(d_r_old, d_b, (sizeof *d_r_old)*m_B, cudaMemcpyDeviceToDevice);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_b, 1, &norm_b);
  cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_x, &ONE_C, descr_r_old, CUDA_C_64F,
    CUSPARSE_CSRMV_ALG1, NULL);
  cublasDznrm2(context_gpu->cublas_handle, m_B, d_r_old, 1, &r_norm);
  cudaMemcpy(d_dvec, d_r_old, (sizeof *d_dvec)*m_B, cudaMemcpyDeviceToDevice);

  /* main loop (iterations) */
  while ((i < meta.iterations) && (r_norm/norm_b > meta.tolerance))
  {
    cusparseSpMV(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      &ONE_C, descr_A, descr_direction, &ZERO_C, descr_r_new, CUDA_C_64F,
      CUSPARSE_CSRMV_ALG1, NULL);
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B, d_r_new, 1, d_dvec, 1, &alpha);
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_complex);
    cudaDeviceSynchronize();
    alpha = mp_cuda_scalar_z_divide(temp_complex, alpha);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &alpha, d_dvec, 1, d_x, 1);

    alpha = mp_cuda_scalar_z_invert_sign(alpha);
    cublasZscal(context_gpu->cublas_handle, m_B, &alpha, d_r_new, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &ONE_C, d_r_old, 1, d_r_new, 1);
    cusparseDnVecSetValues(descr_r_new, d_r_new);

    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B, d_r_new, 1, d_r_new, 1, &beta);
    cudaDeviceSynchronize();
    cublasZdotc(context_gpu->cublas_handle, m_B, d_r_old, 1, d_r_old, 1,
      &temp_complex);
    cudaDeviceSynchronize();
    beta = mp_cuda_scalar_z_divide(beta, temp_complex);
    cublasZscal(context_gpu->cublas_handle, m_B, &beta, d_dvec, 1);
    cublasZaxpy(context_gpu->cublas_handle, m_B, &ONE_C, d_r_new, 1, d_dvec, 1);
    cusparseDnVecSetValues(descr_direction, d_dvec);

    cublasDznrm2(context_gpu->cublas_handle, m_B, d_r_new, 1, &r_norm);
    d_temp_vector = d_r_old;
    d_r_old = d_r_new;
    d_r_new = d_temp_vector;
    cusparseDnVecSetValues(descr_r_old, d_r_old);
    cusparseDnVecSetValues(descr_r_new, d_r_new);
    i = i + 1;
  }
  #if DEBUG == 1
      printf("relative residual: %1.4E\n", r_norm / norm_b);
      printf("iterations completed: %d\n", i);
  #endif

  d_r_new = NULL;
  d_r_old = NULL;
  d_temp_vector = NULL;
}

void mp_cuda_zsy_block_cg
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
  /* constants */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  CudaInt i = 0;
  CudaInt m_B = m_A;
  CudaInt blk = meta.blk;
  double B_norm = 0.0;
  double R_norm = 0.0;
  size_t size_buffer;
  /* memory cpu */
  MPComplexDouble *R = memory;
  MPComplexDouble *alpha_matrix = &R[m_B*blk];
  MPComplexDouble *beta_matrix  = &alpha_matrix[blk*blk];
  MPComplexDouble *zeta_matrix  = &beta_matrix[blk*blk];
  MPComplexDouble *Dvec = &zeta_matrix[blk*blk];
  /* memory gpu */
  cuDoubleComplex *d_Rold = memory;
  cuDoubleComplex *d_Rnew = &d_Rold[m_B*blk];
  cuDoubleComplex *d_Dvec = &d_Rnew[m_B*blk];
  cuDoubleComplex *d_Dvec_new = &d_Dvec[m_B*blk];
  cuDoubleComplex *d_Dvec_temp = &d_Dvec_new[m_B*blk];
  cuDoubleComplex *d_R = &d_Dvec_temp[m_B*blk];
  cuDoubleComplex *d_alpha_matrix = &d_R[m_B*blk];
  cuDoubleComplex *d_beta_matrix = &d_alpha_matrix[blk*blk];
  cuDoubleComplex *d_zeta_matrix = &d_beta_matrix[blk*blk];
  cuDoubleComplex *d_temp_vecblk = NULL;
  cuDoubleComplex *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  MPInt *pivots_array = mp_malloc((sizeof *pivots_array) * blk);
  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_r_old;
  CusparseDenseMatrixDescriptor descr_r_new;
  CusparseDenseMatrixDescriptor descr_direction;
  CusparseDenseMatrixDescriptor descr_direction_new;
  CusparseDenseMatrixDescriptor descr_dvec_temp;
  /* initializes cuda descrs */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction_new, m_B, blk, m_B, d_Dvec_new,
    CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUDA_C_64F, CUSPARSE_ORDER_COL);
  /* krylov initialization */
  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r_old, CUDA_R_64F,
    CUSPARSE_CSRMM_ALG1, &size_buffer);
  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc((void **) &d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
      &ONE_C, descr_r_old, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_X,
      &ONE_C, descr_r_old, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
  }
  cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_B, 1, &B_norm);
  cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_Rold, 1, &R_norm);
  #if DEBUG == 1
    printf("R_norm: %1.4E\n", R_norm);
  #endif
  cudaMemcpy(Dvec, d_Rold, (sizeof *Dvec)*m_B*blk, cudaMemcpyDeviceToHost);
  mp_gram_schmidt_zge(m_B, blk, Dvec, R, m_B);;
  cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec)*m_B*blk, cudaMemcpyHostToDevice);

  /* main loop */
  while ((i < meta.iterations) && (R_norm/B_norm > meta.tolerance))
  {
    /* computes alpha_matrix */
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C,
      descr_A, descr_direction, &ZERO_C, descr_direction_new, CUDA_C_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_direction, &ZERO_C, descr_direction_new,
          CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_direction,
      &ZERO_C, descr_direction_new, CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
      blk, m_B, &ONE_C, d_Dvec, m_B, d_Dvec_new, m_B, &ZERO_C, d_beta_matrix,
      blk);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
      blk, m_B, &ONE_C, d_Rold, m_B, d_Rold, m_B, &ZERO_C, d_alpha_matrix,
      blk);
    cudaMemcpy(d_zeta_matrix, d_alpha_matrix, (sizeof *d_zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(alpha_matrix, d_alpha_matrix, (sizeof *alpha_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
      CblasNonUnit, blk, blk, &ONE_C, R, m_B, alpha_matrix, blk);
    cudaMemcpy(beta_matrix, d_beta_matrix, (sizeof *beta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);
    LAPACKE_zgesv(CblasColMajor, blk, blk, beta_matrix, blk, pivots_array,
      alpha_matrix, blk);

    /* updates solution X and residual */
    cudaMemcpy(d_alpha_matrix, alpha_matrix, (sizeof *d_alpha_matrix)*blk*blk,
       cudaMemcpyHostToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
      blk, blk, &ONE_C, d_Dvec, m_B, d_alpha_matrix, blk, &ONE_C, d_X, m_B);

    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
      blk, blk, &ONE_C, d_Dvec, m_B, d_alpha_matrix, blk, &ZERO_C,
      d_Dvec_temp, m_B);

    cusparseDnMatSetValues(descr_dvec_temp, d_Dvec_temp);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_dvec_temp, &ONE_C, descr_r_new, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &MINUS_ONE_C, descr_A, descr_dvec_temp,
        &ONE_C, descr_r_new, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
        d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &MINUS_ONE_C, descr_A,
        descr_dvec_temp, &ONE_C, descr_r_new, CUDA_C_64F, CUSPARSE_CSRMM_ALG1,
        NULL);
    }

    /* computes beta parameters */
    mp_zimatcopy('C', 'T', blk, blk, ONE_C_CPU, R, m_B, blk);
    cudaMemcpy(zeta_matrix, d_zeta_matrix, (sizeof *zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    mp_zgesv(CblasColMajor, blk, blk, zeta_matrix, blk, pivots_array, R, blk);
    mp_zimatcopy('C', 'T', blk, blk, ONE_C_CPU, R, blk, m_B);
    cudaMemcpy(d_R, R, (sizeof *d_R) * m_B * blk, cudaMemcpyHostToDevice);

    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, blk,
      blk, m_B, &ONE_C, d_Rnew, m_B, d_Rnew, m_B, &ZERO_C, d_beta_matrix,
      blk);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, blk,
      blk, blk, &ONE_C, d_R, m_B, d_beta_matrix, blk, &ZERO_C, d_beta_matrix,
      blk);

    /* compute zeta (reorthogonalize) */
    cudaMemcpy(d_Rold, d_Rnew, (sizeof *d_Rold)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
      blk, blk, &ONE_C, d_Dvec, m_B, d_beta_matrix, blk, &ONE_C, d_Rold, m_B);
    //cudaMemcpy(d_Dvec, d_Rold, (sizeof *Dvec) * m_B * blk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Dvec, d_Rold, (sizeof *Dvec)*m_B*blk, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Dvec, d_Dvec, (sizeof *Dvec) * m_B * blk, cudaMemcpyDeviceToHost);
    mp_gram_schmidt_zge(m_B, blk, Dvec, R, m_B);
    cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_Rnew, 1, &R_norm);
    cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec)*m_B*blk,
      cudaMemcpyHostToDevice);
    d_temp_vecblk = d_Rold;
    d_Rold = d_Rnew;
    d_Rnew = d_temp_vecblk;
    i += 1;
  }
  #if DEBUG == 1
    printf("R_norm: %1.4E\n", R_norm / B_norm);
    printf("iterations: %d/%d\n", i, meta.iterations);
  #endif

  /* memory deallocation */
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r_old);
  cusparseDestroyDnMat(descr_r_new);
  cusparseDestroyDnMat(descr_direction);
  cusparseDestroyDnMat(descr_direction_new);
  cusparseDestroyDnMat(descr_dvec_temp);
  mp_free(reflectors_array);
  mp_free(pivots_array);
}

void mp_cuda_zhe_block_cg
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
  /* constants */
  MPComplexDouble ONE_C_CPU = mp_scalar_z_init(1.0, 0.0);
  cuDoubleComplex MINUS_ONE_C = mp_cuda_scalar_z_init(-1.0, 0.0);
  cuDoubleComplex ONE_C = mp_cuda_scalar_z_init(1.0, 0.0);
  cuDoubleComplex ZERO_C = mp_cuda_scalar_z_init(0.0, 0.0);
  /* solver context */
  CudaInt i = 0;
  CudaInt m_B = n_A;
  CudaInt blk = meta.blk;
  double B_norm = 0.0;
  double R_norm = 0.0;
  size_t size_buffer;
  /* memory cpu */
  MPComplexDouble *R = (MPComplexDouble *) memory_cuda;
  MPComplexDouble *alpha_matrix = &R[m_B*blk];
  MPComplexDouble *beta_matrix = &alpha_matrix[blk*blk];
  MPComplexDouble *zeta_matrix = &beta_matrix[blk*blk];
  MPComplexDouble *Dvec = &zeta_matrix[blk*blk];
  /* memory gpu */
  cuDoubleComplex *d_Rold = memory_cuda;
  cuDoubleComplex *d_Rnew = &d_Rold[m_B*blk];
  cuDoubleComplex *d_Dvec = &d_Rnew[m_B*blk];
  cuDoubleComplex *d_Dvec_new = &d_Dvec[m_B*blk];
  cuDoubleComplex *d_Dvec_temp = &d_Dvec_new[m_B*blk];
  cuDoubleComplex *d_R = &d_Dvec_temp[m_B*blk];
  cuDoubleComplex *d_alpha_matrix = &d_R[m_B*blk];
  cuDoubleComplex *d_beta_matrix = &d_alpha_matrix[blk*blk];
  cuDoubleComplex *d_zeta_matrix = &d_beta_matrix[blk*blk];
  cuDoubleComplex *d_temp_vecblk = NULL;
  cuDoubleComplex *reflectors_array = mp_malloc((sizeof *reflectors_array)*blk);
  MPInt *pivots_array = mp_malloc((sizeof *pivots_array)*blk);
  /* cuda descrs */
  CusparseCsrMatrixDescriptor descr_A;
  CusparseDenseMatrixDescriptor descr_B;
  CusparseDenseMatrixDescriptor descr_X;
  CusparseDenseMatrixDescriptor descr_r_old;
  CusparseDenseMatrixDescriptor descr_r_new;
  CusparseDenseMatrixDescriptor descr_direction;
  CusparseDenseMatrixDescriptor descr_direction_new;
  CusparseDenseMatrixDescriptor descr_dvec_temp;
  /* initializes cuda descrs */
  printf("IN CUDA_ZHE_BLOCK_CG\n");
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_C_64F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUDA_C_64F,
     CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec, CUDA_C_64F,
    CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_direction_new, m_B, blk, m_B, d_Dvec_new,
    CUDA_C_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUDA_C_64F, CUSPARSE_ORDER_COL);

  /* krylov initialization */
  cudaMemcpy(d_Rold, d_B, (sizeof *d_Rold)*m_B*blk, cudaMemcpyDeviceToDevice);
  size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r_old, CUDA_C_64F,
    CUSPARSE_CSRMM_ALG1, &size_buffer);
  if (size_buffer > 0)
  {
    void *d_external_buffer = NULL;
    cudaMalloc((void **) &d_external_buffer, size_buffer);
    cusparseSpMM(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r_old, CUDA_C_64F,
      CUSPARSE_CSRMM_ALG1, d_external_buffer);
    cudaFree(d_external_buffer);
  }
  else if (size_buffer == 0)
  {
    cusparseSpMM(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_X, &ONE_C, descr_r_old, CUDA_C_64F,
      CUSPARSE_CSRMM_ALG1, NULL);
  }
  cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_B, 1, &B_norm);
  cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_Rold, 1, &R_norm);
  #if DEBUG == 1
      printf("B_norm: %1.4E\n", B_norm);
  #endif
  cudaMemcpy(Dvec, d_Rold, (sizeof *Dvec)*m_B*blk, cudaMemcpyDeviceToHost);
  mp_gram_schmidt_zhe(m_B, blk, Dvec, R, m_B);
  cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec) * m_B * blk, cudaMemcpyHostToDevice);

  /* main loop */
  while ((i < meta.iterations) && (R_norm/B_norm > meta.tolerance))
  {
    /* computes alpha_matrix */
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &ONE_C, descr_A, descr_direction, &ZERO_C, descr_direction_new,
      CUDA_C_64F, CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_direction, &ZERO_C, descr_direction_new,
        CUDA_C_64F, CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &ONE_C, descr_A, descr_direction, &ZERO_C, descr_direction_new,
        CUDA_C_64F, CUSPARSE_CSRMM_ALG1, NULL);
    }

    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_C, d_Dvec, m_B, d_Dvec_new, m_B, &ZERO_C, d_beta_matrix, blk);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_C, d_Rold, m_B, d_Rold, m_B, &ZERO_C, d_alpha_matrix, blk);
    cudaMemcpy(d_zeta_matrix, d_alpha_matrix, (sizeof *d_zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(alpha_matrix, d_alpha_matrix, (sizeof *alpha_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
      CblasNonUnit, blk, blk, &ONE_C, R, m_B, alpha_matrix, blk);
    cudaMemcpy(beta_matrix, d_beta_matrix, (sizeof *beta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);
    LAPACKE_zgesv(CblasColMajor, blk, blk, beta_matrix, blk, pivots_array,
      alpha_matrix, blk);

    /* updates solution X and residual */
    cudaMemcpy(d_alpha_matrix, alpha_matrix, (sizeof *d_alpha_matrix)*blk*blk,
      cudaMemcpyHostToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B,
      blk, blk, &ONE_C, d_Dvec, m_B, d_alpha_matrix, blk, &ONE_C, d_X, m_B);

    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &ONE_C, d_Dvec, m_B, d_alpha_matrix, blk, &ZERO_C, d_Dvec_temp, m_B);

    cusparseDnMatSetValues(descr_dvec_temp, d_Dvec_temp);
    cusparseDnMatSetValues(descr_r_new, d_Rnew);
    size_buffer = cusparseSpMM_bufferSize(context_gpu->cusparse_handle,
      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &MINUS_ONE_C, descr_A, descr_dvec_temp, &ONE_C, descr_r_new, CUDA_R_64F,
      CUSPARSE_CSRMM_ALG1, &size_buffer);
    if (size_buffer > 0)
    {
      void *d_external_buffer = NULL;
      cudaMalloc(d_external_buffer, size_buffer);
      cusparseSpMM(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &MINUS_ONE_C, descr_A, descr_dvec_temp, &ONE_C, descr_r_new, CUDA_C_64F,
        CUSPARSE_CSRMM_ALG1, d_external_buffer);
      cudaFree(d_external_buffer);
    }
    else if (size_buffer == 0)
    {
      cusparseSpMM(context_gpu->cusparse_handle,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &MINUS_ONE_C, descr_A, descr_dvec_temp, &ONE_C, descr_r_new, CUDA_C_64F,
        CUSPARSE_CSRMM_ALG1, NULL);
    }

    /* computes beta parameters */
    mp_zimatcopy('C', 'C', blk, blk, ONE_C_CPU, R, m_B, blk);
    cudaMemcpy(zeta_matrix, d_zeta_matrix, (sizeof *zeta_matrix)*blk*blk,
      cudaMemcpyDeviceToHost);

    mp_zgesv(CblasColMajor, blk, blk, zeta_matrix, blk, pivots_array, R, blk);
    mp_zimatcopy('C', 'C', blk, blk, ONE_C_CPU, R, blk, m_B);
    cudaMemcpy(d_R, R, (sizeof *d_R)*m_B*blk, cudaMemcpyHostToDevice);

    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, blk, blk,
      m_B, &ONE_C, d_Rnew, m_B, d_Rnew, m_B, &ZERO_C, d_beta_matrix, blk);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, blk, blk,
      blk, &ONE_C, d_R, m_B, d_beta_matrix, blk, &ZERO_C, d_beta_matrix, blk);

    /* compute zeta (reorthogonalize) */
    cudaMemcpy(d_Rold, d_Rnew, (sizeof *d_Rold)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasZgemm(context_gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_B, blk,
      blk, &ONE_C, d_Dvec, m_B, d_beta_matrix, blk, &ONE_C, d_Rold, m_B);
    //cudaMemcpy(d_Dvec, d_Rold, (sizeof *Dvec) * m_B * blk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Dvec, d_Rold, (sizeof *Dvec) * m_B * blk, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Dvec, d_Dvec, (sizeof *Dvec) * m_B * blk, cudaMemcpyDeviceToHost);
    mp_gram_schmidt_zhe(m_B, blk, Dvec, R, m_B);
    cublasDznrm2(context_gpu->cublas_handle, m_B * blk, d_Rnew, 1, &R_norm);
    cudaMemcpy(d_Dvec, Dvec, (sizeof *d_Dvec) * m_B * blk,
      cudaMemcpyHostToDevice);
    d_temp_vecblk = d_Rold;
    d_Rold = d_Rnew;
    d_Rnew = d_temp_vecblk;
    printf("r_norm: %1.8E\n", R_norm/B_norm);
    i += 1;
  }
  #if DEBUG == 1
    printf("R_norm: %1.4E\n", R_norm / B_norm);
    printf("iterations: %d/%d\n", i, meta.iterations);
  #endif

  /* memory deallocation */
  cusparseDestroySpMat(descr_A);
  cusparseDestroyDnMat(descr_B);
  cusparseDestroyDnMat(descr_X);
  cusparseDestroyDnMat(descr_r_old);
  cusparseDestroyDnMat(descr_r_new);
  cusparseDestroyDnMat(descr_direction);
  cusparseDestroyDnMat(descr_direction_new);
  cusparseDestroyDnMat(descr_dvec_temp);
  mp_free(reflectors_array);
  mp_free(pivots_array);
}

void mp_cuda_csy_global_cg
(
  /* solver parameters */
  MPContextGpuCuda *context_gpu,
  const KrylovMeta meta,

  /* data */
  const MPInt n_A,
  const MPInt nz_A,
  const MPSparseCsr_Cuda A,
  cuComplex *d_B,
  cuComplex *d_X,
  void *memory,
  void *memory_cuda,

  /* collected metadata */
  MPSolverInfo *info
)
{
  /* constants */
  cuComplex ONE_C = mp_cuda_scalar_c_init(1.0, 0.0);
  cuComplex ZERO_C = mp_cuda_scalar_c_init(0.0, 0.0);
  cuComplex temp_trace = ZERO_C;
  /* solver context */
  MPInt i = 0;
  MPInt j = 0;
  MPInt m_B = n_A;
  MPInt blk = meta.blk;
  float B_norm = 0.0;
  float r_norm = 1.0;
  cuComplex alpha, beta, gamma;
  cuComplex trace_r_old = ZERO_C;
  cuComplex trace_r_new = ZERO_C;
  /* memory cpu */
  cuComplex *d_Rold = memory;
  cuComplex *d_Rnew = &d_Rold[m_B*blk];
  cuComplex *d_residual_temp = &d_Rnew[m_B*blk];
  cuComplex *d_Dvec = &d_Rnew[m_B*blk];
  cuComplex *d_Dvec_temp = NULL;
  /* cuda descrs */
  MPCusparseCsrMatrixDescriptor descr_A;
  MPCusparseDenseMatrixDescriptor descr_B;
  MPCusparseDenseMatrixDescriptor descr_X;
  MPCusparseDenseMatrixDescriptor descr_r_old;
  MPCusparseDenseMatrixDescriptor descr_r_new;
  MPCusparseDenseMatrixDescriptor descr_r_temp;
  MPCusparseDenseMatrixDescriptor descr_direction;
  MPCusparseDenseMatrixDescriptor descr_dvec_temp;
  /* Initializes descrs */
  cusparseCreateCsr(&descr_A, m_B, m_B, nz_A, A.d_row_pointers, A.d_cols,
    A.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_B, m_B, blk, m_B, d_B, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_X, m_B, blk, m_B, d_X, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_old, m_B, blk, m_B, d_Rold, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_new, m_B, blk, m_B, d_Rnew, CUSPARSE_ORDER_COL,
    CUDA_R_32F);
  cusparseCreateDnMat(&descr_r_temp, m_B, blk, m_B, d_residual_temp,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cusparseCreateDnMat(&descr_direction, m_B, blk, m_B, d_Dvec,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cusparseCreateDnMat(&descr_dvec_temp, m_B, blk, m_B, d_Dvec_temp,
    CUSPARSE_ORDER_COL, CUDA_R_32F);
  cublasScnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);

  /* main loop */
  cublasScnrm2(context_gpu->cublas_handle, m_B*blk, d_B, 1, &B_norm);
  while ((i < meta.iterations) && (r_norm < meta.tolerance))
  {
    /* computes alpha, gamma */
    cusparseSpMM(context_gpu->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_C, descr_A, descr_direction,
      &ZERO_C, descr_r_new, CUDA_R_32F, CUSPARSE_CSRMM_ALG1, NULL);
    trace_r_old = ZERO_C;
    gamma = ZERO_C;
    for (j = 0; j < blk; ++j)
    {
      cublasCdotu(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
        &d_Rold[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      gamma = mp_cuda_scalar_c_add(gamma, temp_trace);
      cudaDeviceSynchronize();
    }
    alpha = mp_cuda_scalar_c_divide(trace_r_old, gamma);
    /* updates X and residual vecblk */
    cublasCaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec, 1, d_X, 1);
    cudaMemcpy(d_Rnew, d_Rold, (sizeof *d_Rnew)*m_B*blk,
      cudaMemcpyDeviceToDevice);
    cublasCaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Dvec_temp, 1,
      d_Rnew, 1);
    /* computes beta */
    trace_r_new = ZERO_C;
    for (j = 0; j < blk; ++j)
    {
      cublasCdotu(context_gpu->cublas_handle, m_B, &d_Rold[m_B*j], 1,
        &d_Rnew[m_B*j], 1, &temp_trace);
      cudaDeviceSynchronize();
      gamma = mp_cuda_scalar_c_add(gamma, temp_trace);
      trace_r_new = mp_cuda_scalar_c_add(trace_r_new, temp_trace);
      cudaDeviceSynchronize();
    }
    beta = mp_cuda_scalar_c_divide(trace_r_new, trace_r_old);

    /* update direction vecblk */
    cublasCscal(context_gpu->cublas_handle, m_B*blk, &beta, d_Dvec, 1);
    cublasCaxpy(context_gpu->cublas_handle, m_B*blk, &alpha, d_Rnew, 1,
      d_Dvec, 1);
    cublasScnrm2(context_gpu->cublas_handle, m_B, d_Rnew, 1, &r_norm);
    r_norm = r_norm/B_norm;
    /* swaps old with new residual vecblks*/
    d_residual_temp = d_Rnew;
    d_Rnew = d_Rold;
    d_Rold = d_residual_temp;
  }
}

