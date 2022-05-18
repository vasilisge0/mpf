#ifndef MP_CUDA_AUXILIARY_H
#define MP_CUDA_AUXILIARY_H

/* ----------------- complex scalar manipulation functions ------------------ */

cuDoubleComplex mp_cuda_scalar_z_set
(
  const double variable_real
);

cuDoubleComplex mp_cuda_scalar_z_init
(
  const double real_value,
  const double imag_value
);

cuDoubleComplex mp_cuda_scalar_z_add
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
);

cuDoubleComplex mp_cuda_scalar_z_divide
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
);

cuDoubleComplex mp_cuda_scalar_z_multiply
(
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta
);

cuDoubleComplex mp_cuda_scalar_z_normalize
(
  cuDoubleComplex alpha,
  const double beta
);

cuDoubleComplex mp_cuda_scalar_z_subtract
(
  cuDoubleComplex alpha,
  const cuDoubleComplex beta
);

cuDoubleComplex mp_cuda_scalar_z_invert_sign
(
  cuDoubleComplex alpha
);

cuComplex mp_cuda_scalar_c_init
(
  const float real_value,
  const float imag_value
);

cuComplex mp_cuda_scalar_c_add
(
  const cuComplex alpha,
  const cuComplex beta
);

cuComplex mp_cuda_scalar_c_divide
(
  const cuComplex alpha,
  const cuComplex beta
);

cuComplex mp_cuda_scalar_c_multiply
(
  const cuComplex alpha,
  const cuComplex beta
);

cuComplex mp_cuda_scalar_c_normalize
(
  cuComplex alpha,
  const float beta
);

cuComplex mp_cuda_scalar_c_invert_sign
(
  cuComplex alpha
);

cuComplex mp_cuda_scalar_c_subtract
(
  cuComplex alpha,
  const cuComplex beta
);

cuDoubleComplex mp_scalar_cuda_z_init
(
  const double real_value,
  const double imag_value
);


/* --------------------------- printout functions ----------------------------*/


void mp_cuda_matrix_z_announce
(
  cuDoubleComplex *A,
  const CudaInt m_A,
  const CudaInt n_A,
  const CudaInt ld_A,
  char filename[100]
);

void mp_matrix_cuda_print
(
  double *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
);

void mp_cuda_matrix_d_print
(
  double *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
);

void mp_cuda_matrix_z_print
(
  cuDoubleComplex *Acuda,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt ld_A
);

void mp_debug_cuda
(
  char *filename,
  double *A,
  MPInt m_A,
  MPInt n_A
);

/* ------------------------  cuda context initializer ----------------------- */


void mp_context_cuda_init
(
  MPContext* context
);

void mp_context_gpu_cuda_finish
(
  MPContext* context
);


/* ------------------------- sparse object initializer -----------------------*/


void mp_cuda_sparse_descr_create
(
  MPDataType data_type,
  CudaInt m_A,
  CudaInt n_A,
  CudaInt nz_A,
  cusparseSpMatDescr_t descr_A,
  MPSparseCsr_Cuda *A
);


/* --------------------------- solver initializers -------------------------- */


void mp_cuda_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
);

void mp_cuda_block_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
);

void mp_cuda_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_block_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_global_gmres_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_block_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_global_lanczos_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations,
  MPInt restarts
);

void mp_cuda_global_cg_init
(
  MPContext *context,
  double tolerance,
  MPInt iterations
);

void mp_cuda_probing_blocking_init
(
  MPContext *context,
  MPInt n_threads_probing,
  MPInt blk_threads_probing,
  MPInt blk,
  MPInt n_levels
);

void mp_cuda_batch_init
(
  MPContext *context
);

void mp_cuda_matrix_d_announce
(
  double *A,
  const CudaInt m_A,
  const CudaInt n_A,
  const CudaInt ld_A,
  char filename[100]
);

#endif /* END OF MP_CUDA_AUXILLIARY */
