#ifndef MP_CUDA_SOLVE_H
#define MP_CUDA_SOLVE_H



/* ---------------------------- GMRES solvers ------------------------------- */



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
);

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
);

void mp_cuda_zsy_gmres
(
  /* solver parameters */
  MPContextCuda *context_gpu,   /* (input) */
  const KrylovMeta meta,        /* (input) */

  /* data */
  MPSparseCsr_Cuda *A,          /* (input) */
  const MPInt n,                /* (input) */
  const cuDoubleComplex *d_b,   /* (input) */
  cuDoubleComplex *d_x,         /* (output) */
  MPComplexDouble *memory_cpu,  /* (input/output) */
  cuDoubleComplex *memory_gpu,  /* (input/output) */

  /* solver metadata */
  MPSolverInfo *info            /* (output )*/
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);



/* --------------------- memory allocation for GMRES ------------------------ */




void mp_cuda_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  MPInt n,
  KrylovMeta meta,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
);

void mp_cuda_block_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
);

void mp_cuda_global_gmres_memory_get
(
  MPDataType data_type,
  MPMatrixType structure_type,
  MPInt n,
  KrylovMeta meta,
  MPInt *memory_bytes_cpu,
  MPInt *memory_bytes_gpu
);



/* ----------------------------- Lanczos solvers ---------------------------- */



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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);



/* ----------------- memory allocation for Lanczos solvers ------------------ */



void mp_cuda_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);

void mp_cuda_block_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);

void mp_cuda_global_lanczos_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);



/* ----------------------------- CG solvers --------------------------------- */



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
);

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
);

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
);



/* -------------------------- CG memory management -------------------------- */



void mp_cuda_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);

void mp_cuda_block_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);

void mp_block_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes
);

void mp_cuda_global_cg_memory_get
(
  MPDataType data_type,
  MPMatrixType struct_type,
  KrylovMeta meta,
  MPInt n,
  MPInt *memory_bytes,
  MPInt *memory_cuda_bytes
);


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
);

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
);

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
);


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
);

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
);

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
);

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
);

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
);

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
);

#endif /* MP_CUDA_SOLVE_H */
