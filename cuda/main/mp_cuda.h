#ifndef MP_CUDA_H
#define MP_CUDA_H

/* === cuda related === */

#ifndef MP_DEVICE_GPU
    #define MP_DEVICE_GPU CUDA
#endif
#define DEBUG 0

#define MP_HEAP_DEFRAG_TRHES 0.6

#if MP_DEVICE_GPU == CUDA
  #include "cusparse_v2.h"
  #include "cublas_v2.h"
  #include "cuda_runtime.h"
  #include "device_launch_parameters.h"
  #include <cuda_fp16.h>

  #define MAX_NUM_DENSE_DESCRIPTORS 10
  #define MAX_NUM_SPARSE_DESCRIPTORS 1

  typedef cusparseHandle_t     MPCusparseHandle;
  typedef cublasHandle_t       MPCublasHandle;
  typedef cusparseMatDescr_t   MPCusparseMatrixDescriptor;
  typedef int                  CusparseInt;
  typedef int                  CublasInt;
  typedef cusparseDnMatDescr_t CusparseDenseMatrixDescriptor;
  typedef cusparseDnMatDescr_t MPCusparseDenseMatrixDescriptor;
  typedef cusparseDnVecDescr_t CusparseDenseVectorDescriptor;
  typedef cusparseDnVecDescr_t MPCusparseDenseVectorDescriptor;
  typedef cusparseSpMatDescr_t CusparseCsrMatrixDescriptor;
  typedef cusparseSpMatDescr_t CudaCsrDescr;

  typedef cusparseSpMatDescr_t MPCusparseCsrMatrixDescriptor;
  typedef int                  CudaInt;
#endif

typedef struct{
  CudaInt n;
  CudaInt nz;
  CudaInt *d_row_pointers;
  CudaInt *d_cols;
  void *d_data;
}MPSparseCsr_Cuda;

union MPSparseCuda{
  MPSparseCsr_Cuda csr;
};

union MPDenseDescriptors
{
  MPCusparseDenseMatrixDescriptor matrix[MAX_NUM_DENSE_DESCRIPTORS];
  MPCusparseDenseVectorDescriptor vector[MAX_NUM_DENSE_DESCRIPTORS];
};

typedef struct MPContextGpuCuda{
  MPCublasHandle cublas_handle;
  MPCusparseHandle cusparse_handle;
  union MPDenseDescriptors dense_descriptors;
  MPCusparseCsrMatrixDescriptor sparse_descriptors[MAX_NUM_SPARSE_DESCRIPTORS];
}MPContextGpuCuda;

typedef struct MPContextCuda{
  MPCublasHandle cublas_handle;
  MPCusparseHandle cusparse_handle;
  union MPDenseDescriptors dense_descriptors;
  MPCusparseCsrMatrixDescriptor sparse_descriptors[MAX_NUM_SPARSE_DESCRIPTORS];
}MPContextCuda;

#endif
