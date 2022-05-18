#ifndef MP_CUDA_PROBING
#define MP_CUDA_PROBING

#include "mp.h"

#ifdef __cplusplus
extern "C"
{
#endif

__host__ void mp_heap_min_cuda_fibonacci_internal_allocate
(
  MPHeapMin_Fibonacci *T
);

__device__ void mp_heap_min_cuda_fibonacci_reset
(
  MPHeapMin_Fibonacci *T,
  MPInt m
);

__device__ void mp_heap_min_cuda_fibonacci_init
(
  MPHeapMin_Fibonacci *T,
  MPInt m_max,
  MPInt inc
);

__host__ void mp_heap_min_cuda_fibonacci_internal_free
(
  MPHeapMin_Fibonacci *T
);

__device__ void mp_heap_min_cuda_fibonacci_insert
(
  MPHeapMin_Fibonacci *T,
  MPInt new_key
);

__device__ void mp_heap_min_cuda_fibonacci_defragment
(
  MPHeapMin_Fibonacci *T
);


__device__ void mp_heap_min_cuda_fibonacci_node_move
(
  MPHeapMin_Fibonacci *T,
  MPInt source,
  MPInt dest
);

__device__ MPInt mp_heap_min_cuda_fibonacci_extract_min
(
  MPHeapMin_Fibonacci *T,
  MPInt *return_key
);

__device__ void mp_heap_min_cuda_fibonacci_decrease
(
  MPHeapMin_Fibonacci *T,
  MPInt i,
  MPInt new_key
);

__device__ void mp_heap_min_cuda_fibonacci_delete_min
(
  MPHeapMin_Fibonacci *T
);

typedef struct{
  MPInt m;
  MPInt *m_internal;
  MPInt m_internal_max;
  MPHeapMin_Fibonacci *d_heap_array;
}MPPatternHeap_Cuda;

__host__ void mp_cuda_pattern_heap_allocate
(
  MPPatternHeap_Cuda *T
);

__global__ void mp_cuda_blocking_contract
(
  int n_levels,
  int blk,
  int m,
  int nz,
  int *d_row_pointers,
  int *d_cols,
  int *d_row_pointers_o,
  int *d_cols_o,
  int *temp_array,
  MPPatternHeap_Cuda *T
);

__global__ void mp_cuda_psy_expand_distance_22_symbolic
(
  MPInt P0_m,           /* number of rows of P0 */
  MPInt P0_nz,          /* number of nonzeros of P0 */
  MPInt P1_m,           /* number of rows of P1 */
  MPInt P1_nz,          /* number of nonzeros of P1 */
  MPInt *P0_row_ptr,
  MPInt *P0_cols,
  MPInt *P1_row_ptr,
  MPInt *P1_cols,
  MPInt *temp_array
);

__global__ void mp_cuda_psy_expand_distance_22
(
  MPInt P0_m,
  MPInt P0_nz,
  MPInt P1_m,
  MPInt *P1_nz,
  MPInt *P0_row_ptr,
  MPInt *P0_cols,
  MPInt *P1_row_ptr,
  MPInt *P1_cols,
  MPInt *temp_array
);

__global__ void mp_cuda_blocking_contract_2 /* without swapping in memory (not done already) */
(
  MPInt blk,                  /* block size */
  MPInt m,
  MPInt *P0_d_row_ptr,
  MPInt *P0_d_cols,
  MPInt *P1_d_row_ptr,
  MPInt *P1_d_cols,
  MPInt *temp_array          /* temporary storage array */
);

__host__ void mp_cuda_blocking_psy_fA_2
(
  MPContext *context
);

__global__ void mp_i_prefix_scan
(
  MPInt m,  /* number of entries */
  MPInt *v, /* input vector */
  MPInt *w  /* output vector */
);

__global__ void mp_cuda_blocking_compact
(
  MPInt n,
  MPInt *d_rows_ptr,
  MPInt *d_cols
);

__global__ void mp_i_prefix_scan_uneven
(
  int m_gl,  /* number of entries */
  int *v, /* input vector */
  int *w,  /* output vector */
  int *acc
);

__global__ void mp_partial_sums_join
(
  int n_V,
  int n_acc,
  int *v,
  int *acc
);

__host__ void mp_cuda_d_parsum
(
  int n_threads,  /* n_threads per block */
  int n,
  int *d_v_in,
  int *d_v_out,
  int *d_acc
);

__host__ void mp_cuda_d_parsum_off
(
  int n_threads,  /* n_threads per block */
  int n,
  int *d_v_in,
  int *d_v_out,
  int *d_acc
);

__global__ void mp_i_prefix_scan_uneven_off
(
  int m_gl,  /* number of entries */
  int *v, /* input vector */
  int *w,  /* output vector */
  int *acc
);

__global__ void mp_cuda_blocking_contract_2_symbolic /* without swapping in memory (not done already) */
(
  MPInt blk,                  /* block size */
  MPInt m,
  MPInt *P0_d_row_ptr,
  MPInt *P0_d_cols,
  MPInt *P1_d_row_ptr,
  MPInt *P1_d_cols,
  MPInt *temp_array          /* temporary storage array */
);

#ifdef __cplusplus
}
#endif
#endif
