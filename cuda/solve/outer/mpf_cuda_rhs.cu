#include "mpf.h"
#include "mp_cuda.h"

void mpf_cuda_d_generate_B_kernel
(
  int m,
  int n,
  int max_blk_fA,
  int current_rhs,
  int *colorings_array,
  double *B
)
{
  int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
  int i = thread_id%B->m;
  int j = thread_id/B->m;

  ((double*)B->data[B->m*i+j]) =
    ((i+current_rhs)/max_blk_fA == colorings_array[j/max_blk_fA]) &&
      (j%max_blk_fA == (i+current_rhs)%max_blk_fA)*1.0;
}

void mpf_cuda_d_generate_B
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Dense *B
)
{
  CudaInt N = B->m * B->n;
  dim3 cuda_threads_per_block(solver->cuda_nthreads_per_block, 1, 1);
  dim3 cuda_blocks_per_grid(((N+1)/2+cuda_threads_per_block.x-1)/cuda_threads_per_block.x, 1, 1);

  mpf_cuda_d_generate_B_kernel<<<cuda_blocks_per_grid, cuda_threads_per_block>>>
  (
    B->m,
    B->n,
    solver->max_blk_fA,
    solver->current_rhs,
    probe->colorings_array,
    (double*)B->data
  );
}
