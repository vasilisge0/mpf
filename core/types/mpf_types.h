#ifndef MPF_TYPES_H /* MPF_TYPES_H.h -- start */
#define MPF_TYPES_H

#include "ginkgo/ginkgo.hpp"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "string.h"
#include "time.h"
#include "stdbool.h"
#include "math.h"
#include "limits.h"
#include "pthread.h"
#include "omp.h"

#include "mpf_mmio.h"
#include "mpf_blas_mkl_internal.h"
#include "mpf_blas.h"

#define MPF_MEASURE 1
#define MPF_PRINTOUT 1

#define TRUE 1
#define FALSE 0

#define MPF_MAX_LINE_LENGTH 250
#define MPF_MAX_STRING_SIZE 250

#define MPF_IO_CODE_SIZE 4

#include "mpf_blas.h"

#ifndef MPF_DEVICE_GPU
    #define MPF_DEVICE_GPU CUDA
#endif

#if MPF_DEVICE_GPU == CUDA
  //#include "mpf_cuda.h"
#endif

typedef char MM_typecode[MM_TYPECODE_SIZE];

typedef struct{
  double probe;
  double total;
  double contract;
  double expand;
  double alloc;
  double coloring;
  double reconstruct;
}MPF_Runtime_Probe;

typedef struct{
  double total;
  double reconstruct;
  double alloc;
  double select;
}MPF_Runtime_Solve;

typedef struct MPF_Args{
  MPF_Int n_total; // 10
  MPF_Int n_outer_solve;
  MPF_Int n_inner_solve;
  MPF_Int n_probe;
  char filename_A[MPF_MAX_STRING_SIZE];
  char filename_M[MPF_MAX_STRING_SIZE];
  char filename_V[MPF_MAX_STRING_SIZE];
  char filename_fA[MPF_MAX_STRING_SIZE];
  char filename_fA_exact[MPF_MAX_STRING_SIZE];
  char filename_meta[MPF_MAX_STRING_SIZE];
  char filename_plot_error[MPF_MAX_STRING_SIZE];
  char filename_caller[MPF_MAX_STRING_SIZE];
}MPF_Args;

typedef enum
{
  MPF_HEAP_UNDEFINED,
  MPF_HEAP_NULL,
  MPF_HEAP_UNMARKED,
  MPF_HEAP_MARKED
}MPF_HEAP_MARK;


typedef enum
{
  MPF_A_INPUT,
  MPF_A,
  MPF_SP_FA,
  MPF_DIAG_FA,
  MPF_M
}MPF_Target;

typedef enum
{
  MPF_BLAS_MKL
}MPF_Blas;

typedef enum
{
  MPF_OPERATION_REDUCTION_ASSIGN,
  MPF_OPERATION_REDUCTION_NUMERIC
}MPF_OPERATION_TYPE;

typedef enum
{
  MPF_PTHREAD_SCHEDULE_STATIC,
  MPF_PTHREAD_SCHEDULE_DYNAMIC
}MPF_PTHREAD_SCHEDULE_TYPE;

typedef enum
{
  MPF_MODE_RUN,
  MPF_MODE_QUIC
}MPF_Mode;

typedef enum
{
  MPF_SOLVER_FRAME_MPF,
  MPF_SOLVER_FRAME_GKO
}MPF_SolverFramework;

#ifndef BLAS
  #define BLAS MPF_BLAS_MKL
  #include "mpf_blas.h"
#endif

#define PHI 1.618033988749895
#define PI  3.141592653589793

#define MPF_SOLVE_MAX_RESTARTS 5

typedef enum
{
  MPF_SOLVER_INNER_UNDEFINED         = 0,
  MPF_SOLVER_DGE_GMRES               = 1,
  MPF_SOLVER_DSY_LANCZOS             = 2,
  MPF_SOLVER_CG0                     = 3,
  MPF_SOLVER_DSY_CG                  = 4,
  MPF_SOLVER_DGE_BLK_GMRES           = 5,
  MPF_SOLVER_DSY_BLK_LANCZOS         = 6,
  MPF_SOLVER_DSY_BLK_CG              = 7,
  MPF_SOLVER_DSY_GBL_CG              = 8,
  MPF_SOLVER_DSY_GBL_PCG             = 9,
  MPF_SOLVER_DGE_GBL_GMRES           = 10,
  MPF_SOLVER_DSY_GBL_LANCZOS         = 11,
  MPF_SOLVER_ZSY_GMRES               = 12,
  MPF_SOLVER_ZSY_LANCZOS             = 13,
  MPF_SOLVER_ZSY_CG                  = 14,
  MPF_SOLVER_ZSY_BLK_GMRES           = 15,
  MPF_SOLVER_ZSY_BLK_LANCZOS         = 16,
  MPF_SOLVER_ZSY_BLK_CG              = 17,
  MPF_SOLVER_ZSY_GBL_GMRES           = 18,
  MPF_SOLVER_ZSY_GBL_LANCZOS         = 19,
  MPF_SOLVER_DSY_BCK_LANCZOS         = 24,
  MPF_SOLVER_DSY_GBL_GMRES           = 26,
  MPF_SOLVER_DSY_BCK_CG              = 28,
  MPF_SOLVER_ZSY_GBL_CG              = 38,
  MPF_SOLVER_ZGE_GMRES               = 40,
  MPF_SOLVER_ZGE_BLK_GMRES           = 41,
  MPF_SOLVER_ZGE_GBL_GMRES           = 42,
  MPF_SOLVER_ZHE_LANCZOS             = 43,
  MPF_SOLVER_ZHE_BLK_LANCZOS         = 44,
  MPF_SOLVER_ZHE_GOAL_LANCZOS        = 45,
  MPF_SOLVER_ZHE_CG                  = 46,
  MPF_SOLVER_ZHE_BLK_CG              = 47,
  MPF_SOLVER_ZHE_GBL_CG              = 48,
  MPF_SOLVER_ZHE_BLK_GMRES           = 50,
  MPF_SOLVER_ZHE_GBL_LANCZOS         = 53,
  MPF_SOLVER_DSY_SPBASIS_LANCZOS     = 57,
  MPF_SOLVER_DSY_SPBASIS_BLK_LANCZOS = 58,
  MPF_SOLVER_DSY_SPBASIS_GBL_LANCZOS = 59,
  MPF_SOLVER_ZSY_SPBASIS_LANCZOS     = 60,
  MPF_SOLVER_ZSY_SPBASIS_BLK_LANCZOS = 61,
  MPF_SOLVER_ZSY_SPBASIS_GBL_LANCZOS = 62,
  MPF_SOLVER_ZHE_SPBASIS_LANCZOS     = 63,
  MPF_SOLVER_ZHE_SPBASIS_BLK_LANCZOS = 64,
  MPF_SOLVER_ZHE_SPBASIS_GBL_LANCZOS = 65,
  MPF_SOLVER_DSY_CHEB                = 66,
  MPF_SOLVER_GKO                     = 67
}MPF_SolverType;

typedef enum{
  MPF_ERROR_NONE,
  MPF_ERROR_INVALID_ARGUMENT
}MPF_Error;

typedef enum
{
  MPF_APPROX_UNDEFINED,
  MPF_APPROX_DIAG,
  MPF_APPROX_MATRIX
}MPF_ApproxType;

typedef enum
{
  MPF_MATRIX_GENERAL,
  MPF_MATRIX_SYMMETRIC,
  MPF_MATRIX_HERMITIAN
}MPF_MatrixType;

typedef enum
{
  MPF_PROBE_UNDEFINED,
  MPF_PROBE_BLOCKING,
  MPF_PROBE_SAMPLING,
  MPF_PROBE_PATH_SAMPLING,
  MPF_PROBE_AVG_PATH_SAMPLING,
  MPF_PROBE_BLOCKING_PTHREAD,
  MPF_PROBE_BLOCKING_CUDA,
  MPF_PROBE_BATCH_BLOCKING,
  MPF_PROBE_BATCH_COMPACT_BLOCKING,
  MPF_PROBE_BLOCKING_REORDER
}MPF_ProbeType;

typedef enum
{
  MPF_DEVICE_UNDEFINED,
  MPF_DEVICE_CPU,
  MPF_DEVICE_CUDA
}MPF_DeviceType;

typedef enum
{
  MPF_DATATYPE_UNDEFINED,
  MPF_REAL,
  MPF_COMPLEX,
  MPF_COMPLEX_HERMITIAN,
  MPF_REAL_64,
  MPF_REAL_32,
  MPF_COMPLEX_64,
  MPF_COMPLEX_32,
  MPF_COMPLEX_SYMMETRIC,
  MPF_INT
}MPF_DataType;

typedef enum
{
  MPF_SOLVER_BATCH_UNDEFINED,
  MPF_SOLVER_BATCH,
  MPF_SOLVER_BATCH_2PASS,
  MPF_SOLVER_BATCH_CHEB,
  MPF_SOLVER_BATCH_CUDA,
  MPF_SOLVER_BATCH_LS,
  MPF_SOLVER_BATCH_BLK_LS,
  MPF_SOLVER_BATCH_GLB_LS,
  MPF_SOLVER_BATCH_LS_DIAG_BLK,
  MPF_SOLVER_BATCH_MATRIX,
  MPF_SOLVER_BATCH_PTHREADS,
  MPF_SOLVER_BATCH_OPENMPF,
  MPF_SOLVER_BATCH_SPBASIS,
  MPF_SOLVER_BATCH_DEFL,
  MPF_SOLVER_BATCH_DEFL_SEED
}MPF_SolverOuterType;

typedef enum
{
  MPF_DEFL_NONE,
  MPF_DEFL_RECYCLE
}MPF_DeflType;

typedef enum{
  MPF_IOWrite,
  MPF_IOAppend
}MPF_IOType;

/* == solvers == */

typedef enum
{
  MPF_SPARSE_COO,
  MPF_SPARSE_CSR
}MPF_MemFormat;

typedef struct{
  pthread_t pthread_id;
  MPF_Int mpf_thread_id;
  char *argv;
}MPF_PthreadInfo;

typedef struct MPF_LinkedList{
  MPF_Int max_n_entries;
  MPF_Int n_entries;
  MPF_Int start;
  MPF_Int end;
  MPF_Int end_internal;
  MPF_Int *id;
  MPF_Int *next;
}MPF_LinkedList;

typedef struct{
  MPF_Int n_iterations_array[MPF_SOLVE_MAX_RESTARTS];
  MPF_Int n_restarts;
  MPF_Int *residuals_array;
}MPF_SolverInfo;

typedef struct MPF_BucketArray{
  MPF_Int *bins_start;
  MPF_Int *bins_end;
  MPF_Int *bins_size;
  MPF_Int *values;
  MPF_Int *next;
  MPF_Int n_bins;
  MPF_Int max_n_bins;
  MPF_Int n_values;
  MPF_Int max_n_values;
  MPF_Int max_bin_size;
  MPF_Int values_mem_increment;
  MPF_Int bins_mem_increment;
}MPF_BucketArray;

typedef struct MPF_HeapMin_Fibonacci{
/*==============================================================================

parent[]

     |0|1,2|3,4,5,6|7,8,9,10,11,12,13,14,15| <- indicate positions in child and
                                                parent arrays.
key: |-|-,-|-,-,-,-|-,-,-,- ,- ,- ,- ,- ,- |

==============================================================================*/

  /* dimensions */
  MPF_DataType data_type;
  MPF_Int n_lost; /* refers to the the number of mem locations that there is */
                  /* no way to access, you should call refragmatation when   */
                  /* they become a alot                                      */

  MPF_Int mem_increment;
  MPF_Int n_roots;
  MPF_Int n_null;
  MPF_Int m;      /* number of nodes */
  MPF_Int m_max;  /* maximum number of nodes */
  MPF_Int deg_mark_length;

  /* indices*/
  MPF_Int root_first;
  MPF_Int root_last;
  MPF_Int root_new;
  MPF_Int min_index;

  /* array containing degrees */
  MPF_Int *deg_mark;

  /* essential pointers */
  MPF_Int *parent;
  MPF_Int *previous;
  MPF_Int *next;
  MPF_Int *child;

  /* */
  MPF_Int *mark;
  MPF_Int *deg;
  MPF_Int *key;
  MPF_Int *map1;  /* assigned to an external array, to be used for */
                  /* comparisons instead of processing keys        */
  MPF_Int *map2;
}MPF_HeapMin_Fibonacci;

union MPF_HeapMin{
  MPF_HeapMin_Fibonacci fibonacci;
};


/* ----------------------- matrices in sparse format ------------------------ */

typedef struct{
  MPF_Int *cols;
  MPF_Int *rs;
  MPF_Int *re;
  void *data;
}MPF_SparseCsr;

typedef struct{
  MPF_Int *rows;
  MPF_Int *cols_start;
  MPF_Int *cols_end;
  void *data;
}MPF_SparseCsc;

typedef struct{
  MPF_Int *rows;
  MPF_Int *cols;
  void *data;
}MPF_SparseCoo;

typedef struct{
  CusparseInt *d_row_pointers;
  CusparseInt *d_cols;
  void *d_data;
}MPF_Cuda_SparseCsr;

typedef struct{
  MPF_FunctionPtr update_X;
  MPF_FunctionPtr rec_X;
  MPF_FunctionPtr update_B;
  MPF_FunctionPtr seed;
}MPF_DeflOperands;

union MPF_SparseMem{
  MPF_SparseCsr csr;
  MPF_SparseCsc csc;
  MPF_SparseCoo coo;
};

typedef struct MPF_Sparse{
  MPF_MemFormat format;
  sparse_index_base_t index;
  MPF_Int m;
  MPF_Int n;
  MPF_Int nz;
  MPF_Int nz_max;
  void (*export_mem_function)(MPF_Sparse *);
  MPF_DataType data_type;
  MPF_MatrixType matrix_type;
  MPF_SparseDescr descr;
  MPF_SparseHandle handle;  /* used from sparse blas libraries */
  union MPF_SparseMem mem;
}MPF_Sparse;

typedef struct{
  MPF_Layout layout;
  MPF_DataType data_type;
  MPF_Int m;
  MPF_Int n;
  void *data;
  gko::matrix::Dense<> *gko_obj;
}MPF_Dense;

/* ------------------------------- Probe ------------------------------------ */

typedef struct MPF_Probe{
  MPF_ProbeType type;
  MPF_DeviceType device;
  MPF_Int m;
  MPF_Int stride;
  MPF_Int max_blk;
  MPF_Int batch;
  MPF_Int n_nodes;
  MPF_Int n_colors;
  MPF_Int n_endpoints;
  MPF_Int n_levels;
  MPF_Int expansion_degree;
  MPF_Int iterations;
  MPF_Int n_threads;
  MPF_Int offset_rows;
  MPF_Int offset_cols;
  MPF_Int bytes_colorings;
  MPF_Int bytes_buffer;
  MPF_Sparse P;
  MPF_Int *colorings_array;
  MPF_Int *d_colorings_array;
  MPF_Int *mappings_array;
  MPF_Int *endpoints_array;
  void *buffer;
  MPF_Int *nz_per_level;
  //MPF_Runtime_Probe runtime;
  double runtime_total;
  double runtime_contract;
  double runtime_expand;
  double runtime_color;
  double runtime_other;
  MPF_Int current_iteration;

  void (*alloc_function)(MPF_Probe *);
  void (*find_pattern_function)(MPF_Probe *, MPF_Sparse *);
  void (*color_function)(MPF_Probe *);
}MPF_Probe;

/* --------------------------- preconditioning -------------------------------*/

typedef enum{
  MPF_PRECOND_NONE,
  MPF_PRECOND_SPAI,
  MPF_PRECOND_JACOBI,
  MPF_PRECOND_EYE,
  MPF_PRECOND_BLK_DIAG
}MPF_PrecondType;

typedef enum{
  MPF_PRECOND_STATIC,
  MPF_PRECOND_ADAPTIVE
}MPF_PrecondUpdate;

typedef struct{
  MPF_Sparse M_input;   /* preconditioner matrix */
//union MPF_SparseCuda M_gpu; /* preconditioner matrix */
  MPF_FunctionPtr precond_func;
  MPF_Int n_shifts;
  MPF_Int *shifts_array;
}KrylovMeta;

typedef struct{
  pthread_t pthread_id;
  MPF_Int mpf_thread_id;
  MPF_Layout layout_B;
  KrylovMeta meta;
  MPF_Int m_B;
  MPF_Int blk;
  MPF_Int blk_fA;
  MPF_Int blk_max_fA;
  MPF_Int n_blocks;
  MPF_Int n_threads;
  MPF_Int n_max_B;
  MPF_SparseDescr A_descr;
  MPF_SparseHandle A_handle;
  MPF_FunctionPtr solve_func;
  void *mem_colorings;
  void *B;
  void *X;
  void *buffer;
  void *diag_fA;
  void *mem_inner;
}MPF_PthreadInputsSolver;

typedef struct MPF_Solver
{
  MPF_MatrixType matrix_type;
  MPF_DataType data_type;
  MPF_SolverFramework framework;
  MPF_Int current_rhs;
  MPF_Int current_batch;
  MPF_BucketArray color_to_node_map;

  MPF_Int ld;
  MPF_Int n_shifts;
  MPF_DeviceType device;

  MPF_Int blk_fA;
  MPF_Int max_blk_fA;
  MPF_Int n_max_B;

  MPF_Int outer_nthreads;
  MPF_Int inner_nthreads;

  MPF_Int use_defl;
  MPF_Int use_precond;

  MPF_DeflType defl_type;

  void (*inner_get_mem_size_function)(MPF_Solver*);
  void (*generate_rhs_function)(MPF_Probe *, MPF_Solver *, MPF_Dense *);
  void (*generate_initial_solution_function)(MPF_Dense *);
  void (*reconstruct_function)(MPF_Probe *, MPF_Solver *);
  void (*precond_apply_function)(MPF_Solver*, double*, double*);
  void (*precond_generate_function)(MPF_Solver*, MPF_Sparse*);
  void (*precond_alloc_function)(MPF_Solver*);
  void (*precond_free_function)(MPF_Solver*);
  void (*defl_apply_function)(MPF_Solver*);
  void (*defl_update_function)(MPF_Solver*);
  void (*defl_alloc_function)(MPF_Solver*);
  void (*defl_free_function)(MPF_Solver*);

  void (*outer_function)(MPF_Probe *, MPF_Solver *, MPF_Sparse *, void *);
  void (*inner_function)(MPF_Solver *, MPF_Sparse *A, MPF_Dense *B, MPF_Dense *X);
  void (*inner_call_function)(MPF_Solver *, MPF_Sparse *A);
  void (*pre_process_function)(MPF_Probe *, MPF_Solver *);
  void (*post_process_function)(MPF_Probe *, MPF_Solver *, void *);

  void (*precond_update_function)(MPF_Probe *, MPF_Solver *);

  /* allocation functions */
  void (*outer_alloc_function)(MPF_Solver *);
  void (*inner_alloc_function)(MPF_Solver *);

  /* deallocation functions */
  void (*outer_free_function)(MPF_Solver *);
  void (*inner_free_function)(MPF_Solver *);

  MPF_Target recon_target;

  /* threading */
  MPF_Int n_threads_pthreads;
  MPF_Int n_threads_omp;

  /* preconditioning */
  MPF_Sparse M;

  MPF_PrecondType precond_type;
  MPF_PrecondUpdate precond_update;

  /* deflation */
  MPF_FunctionPtr defl_X_func;
  MPF_FunctionPtr defl_B_func;
  MPF_FunctionPtr defl_get_mem_func;
  MPF_FunctionPtr defl_alloc_func;
  MPF_Int bytes_defl;
  void *mem_defl;
  void *Vdefl;
  double *shifts_array;

  /* solver types */
  MPF_SolverOuterType outer_type;
  MPF_FunctionPtr outer;
  MPF_SolverType  inner_type;
  MPF_FunctionPtr inner;

  /* Chebyshev polynomials */
  MPF_Int cheb_M;
  MPF_Int cheb_ev_iterations;
  double A_lmin;
  double A_lmax;
  MPF_FunctionPtr target_func;
  MPF_FunctionPtr solver_ev_max_func;
  MPF_FunctionPtr solver_ev_min_func;
  double *cheb_coeffs;
  double *cheb_fapprox;
  MPF_Int defl;
  MPF_Int defl_n_ev_max;
  MPF_Int outer_bytes;
  MPF_Int inner_bytes;
  MPF_Int n_defl;

  /* memory used by inner/outer solvers */
  void *outer_mem;
  void *inner_mem;
  void *defl_mem;
  void *outer_mem_cuda;
  void *inner_mem_cuda;
  void *buffer;
  void *buffer_cuda;
  void *ev_max_mem;
  void *ev_min_mem;
  MPF_Int iterations_ev;

  MPF_Int use_inner;

  /* */
  MPF_Int batch;
  MPF_Int n_batches;
  MPF_Int restarts;
  MPF_Int iterations;
  double tolerance;

  void *fA;
  MPF_Sparse *Pmask;

  //MPF_Runtime_Solve runtime;
  double runtime_total;
  double runtime_alloc;
  double runtime_pre_process;
  double runtime_generate_rhs;
  double runtime_inner;
  double runtime_select;
  double runtime_reconstruct;
  double runtime_post_process;
  double runtime_other;
  double runtime_precond;
  double runtime_defl;

  /* linear algebra objects */
  MPF_Dense B;
  MPF_Dense X;
  void *A;

  std::unique_ptr<gko::matrix::Dense<>,
      std::default_delete<gko::matrix::Dense<>>> B_gko;
  std::unique_ptr<gko::matrix::Dense<>,
      std::default_delete<gko::matrix::Dense<>>> X_gko;
  std::shared_ptr<const gko::Executor> exec;

}MPF_Solver;

#define MPF_PROBING_BLOCKING_PATTERN_LENGTH 4
#define MPF_MULTILEVEL_SAMPLING_PATTERN_LENGTH 4
#define MPF_MULTIPATH_SAMPLING_PATTERN_LENGTH 4
#define MPF_AVERAGE_MULTIPATH_SAMPLING_PATTERN_LENGTH 6
#define MAX_NUM_DENSE_DESCRIPTORS 10
#define MAX_NUM_SPARSE_DESCRIPTORS 1
#define N_INPUT_BLOCKS 6

typedef struct MPF_Context{

  MPF_Args args;
  MPF_Mode mode;

  /* @RENAME _vecblks part */
  MPF_Int n_defl; /* used by deflated cg not necessarily equal to */
                  /* n_iterations (<= n_iterations)               */
  MPF_ApproxType output_type;

  /* Used for combinatorial algorithms */
  union MPF_HeapMin heap;

  /* probing */
  void (*fA_alloc_function)(MPF_Context *);
  MPF_Probe probe;

  /* solver */
  MPF_Solver solver;

  /* mem management metadata */
  MPF_DataType data_type;
  MPF_Int      mem_increment;

  /* I/O codes */
  char typecode_B[MPF_IO_CODE_SIZE];
  char typecode_A[MPF_IO_CODE_SIZE];
  char typecode_M[MPF_IO_CODE_SIZE];
  char typecode_V[MPF_IO_CODE_SIZE];
  char typecode_fA[MPF_IO_CODE_SIZE];

  /* block sizes for various tasks */
  MPF_Int blk_fA;
  MPF_Int blk_max_fA;

  /* runtime information */
  MPF_Runtime_Probe runtime;
  double runtime_create;

  /* Matrices A and fA */
  MPF_Sparse A_input;
  MPF_Sparse A_output;
  MPF_Sparse A;
  MPF_Sparse fA;
  gko::matrix::Coo<> A_in_gko;
  gko::matrix::Csr<> A_gko;
  MPF_Dense diag_fA;
  void *fA_out;
}MPF_Context;

typedef struct{
  pthread_t thread_id;
  //long id;
  pthread_attr_t attributes;
  MPF_Context *shared_context;
}MPF_ContextPthreads;

typedef struct{

  long thread_id;
  pthread_attr_t attributes;

  MPF_Solver *shared_context;

  MPF_Int remainder_batches;
  MPF_Int remainder_blocks;
  MPF_Int block_size;
  MPF_Int batch_size;
  MPF_Int n_batches_per_thread; /* @NOT_USED */
  MPF_Int n_blocks;
  MPF_Int n_batches;

}MPF_SolverOuter_Pthreads;

typedef struct{
  //pthread_t thread_id;
  long thread_id;

  MPF_Context *shared_context;

  MPF_Int remainder_batches;
  MPF_Int remainder_blocks;
  MPF_Int block_size;
  MPF_Int batch_size;
  MPF_Int n_batches_per_thread;

  void *B;
  void *X;

  MPF_Int n_blocks;
  MPF_Int n_batches;
}MPF_SolverOuterOpenmp;

typedef struct{
  double tolerance;
  double *preconditioner_matrix;
  double *preconditioner_temp;
  MPF_Int max_n_iterations;
  void (*func)();
  MPF_Int n_iterations_completed;
  double norm_residual_achieved;
}MPF_SolverCgMeta;


/* == data types == */

typedef struct{
  MPF_Int *cols;
  MPF_Int *rs;
  MPF_Int *re;
  double *values;
  void *internal;
}MPF_SparseCsr_d;

typedef struct {
  char code[MM_TYPECODE_SIZE];
  char filename[MPF_MAX_STRING_SIZE];
}MPF_SparseIo;

typedef struct {
  char filename[MPF_MAX_STRING_SIZE];
  MPF_DataType type;
}MPF_MatrixDenseIo;

/* ----------------------- fibonacci heap functions --------------------------*/

void mpf_heap_min_fibonacci_internal_alloc
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_heap_min_fibonacci_init
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int max_n_nodes,
  MPF_Int mem_inc
);

void mpf_heap_min_fibonacci_internal_reallocate
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int increment
);

void mpf_heap_min_fibonacci_internal_free
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_heap_min_fibonacci_insert
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int new_key
);

MPF_Int mpf_heap_min_fibonacci_extract_min
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int *return_key
);

void mpf_heap_min_fibonacci_delete_min
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_heap_min_fibonacci_view_roots
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_heap_min_fibonacci_plot_roots
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_heap_min_fibonacci_decrease
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int i,
  MPF_Int new_key
);

typedef MPF_Context* MPF_ContextHandle;

void mpf_run
(
  MPF_ContextHandle context
);

int mpf_data_init
(
  int argc,
  char **argv
);

void mpf
(
  int argc,
  char **argv
);

void mpf_heap_min_fibonacci_reset
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m
);

void mpf_heap_min_fibonacci_node_move
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int source,
  MPF_Int dest
);

void mpf_heap_min_fibonacci_defragment
(
  MPF_HeapMin_Fibonacci *T
);

void mpf_i_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v
);

void mpf_d_id_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v,
  double *v_data,
  double *buffer
);

void mpf_z_id_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v,
  MPF_ComplexDouble *v_data,
  MPF_ComplexDouble *buffer
);

typedef MPF_Context* MPF_ContextHandle;


int elapsed
(
  double *sec
);

typedef struct{
  MPF_Int nz;
  MPF_HeapMin_Fibonacci *heap; /* used for loop scheduling */
  MPF_Int max_priority_id;
  MPF_Int *signals_continue;
  //MPF_HeapMin_Fibonacci *heap_merge;
}MPF_PthreadShared_Probing;

typedef struct{

  pthread_t pthread_id;
  MPF_Int mpf_thread_id;
  MPF_Int mpf_job_id;

  MPF_Layout layout_B;
  KrylovMeta meta;
  MPF_Int m_B;
  MPF_Int nz;
  MPF_Int blk;
  MPF_Int blk_fA;
  MPF_Int blk_max_fA;
  MPF_Int n_blocks;
  MPF_Int n_threads;
  MPF_Int n_max_B;
  MPF_SparseDescr A_descr;
  MPF_SparseHandle A_handle;
  MPF_FuncPtr solve_func;
  MPF_Int mem_inc;

  MPF_PthreadShared_Probing *shared;
  void *mem_colorings;
  void *B;
  void *X;
  void *buffer;
  void *diag_fA;
  void *mem_inner;
  void *buffer_cols;

  MPF_HeapMin_Fibonacci heap; /* used for loop scheduling */

  MPF_Int *tempf_array;
  MPF_Int *tempf_inverted_array;
  MPF_Int *tempf_cols;

  MPF_Int m_P;
  MPF_Int range_start;
  MPF_Int range_end;
  MPF_Int blk_threads_probing;
  MPF_Int current_block;
  MPF_Int n_levels;
}MPF_PthreadContext_Probing;

typedef struct{
  MPF_Int m_array;
}MPF_PthreadContextShared_Coloring;

typedef struct{

  pthread_t pthread_id;
  MPF_Int thread_id;
  MPF_Int job_id;

  MPF_Int m;
  MPF_Int n_threads;

  MPF_LinkedList list;
  MPF_HeapMin_Fibonacci heap; /* used for loop scheduling */
  MPF_Int *shared_colorings;

  MPF_PthreadContextShared_Coloring *shared_scalars;
  MPF_Int *shared_array;
  MPF_Int *shared_array_m;
  MPF_Sparse *shared_P;
  MPF_Int max_coloring;

}MPF_PthreadContext_Coloring;

//changed for cuda
typedef struct{
  MPF_Int m;
  MPF_Int nz;
  MPF_Int nz_max;
  MPF_Int *d_row_pointers;
  MPF_Int *d_cols;
}MPF_PatternCsr_Cuda;

void mpf_sparse_d_csr_alloc
(
  MPF_Sparse *A
);

void mpf_sparse_csr_free
(
  MPF_Sparse *A
);

void mpf_sparse_coo_free
(
  MPF_Sparse *A
);

//original
//typedef struct{
//  int m;
//  int nz;
//  int nz_max;
//  int *d_row_pointers;
//  int *d_cols;
//}MPF_PatternCsr_Cuda;

void mpf_diag_fA_average
(
  MPF_Context *context
);

void mpf_args_init
(
  MPF_Args *args
);

void mpf_args_printout
(
  MPF_Args *args
);

void mpf_sparse_coo_d_threshold_apply
(
  MPF_Int blk_r,
  MPF_Int blk_c,
  MPF_Int I_r,
  MPF_Int I_c,
  MPF_Int m_V,
  MPF_Int n_V,
  double *M,
  MPF_Int ld_M,
  double threshold,
  MPF_Sparse *A,  /* requires coo format */
  MPF_Int memory_inc
);

void mpf_sparse_csr_alloc
(
  MPF_Sparse *A_in
);

void mpf_sparse_csr_to_coo_convert
(
  MPF_Sparse *P_in,
  MPF_Sparse *P_out
);

void mpf_sparse_coo_to_csr_convert
(
  MPF_Sparse *A,
  MPF_Sparse *B
);

void mpf_sparse_d_export_csr_mem
(
  MPF_Sparse *A
);

void mpf_sparse_z_export_csr_mem
(
  MPF_Sparse *A
);

void mpf_sparse_csr_get_nz
(
  MPF_Sparse *A
);

MPF_Int mpf_sparse_csr_get_max_row_nz
(
  MPF_Sparse *A
);

void mpf_d_zeros
(
  MPF_Dense *A
);

void mpf_z_zeros
(
  MPF_Dense *A
);

void mpf_sparse_export_csr
(
  MPF_Sparse *A
);

void mpf_sparse_copy_meta
(
  MPF_Sparse *A,
  MPF_Sparse *B
);

void mpf_sparse_d_copy
(
  MPF_Int start_A,
  MPF_Int end_A,
  MPF_Sparse *A,
  MPF_Int start_B,
  MPF_Int end_B,
  MPF_Sparse *B
);

void mpf_sparse_d_eye
(
  MPF_Sparse *A
);

void mpf_bucket_array_find_min
(
  MPF_BucketArray *H,
  MPF_Int start,
  MPF_Int bin,
  MPF_Int *nentries,
  MPF_Int *min
);

void mpf_context_set_real
(
  MPF_Context* context
);

void mpf_context_set_complex
(
  MPF_Context* context
);

void mpf_context_layout_set
(
  MPF_Context* context,
  MPF_Layout layout
);

void mpf_context_matrix_type_set
(
  MPF_Context* context,
  MPF_MatrixType type
);

void mpf_context_output_set
(
  MPF_Context* context,
  MPF_Target output
);

void mpf_context_create
(
  MPF_Context** context,
  MPF_Target output,
  MPF_Int blk_fA
);

void mpf_context_set_input
(
  MPF_ContextHandle context,
  char filename[]
);

void mpf_context_set_output
(
  MPF_ContextHandle context,
  char *filename
);

void mpf_context_set_meta
(
  MPF_ContextHandle context,
  char *filename
);

void mpf_context_set_caller
(
  MPF_ContextHandle context,
  char *filename
);

void mpf_read_A
(
  MPF_ContextHandle context,
  char filename[]
);

void mpf_bind_solver
(
  MPF_ContextHandle context
);

void mpf_bind_probe
(
  MPF_ContextHandle context
);

void mpf_bind_A
(
  MPF_ContextHandle context
);

void mpf_bind_fA
(
  MPF_Context* context
);

void mpf_convert_csr_sy2ge
(
  MPF_Sparse* A,
  MPF_Sparse* B
);

void mpf_sparse_debug_write
(
  MPF_Sparse* A,
  char filename_A[100]
);

void mpf_sparse_debug_write
(
  MPF_Sparse* A,
  char filename_A[100]
);

void mpf_convert_coo_sy2ge
(
  MPF_Sparse* A,
  MPF_Sparse* B
);

void mpf_dense_init
(
  MPF_Dense* A,
  MPF_Int m,
  MPF_Int n,
  MPF_Layout layout
);

int mpf_dense_meta_read
(
  MPF_Dense *A,
  char *filename_A,
  MM_typecode* typecode_A
);

int mpf_dense_read
(
  char *filename_A,
  MPF_Dense *A,
  MPF_Layout layout
);

void mpf_dense_free
(
  MPF_Dense *A
);

void mpf_sparse_read
(
  char filename[],
  MPF_Sparse* A
);

int mpf_matrix_d_write_2
(
  const char *filename,
  const double *handle,
  const MPF_Int m,
  const MPF_Int n,
  MPF_IOType io_type
);

#endif /* MPF_H -- end */

