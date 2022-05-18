#include "mpf.h"

void mpf_context_alloc
(
  MPF_Context *context
)
{
  mpf_sparse_coo_alloc(&context->A_input);
  mpf_sparse_coo_read(&context->A_input, context->args.filename_A,
    context->typecode_A);
  mpf_sparse_coo_to_csr_convert(&context->A_input, &context->A);
}

void mpf_context_destroy
(
  MPF_ContextHandle context
)
{
  mpf_probe_free(&context->probe);
  mpf_solver_free(&context->solver);

  if ((context->A_input.mem.coo.cols != NULL)
     && (context->A_input.mem.coo.rows != NULL)
     && (context->A_input.mem.coo.data != NULL))
  {
    mpf_sparse_coo_free(&context->A_input);
  }

  if (context->A.handle != NULL)
  {
    mkl_sparse_destroy(context->A.handle);
  }

  if (context->fA.mem.csr.data != NULL)
  {
    mpf_sparse_csr_free(&context->fA);
  }

  mpf_bucket_array_free(&context->solver.color_to_node_map);

  if (context->diag_fA.data != NULL)
  {
    mpf_free(context->diag_fA.data);
    context->diag_fA.data = NULL;
  }

  if (context->solver.mem_defl != NULL)
  {
    mpf_free(context->solver.mem_defl);
  }

  if (context->A_output.mem.coo.cols != NULL)
  {
    mpf_free(context->A_output.mem.coo.cols);
    context->A_output.mem.coo.cols = NULL;
  }

  if (context->A_output.mem.coo.rows != NULL)
  {
    mpf_free(context->A_output.mem.coo.rows);
    context->A_output.mem.coo.rows = NULL;
  }

  if (context->A_output.mem.coo.data != NULL)
  {
    mpf_free(context->A_output.mem.coo.data);
    context->A_output.mem.coo.data = NULL;
  }

  if ((context->solver.outer_type == MPF_SOLVER_BATCH_MATRIX) &&
      (context->heap.fibonacci.m_max > 0))
  {
    mpf_heap_min_fibonacci_internal_free(&context->heap.fibonacci);
  }

  mpf_free(context);
}

MPF_Error mpf_context_create
(
  MPF_Context **context_handle,
  MPF_Int argc,
  char *argv[]
)
{
  /* initialy sets the number of threads to 1 */
  mkl_set_num_threads(1);

  /* allocates context in heap */
  *context_handle = (MPF_Context*)mpf_malloc(sizeof(MPF_Context));
  MPF_Context *context = *context_handle;
  FILE *file_handle = NULL;

  struct timespec start;
  struct timespec finish;

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &start);
  #endif

  /* filenames */
  memset(context->args.filename_A, 0, MPF_MAX_STRING_SIZE);
  memset(context->args.filename_meta, 0, MPF_MAX_STRING_SIZE);
  memset(context->args.filename_fA, 0, MPF_MAX_STRING_SIZE);
  memset(context->typecode_A, 0, MPF_MAX_STRING_SIZE);

  mpf_args_init(&context->args);

  /* input argument parsing  */
  MPF_Int current_index = 1;  /* starting entry of argv array */

  if (argv != NULL)
  {
    /* mode */
    if (strcmp(argv[current_index], "run") == 0)
    {
      context->mode = MPF_MODE_RUN;
      current_index += 1;
    }

    /* input filename */
    if (context->mode == MPF_MODE_RUN)
    {
      if (argv[current_index] != NULL)
      {
        strcpy(context->args.filename_A, argv[current_index]);
        current_index += 1;
      }
    }

    /* output filename */
    if (argv[current_index] != NULL)
    {
      strcpy(context->args.filename_fA, argv[current_index]);
      current_index += 1;
    }

    /* meta log filename */
    if (argv[current_index] != NULL)
    {
      strcpy(context->args.filename_meta, argv[current_index]);
      current_index += 1;
    }

    /* caller .mk file */
    if (argv[current_index] != NULL)
    {
      strcpy(context->args.filename_caller, argv[current_index]);
      current_index += 1;
    }

    /* datatype of matrix A */
    if (strcmp(argv[current_index], "real") == 0)
    {
      mpf_context_set_real(context);
      current_index += 1;
    }
    else if (strcmp(argv[current_index], "complex") == 0)
    {
      mpf_context_set_complex(context);
      current_index += 1;
    }
    else
    {
      return MPF_ERROR_INVALID_ARGUMENT;
    }

    /* layout of matrix A */
    if (strcmp(argv[current_index], "col_major") == 0)
    {
      mpf_context_layout_set(context, MPF_COL_MAJOR);
      current_index += 1;
    }
    else if (strcmp(argv[current_index], "row_major") == 0)
    {
      mpf_context_layout_set(context, MPF_ROW_MAJOR);
      current_index += 1;
    }
    else
    {
      printf("MPF_ERROR: @MPF_CONTEXT_INIT >> Incorrect argument for layout of \
        matrix A\n");
      return MPF_ERROR_INVALID_ARGUMENT;
    }

    /* structure type of matrix A */
    if (strcmp(argv[current_index], "symmetric") == 0)
    {
      mpf_context_matrix_type_set(context, MPF_MATRIX_SYMMETRIC);
      current_index += 1;
    }
    else if (strcmp(argv[current_index], "general") == 0)
    {
      mpf_context_matrix_type_set(context, MPF_MATRIX_GENERAL);
      current_index += 1;
    }
    else if (strcmp(argv[current_index], "hermitian") == 0)
    {
      mpf_context_matrix_type_set(context, MPF_MATRIX_HERMITIAN);
      current_index += 1;
    }
    else
    {
      printf("MPF_ERROR: @MPF_CONTEXT_INIT >> Incorrect argument for structure \
         of matrix A\n");
      return MPF_ERROR_INVALID_ARGUMENT;
    }

    /* read input matrix */
    context->A.matrix_type = context->A_input.matrix_type;
    mpf_read_A(context, context->args.filename_A);

    /* reads degree and blk_fA */
    context->solver.blk_fA = atoi(argv[current_index]);
    current_index+=1;

    /* set output options */
    if (strcmp(argv[current_index], "diag") == 0)
    {
      mpf_context_output_set(context, MPF_DIAG_FA);
    }
    else if (strcmp(argv[current_index], "spai") == 0)
    {
      mpf_context_output_set(context, MPF_SP_FA);
    }
    current_index += 1;

    /* selects probe method for approximating structure of f(A) */
    if ((strcmp(argv[current_index], "sampling")) == 0)
    {
      mpf_pattern_sample_init(
        context,
        atoi(argv[current_index+1]),    /* stride */
        atoi(argv[current_index+2]));   /* num_levels */
      current_index+=3;
    }
    else if ((strcmp(argv[current_index], "avg_path_sampling")) == 0)
    {
      mpf_pattern_multisample_init(
        context,
        atoi(argv[current_index+1]),    /* stride */
        atoi(argv[current_index+2]),    /* num_levels */
        atoi(argv[current_index+3]));   /* num_endpoints */
      current_index+=4;
    }
    else if ((strcmp(argv[current_index], "blocking")) == 0)
    {
      mpf_blocking_init(
        context,
        atoi(argv[current_index+1]),    /* block_size*/
        atoi(argv[current_index+2]));   /* num_levels */
      current_index+=3;
      context->args.n_probe=3;

      context->probe.n_threads = 1;
    }
    else if ((strcmp(argv[current_index], "blocking_hybrid")) == 0)
    {
      //mpf_blocking_hybrid_init(
      //  context,
      //  atoi(argv[current_index+1]),    /* block_size*/
      //  atoi(argv[current_index+2]));   /* num_levels */
      //current_index+=3;
      //context->args.n_probe=3;

      context->probe.n_threads = 1;
    }
    else if ((strcmp(argv[current_index], "batch_blocking")) == 0)
    {
      mpf_blocking_batch_init(
        context,
        atoi(argv[current_index+1]),    /* block_size*/
        atoi(argv[current_index+2]),
        atoi(argv[current_index+3]),    /* batch size */
        atoi(argv[current_index+4]));   /* expansion */
      current_index+=5;
      context->args.n_probe=4;

      context->probe.n_threads = 1;
    }
    else if ((strcmp(argv[current_index], "batch_blocking_compact")) == 0)
    {
      mpf_blocking_batch_coarse_init(
        context,
        atoi(argv[current_index+1]),    /* block_size*/
        atoi(argv[current_index+2]),
        atoi(argv[current_index+3]),
        atoi(argv[current_index+4]));   /* expansion */
      current_index += 5;
      context->args.n_probe = 4;

      context->probe.n_threads = 1;
    }
    else
    {
      return MPF_ERROR_INVALID_ARGUMENT;
    }
  }

  /* reads the framework used for solving the system of equations */
  if (strcmp(argv[current_index], "mpf") == 0)
  {
    context->solver.framework = MPF_SOLVER_FRAME_MPF;
    current_index+=1;
  }
  else if (strcmp(argv[current_index], "gko") == 0)
  {
    context->solver.framework = MPF_SOLVER_FRAME_GKO;
    current_index+=1;
  }
  else
  {
    printf("error in argument %d, passing solver framework incorrectly\n",
      current_index-1);
    return MPF_ERROR_INVALID_ARGUMENT;
  }

  /* outer solver */
  if (strcmp(argv[current_index], "batch") == 0)
  {
    mpf_batch_init(
      context,
      atoi(argv[current_index+1]),  /* batch_size */
      atoi(argv[current_index+2]),
      atoi(argv[current_index+3]));
    current_index += 4;
  }

  context->solver.device = MPF_DEVICE_CPU;  // this has to be parsed
  context->probe.device = MPF_DEVICE_CPU;

  printf("argv[current_index]: %s\n", argv[current_index]);

  /* preconditioning*/
  if (strcmp(argv[current_index], "none") == 0)
  {
    context->solver.precond_type = MPF_PRECOND_NONE;
    current_index += 1;
  }
  else if (strcmp(argv[current_index], "jacobi") == 0)
  {
    context->solver.precond_type = MPF_PRECOND_JACOBI;
    current_index += 1;
  }

  /* deflation */
  if (strcmp(argv[current_index], "none") == 0)
  {
    context->solver.defl_type = MPF_DEFL_NONE;
    current_index += 1;
  }

  /* inner solver */
  if ((context->solver.use_inner) &&
      (context->solver.framework == MPF_SOLVER_FRAME_MPF))
  {
    if (strcmp(argv[current_index], "gmres") == 0)
    {
      mpf_gmres_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index += 4;
    }
    else if (strcmp(argv[current_index], "blk_gmres") == 0)
    {
      mpf_blk_gmres_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index+=4;
    }
    else if (strcmp(argv[current_index], "gbl_gmres") == 0)
    {
      mpf_gbl_gmres_init(
        context,
        atof(argv[current_index+1]),
        atoi(argv[current_index+2]),
        atoi(argv[current_index+3]));
      current_index+=4;
    }
    else if (strcmp(argv[current_index], "lanczos") == 0)
    {
      mpf_lanczos_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* iterations */
        atoi(argv[current_index+3]));       /* restarts */
      current_index += 4;
    }
    else if (strcmp(argv[current_index], "blk_lanczos") == 0)
    {
      mpf_blk_lanczos_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index+=4;
    }
    else if (strcmp(argv[current_index], "gbl_lanczos") == 0)
    {
      mpf_gbl_lanczos_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index+=4;
    }
    else if (strcmp(argv[current_index], "cg") == 0)
    {
      mpf_cg_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]));       /* num_iterations */
      current_index += 3;
    }
    else if (strcmp(argv[current_index], "blk_cg") == 0)
    {
      mpf_blk_cg_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]));       /* num_iterations */
      current_index+=3;
    }
    else if (strcmp(argv[current_index], "gbl_cg") == 0)
    {
      mpf_gbl_cg_init(
        context,
        atof(argv[current_index+1]),
        atoi(argv[current_index+2]));
      current_index+=3;
    }
    else if (strcmp(argv[current_index], "defl_spbasis_cg") == 0)
    {
      mpf_cg_init(
        context,
        atof(argv[current_index]),          /* tolerance */
        atoi(argv[current_index+1]));       /* num_iterations */
      current_index+=2;
    }
    else if (strcmp(argv[current_index], "spbasis_lanczos") == 0)
    {
      mpf_spbasis_lanczos_init(
        context,
        atof(argv[current_index+1]),
        atoi(argv[current_index+2]),
        atoi(argv[current_index+3]));
      current_index += 4;
    }
    else if (strcmp(argv[current_index], "spbasis_blk_lanczos") == 0)
    {
      mpf_spbasis_blk_lanczos_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index+=4;
    }
    else if (strcmp(argv[current_index], "spbasis_gbl_lanczos") == 0)
    {
      mpf_spbasis_gbl_lanczos_init(
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3]));       /* num_restarts */
      current_index += 4;
    }
    else if (strcmp(argv[current_index], "seq_gbl_lanczos") == 0)
    {
      //mpf_seq_gbl_lancos_init
      mpf_spbasis_gbl_lanczos_init
      (
        context,
        atof(argv[current_index+1]),        /* tolerance */
        atoi(argv[current_index+2]),        /* num_iterations */
        atoi(argv[current_index+3])         /* num_restarts */
      );
      current_index += 4;
    }
  }

  #if DEBUG
    mpf_args_printout(&context->args);
  #endif

  #if MPF_MEASURE
    clock_gettime(CLOCK_MONOTONIC, &finish);
    context->runtime_create = mpf_time(start, finish);
  #endif

  return MPF_ERROR_NONE;
}

void mpf_context_set_real
(
  MPF_Context* context
)
{
  context->data_type = MPF_REAL;
  context->A_input.data_type = MPF_REAL;
  context->A.data_type = MPF_REAL;
  
  context->solver.data_type = MPF_REAL;
  context->solver.B.data_type = MPF_REAL;
  context->solver.X.data_type = MPF_REAL;

  context->solver.B.data_type = context->solver.data_type;
  context->solver.X.data_type = context->solver.data_type;
  context->diag_fA.data_type = context->solver.data_type;
}

void mpf_context_set_complex
(
  MPF_Context* context
)
{
  context->data_type = MPF_COMPLEX;
  context->A_input.data_type = MPF_COMPLEX;
  context->A.data_type = MPF_COMPLEX;
  
  mm_initialize_typecode(&context->typecode_A);
  mm_set_complex(&context->typecode_A);
  mm_set_matrix(&context->typecode_A);
  mm_set_coordinate(&context->typecode_A);
  mm_set_general(&context->typecode_A);
  
  /* reads matrix dimensions */
  mpf_sparse_size_read(&context->A_input, context->args.filename_A);
  context->mem_increment = context->A.nz;
  context->A_input.data_type = MPF_COMPLEX;
  context->solver.data_type = MPF_COMPLEX;

  context->A.export_mem_function = &mpf_sparse_z_export_csr_mem;

  context->mem_increment = context->A.nz;
  context->solver.B.m = context->A.m;
  context->solver.X.m = context->A.m;
  context->solver.B.data_type = MPF_COMPLEX;
  context->solver.X.data_type = MPF_COMPLEX;
  context->A_input.data_type = MPF_COMPLEX;

  context->solver.B.data_type = context->solver.data_type;
  context->solver.X.data_type = context->solver.data_type;
  context->diag_fA.data_type = context->solver.data_type;
}

void mpf_context_layout_set
(
  MPF_Context* context,
  MPF_Layout layout
)
{
  context->solver.B.layout = layout;
  context->solver.X.layout = layout;
  context->diag_fA.layout = layout;
}

void mpf_context_matrix_type_set
(
  MPF_Context* context,
  MPF_MatrixType type
)
{
  context->A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  context->A_input.matrix_type = type;
  context->A.matrix_type = type;
  context->solver.matrix_type = type;
}

void mpf_context_output_set
(
  MPF_Context* context,
  MPF_Target output
)
{
  context->solver.recon_target == output;
  if (output == MPF_DIAG_FA)
  {
    context->solver.recon_target = MPF_DIAG_FA;
    context->fA_out = &context->diag_fA;
    context->fA_alloc_function = &mpf_fA_alloc;
    MPF_Dense *fA = (MPF_Dense*)context->fA_out;
    fA->data_type = context->A.data_type;
    fA->m = context->A.m;
    fA->n = context->solver.blk_fA;
  }
  else if (output == MPF_SP_FA)
  {
    context->solver.recon_target = MPF_SP_FA;
    context->fA_out = &context->fA;
    context->fA_alloc_function = &mpf_fA_alloc;

    MPF_Sparse *fA = (MPF_Sparse*)context->fA_out;
    fA->format = context->A.format;
    fA->matrix_type = context->A.matrix_type;
    fA->data_type = context->A.data_type;
    fA->m = context->A.m;
    fA->n = context->A.n;
    fA->nz = context->A.nz;
    context->solver.Pmask = &context->A;
  }

  /* initialize typecode of fA (output) */
  mm_typecode_init(&context->typecode_fA);
  mm_set_matrix(&context->typecode_fA);

  if (context->solver.recon_target == MPF_DIAG_FA)
  {
    mm_set_dense(&context->typecode_fA);
  }
  else if (context->solver.recon_target == MPF_SP_FA)
  {
    mm_set_coordinate(&context->typecode_fA);
  }

  /* data_type_fA (output) */
  if (context->A.data_type == MPF_REAL)
  {
    mm_set_real(&context->typecode_fA);
  }
  else if (context->A.data_type == MPF_COMPLEX)
  {
    mm_set_complex(&context->typecode_fA);
  }
  mm_set_general(&context->typecode_fA);

}

void mpf_bind_fA
(
  MPF_Context* context
)
{
  if (context->solver.recon_target == MPF_DIAG_FA)
  {
    context->solver.recon_target = MPF_DIAG_FA;
    context->fA_out = &context->diag_fA;
    context->fA_alloc_function = &mpf_fA_alloc;
    MPF_Dense *fA = (MPF_Dense*)context->fA_out;
    fA->data_type = context->A.data_type;
    fA->m = context->A.m;
    fA->n = context->solver.blk_fA;
  }
  else if (context->solver.recon_target == MPF_SP_FA)
  {
    context->solver.recon_target = MPF_SP_FA;
    context->fA_out = &context->fA;
    context->fA_alloc_function = &mpf_fA_alloc;

    MPF_Sparse *fA = (MPF_Sparse*)context->fA_out;
    fA->format = context->A.format;
    fA->matrix_type = context->A.matrix_type;
    fA->data_type = context->A.data_type;
    fA->m = context->A.m;
    fA->n = context->A.n;
    fA->nz = context->A.nz;
    context->solver.Pmask = &context->A;
  }

  /* initialize typecode of fA (output) */
  mm_typecode_init(&context->typecode_fA);
  mm_set_matrix(&context->typecode_fA);

  if (context->solver.recon_target == MPF_DIAG_FA)
  {
    mm_set_dense(&context->typecode_fA);
  }
  else if (context->solver.recon_target == MPF_SP_FA)
  {
    mm_set_coordinate(&context->typecode_fA);
  }

  /* data_type_fA (output) */
  if (context->A.data_type == MPF_REAL)
  {
    mm_set_real(&context->typecode_fA);
  }
  else if (context->A.data_type == MPF_COMPLEX)
  {
    mm_set_complex(&context->typecode_fA);
  }
  mm_set_general(&context->typecode_fA);
}

void mpf_context_create
(
  MPF_Context** context,
  MPF_Target output,
  MPF_Int blk_fA
)
{
  *context = (MPF_ContextHandle)mpf_malloc(sizeof(MPF_Context));

  /* initialy sets the number of threads to 1 */
  mkl_set_num_threads(1);

  memset((*context)->args.filename_A, 0, MPF_MAX_STRING_SIZE);
  memset((*context)->args.filename_meta, 0, MPF_MAX_STRING_SIZE);
  memset((*context)->args.filename_fA, 0, MPF_MAX_STRING_SIZE);
  memset((*context)->typecode_A, 0, MPF_IO_CODE_SIZE);

  mpf_args_init(&(*context)->args);
  (*context)->mode = MPF_MODE_RUN;
  (*context)->solver.blk_fA = blk_fA;
  mpf_context_layout_set(*context, MPF_COL_MAJOR);

  /* sets type of output */
  (*context)->solver.recon_target = output;

  mpf_probe_init(*context);
}

void mpf_context_set_input
(
  MPF_ContextHandle context,
  char filename[]
)
{
  strcpy(context->args.filename_A, filename);
}

void mpf_context_set_output
(
  MPF_ContextHandle context,
  char *filename
)
{
  strcpy(context->args.filename_fA, filename);
}

void mpf_context_set_meta
(
  MPF_ContextHandle context,
  char *filename
)
{
  strcpy(context->args.filename_meta, filename);
}

void mpf_context_set_caller
(
  MPF_ContextHandle context,
  char *filename
)
{
  strcpy(context->args.filename_caller, filename);
}

void mpf_read_A
(
  MPF_ContextHandle context,
  char filename[]
)
{
  /* reads dimensions and matrix type of A */
  strcpy(context->args.filename_A, filename);
  mpf_sparse_meta_read(&context->A_input, context->args.filename_A,
    &context->typecode_A);

  if (mm_is_real(context->typecode_A))
  {
    context->A_input.data_type = MPF_REAL;
    context->A.data_type = MPF_REAL;
  }
  else if (mm_is_complex(context->typecode_A))
  {
    context->A_input.data_type = MPF_COMPLEX;
    context->A.data_type = MPF_COMPLEX;
  }

  mpf_bind_A(context);

  /* read input matrix in coo */
  mpf_sparse_coo_alloc(&context->A_input);
  mpf_sparse_coo_read(&context->A_input, context->args.filename_A,
    context->typecode_A);

  if ((context->A_input.descr.type == SPARSE_MATRIX_TYPE_SYMMETRIC) || (context->A_input.descr.type == SPARSE_MATRIX_TYPE_HERMITIAN))
  {
    /* convert input matrix in csr */
    MPF_Sparse A_tmp;
    A_tmp.export_mem_function = mpf_sparse_export_csr;
    A_tmp.data_type = context->A_input.data_type;

    context->A.m = context->A_input.m;
    context->A.n = context->A_input.n;
    context->A.nz = 2*context->A_input.nz - context->A_input.m;
    context->A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    context->A.data_type = context->A_input.data_type;
    context->A.matrix_type = context->A_input.matrix_type;
    context->A.export_mem_function = mpf_sparse_export_csr;

    A_tmp.m = context->A_input.m;
    A_tmp.n = context->A_input.n;
    A_tmp.nz = 2*context->A_input.nz - context->A_input.m;
    A_tmp.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    A_tmp.data_type = context->A_input.data_type;
    A_tmp.matrix_type = context->A_input.matrix_type;
    //A_tmp.export_mem_function = mpf_sparse_export_coo;

    mpf_sparse_coo_alloc(&A_tmp);
    mpf_convert_coo_sy2ge(&context->A_input, &A_tmp);
    mpf_sparse_coo_to_csr_convert(&A_tmp, &context->A);

    mkl_sparse_destroy(A_tmp.handle);
  }
  else if (context->A_input.descr.type == SPARSE_MATRIX_TYPE_GENERAL)
  {
    /* convert input matrix in csr */
    context->A.export_mem_function = mpf_sparse_export_csr;
    mpf_sparse_coo_to_csr_convert(&context->A_input, &context->A);

    context->A.m = context->A_input.m;
    context->A.n = context->A_input.n;
    context->A.nz = context->A_input.nz;
    context->A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    context->A.data_type = context->A_input.data_type;
    context->A.matrix_type = context->A_input.matrix_type;
    context->A.export_mem_function = mpf_sparse_export_csr;
  }

  if (context->A.data_type == MPF_REAL)
  {
    mkl_sparse_d_create_csr(&context->A.handle, INDEXING, context->A.m,
      context->A.n, context->A.mem.csr.rs, context->A.mem.csr.re,
      context->A.mem.csr.cols, (double*)context->A.mem.csr.data);
  }
  else if (context->A.data_type == MPF_COMPLEX)
  {
    mkl_sparse_z_create_csr(&context->A.handle, INDEXING, context->A.m,
      context->A.n, context->A.mem.csr.rs, context->A.mem.csr.re,
      context->A.mem.csr.cols, (MPF_ComplexDouble*)context->A.mem.csr.data);
  }

  mpf_bind_fA(context);
  mpf_bind_solver(context);
  mpf_bind_probe(context);
}

void mpf_bind_solver
(
  MPF_ContextHandle context
)
{
  context->solver.ld = context->A.m;
  context->solver.B.m = context->A.m;
  context->solver.X.m = context->A.m;
}

void mpf_bind_probe
(
  MPF_ContextHandle context
)
{
  context->probe.n_nodes = context->A.m;
  context->probe.m = context->A.m;
}

void mpf_bind_A
(
  MPF_ContextHandle context
)
{
  /* reads matrix dimensions */
  mm_initialize_typecode(&context->typecode_A);
  context->A.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  context->A.m = context->A_input.m;
  context->A.n = context->A_input.n;
  context->A.nz = context->A_input.nz;

  if (context->A_input.data_type == MPF_REAL)
  {
    mpf_context_set_real(context);
    context->A.export_mem_function = &mpf_sparse_d_export_csr_mem;
    context->A_input.export_mem_function = &mpf_sparse_d_export_csr_mem;
    mm_set_real(&context->typecode_A);
  }
  else if (context->A_input.data_type == MPF_COMPLEX)
  {
    mpf_context_set_complex(context);
    context->A.export_mem_function = &mpf_sparse_z_export_csr_mem;
    context->A_input.export_mem_function = &mpf_sparse_z_export_csr_mem;
    mm_set_complex(&context->typecode_A);
  }

  mm_set_matrix(&context->typecode_A);
  mm_set_coordinate(&context->typecode_A);
  mm_set_general(&context->typecode_A);
  context->mem_increment = context->A.nz;
  context->A_input.data_type = context->A.data_type; 
}

/* ---------------------------- I/O functions --------------------------------*/

void mpf_context_read(MPF_Context *context)
{
  FILE *file_handle;
  int ret = 0;
  file_handle = fopen(context->args.filename_meta, "w+");
  /* I/O */
  ret = fscanf(file_handle, "%s\n", (char *) &context->args.filename_A);
  ret = fscanf(file_handle, "%s\n", (char *) &context->args.filename_meta);
  ret = fscanf(file_handle, "%s\n", (char *) &context->args.filename_fA);
  ret = fscanf(file_handle, "%s\n", (char *) &context->typecode_A);
  /* metadata */
  /* memory */
  /* profiling */
  ret = fscanf(file_handle, "%lf\n", &context->probe.runtime_total);
  ret = fscanf(file_handle, "%lf\n", &context->solver.runtime_total);

  /* probing: */
  ret = fscanf(file_handle, "%d\n", &context->probe.expansion_degree);
  ret = fscanf(file_handle, "%d\n", &context->probe.P.m);
  ret = fscanf(file_handle, "%d\n", &context->probe.P.nz);
  ret = fscanf(file_handle, "%d\n", &context->probe.n_colors);
  ret = fscanf(file_handle, "%d\n", &context->probe.n_levels);
  /* numerical */
  /* outer_iterations array */
  ret = fscanf(file_handle, "\n");
  ret = fscanf(file_handle, "%d\n", (int *) &context->solver.inner_type);
  ret = fscanf
  (
    file_handle, "%lf %d %d\n",
    &context->solver.tolerance,
    &context->solver.iterations,
    &context->solver.restarts
  );
  fclose(file_handle);
  printf("returned value: %d\n", ret);
}

void mpf_context_write
(
  MPF_Context *context
)
{
  FILE *file_handle;
  MPF_Int i = 0;
  file_handle = fopen(context->args.filename_meta, "w+");

  /*-----------*/
  /* filenames */
  /*-----------*/
  fprintf(file_handle, "%s\n", context->args.filename_caller);
  fprintf(file_handle, "%s\n", context->args.filename_A);
  fprintf(file_handle, "%s\n", context->args.filename_fA);
  fprintf(file_handle, "%s\n", context->args.filename_meta);

  /*
      probing
  */

  /* MPF_Probing structs are omited since data are replicated in MPF_Context */
  /* @REPLACE(?) potentially */
  fprintf(file_handle, "%d %d\n", context->probe.expansion_degree, context->probe.type);

  /*
     dimensions of P
  */
  fprintf(file_handle, "%d %d %d\n", context->A.m, context->A.n, context->A.nz);
  fprintf(file_handle, "%d %d %d\n", context->probe.P.m, context->probe.P.n, context->probe.P.nz);
  fprintf(file_handle, "%d %d\n", context->probe.n_colors, context->probe.n_levels);
  fprintf(file_handle, "%d %d %d %d %d\n", context->solver.blk_fA, context->probe.stride,
    context->solver.max_blk_fA, context->solver.batch, context->solver.n_max_B);

  /*--------*/
  /* solver */
  /*--------*/

  /* framework */
  fprintf(file_handle, "%d\n", context->solver.framework);

  /* outer/inner type */
  fprintf(file_handle, "%d %d\n",
    context->solver.outer_type,
    context->solver.inner_type);

  /* nthreads */
  fprintf(file_handle, "%d %d\n",
    context->solver.outer_nthreads,
    context->solver.inner_nthreads);

  /* krylov metadata */
  fprintf(file_handle, "%d %d %1.16E\n", context->solver.restarts,
    context->solver.iterations,
    context->solver.tolerance);

  /*
      runtime_information
  */

  fprintf(file_handle, "%1.2E %1.2E %1.2E %1.2E %1.2E\n",
    context->probe.runtime_contract,
    context->probe.runtime_expand,
    context->probe.runtime_color,
    context->probe.runtime_other,
    context->probe.runtime_total);

  fprintf(file_handle, "%1.2E %1.2E %1.2E %1.2E %1.2E %1.2E %1.2E\n",
    context->solver.runtime_pre_process,
    context->solver.runtime_alloc,
    context->solver.runtime_generate_rhs,
    context->solver.runtime_inner,
    context->solver.runtime_reconstruct,
    context->solver.runtime_post_process,
    context->solver.runtime_total);

  /*
      sparse matrix formats
  */
  //fprintf(file_handle, "%d\n", context->format_A);
  //fprintf(file_handle, "%d\n", context->format_Ain);
  //fprintf(file_handle, "%d\n", context->format_fA);
  //fprintf(file_handle, "%d\n", context->format_Agpu);


  /*
      Matrix handles
  */
  /* A_handle is omitted */
  /* A_input_handle is omitted */
  /* fA_handle is omitted */

  /*
      preinitialized descriptors
  */
  /* solver_descriptors_dense_gpu_array is omitted */
  /* cuda_dense_descriptor_array is omitted */

  /*
      Memory
  */

  /* diag_error_vector */
  //for (i = 0; i < context->m_A; ++i)
  //{
  //  fprintf(file_handle, "%e ", ((double *)context->diag_fA_error)[i]);
  //}
  //fprintf(file_handle, "\n");

  ///* memory_outer */
  //if ((context->bytes_outer > 0) && (context->memory_outer != NULL))
  //{
  //  mpf_byte_buffer_write(file_handle, context->bytes_outer,
  //    context->memory_outer);
  //}

  ///* memory_inner */
  //if ((context->bytes_inner > 0) && (context->memory_inner != NULL))
  //{
  //  mpf_byte_buffer_write(file_handle, context->bytes_inner,
  //    context->memory_inner);
  //}

  /* memory_probing */
  //if ((context->bytes_probing > 0) && (context->memory_probing != NULL))
  //{
  //  mpf_byte_buffer_write(file_handle, context->bytes_probing,
  //    context->memory_probing);
  //}

  /* memory_colorings */
  //if ((context->bytes_colorings > 0) && (context->memory_colorings != NULL))
  //{
  //  mpf_byte_buffer_write(file_handle, context->bytes_colorings,
  //    context->memory_colorings);
  //}

  /* memory_pattern is omitted: bytes_memory_pattern is missing */
  /* @TO_BE_ADDED */

  /* memory_buffer is omitted: bytes_memory_buffer is missing */
  /* @TO_BE_ADDED */

  /* memory_temp is omitted: bytes_memory_temp is missing */
  /* @TO_BE_ADDED */

  /* cuda_memory_outer is omitted: not sure how to handle */
  /* @TO_BE_ADDED */

  /* cuda_memory_inner is omitted: not sure how to handle */
  /* @TO_BE_ADDED */

  /* cuda_buffer is omitted: not sure how to handle */
  /* @TO_BE_ADDED */


  /*
      Handles on linear algebra objects
  */

  /* diag_fA */
  //if (context->diag_fA != NULL)
  //{
  //  mpf_double_buffer_write(file_handle, context->m_A, context->diag_fA);
  //}

  ///* diag_fA_exact */
  //if (context->diag_fA_exact != NULL)
  //{
  //  mpf_double_buffer_write(file_handle, context->m_A, context->diag_fA_exact);
  //}

  ///* diag_fA_error */
  //if (context->diag_fA_error != NULL)
  //{
  //  mpf_double_buffer_write(file_handle, context->m_A, context->diag_fA_error);
  //}


  /*
      Auxiliary structures used as buffers for intermediate results
  */

  /* acc is omitted */
  /* buffer is omitted */
  /* pattern_buffer_handle is omitted */
  /* nz_per_level is omitted @TO_BE_ADDED */

  /*
      Cuda context for gpu processing
  */

  /* context gpu is omitted */

  fclose(file_handle);
}

void mpf_write_log
(
  MPF_Context* context,
  char filename_meta[],
  char filename_caller[]
)
{
  strcpy(context->args.filename_meta, filename_meta);
  strcpy(context->args.filename_caller, filename_caller);

  FILE *file_handle;
  MPF_Int i = 0;
  file_handle = fopen(context->args.filename_meta, "w+");

  /*-----------*/
  /* filenames */
  /*-----------*/
  fprintf(file_handle, "%s\n", context->args.filename_caller);
  fprintf(file_handle, "%s\n", context->args.filename_A);
  fprintf(file_handle, "%s\n", context->args.filename_fA);
  fprintf(file_handle, "%s\n", context->args.filename_meta);

  /*
      probing
  */

  /* MPF_Probing structs are omited since data are replicated in MPF_Context */
  /* @REPLACE(?) potentially */
  fprintf(file_handle, "%d %d\n", context->probe.expansion_degree, context->probe.type);

  /*
     dimensions of P
  */
  fprintf(file_handle, "%d %d %d\n", context->A.m, context->A.n, context->A.nz);
  fprintf(file_handle, "%d %d %d\n", context->probe.P.m, context->probe.P.n, context->probe.P.nz);
  fprintf(file_handle, "%d %d\n", context->probe.n_colors, context->probe.n_levels);
  fprintf(file_handle, "%d %d %d %d %d\n", context->solver.blk_fA, context->probe.stride,
    context->solver.max_blk_fA, context->solver.batch, context->solver.n_max_B);

  /*--------*/
  /* solver */
  /*--------*/

  /* framework */
  fprintf(file_handle, "%d\n", context->solver.framework);

  /* outer/inner type */
  fprintf(file_handle, "%d %d\n",
    context->solver.outer_type,
    context->solver.inner_type);

  /* nthreads */
  fprintf(file_handle, "%d %d\n",
    context->solver.outer_nthreads,
    context->solver.inner_nthreads);

  /* krylov metadata */
  fprintf(file_handle, "%d %d %1.16E\n", context->solver.restarts,
    context->solver.iterations,
    context->solver.tolerance);

  /*
      runtime_information
  */

  fprintf(file_handle, "%1.2E %1.2E %1.2E %1.2E %1.2E\n",
    context->probe.runtime_contract,
    context->probe.runtime_expand,
    context->probe.runtime_color,
    context->probe.runtime_other,
    context->probe.runtime_total);

  fprintf(file_handle, "%1.2E %1.2E %1.2E %1.2E %1.2E %1.2E %1.2E\n",
    context->solver.runtime_pre_process,
    context->solver.runtime_alloc,
    context->solver.runtime_generate_rhs,
    context->solver.runtime_inner,
    context->solver.runtime_reconstruct,
    context->solver.runtime_post_process,
    context->solver.runtime_total);

  fclose(file_handle);
}

void mpf_meta_write
(
  MPF_Context *context
)
{
  FILE *file_handle;
  file_handle = fopen(context->args.filename_meta, "w+");

  fprintf(file_handle, "%1.4E\n", context->solver.runtime_total);
  fprintf(file_handle, "%1.4E\n", context->solver.runtime_reconstruct);
  fprintf(file_handle, "%1.4E\n", context->solver.runtime_alloc);
  fprintf(file_handle, "%1.4E\n", context->probe.runtime_color);
  fprintf(file_handle, "%1.4E\n", context->probe.runtime_total);

  fclose(file_handle);
}

void mpf_output_matrix_write
(
  MPF_Context *context,
  MPF_Target source
)
{
  //@NOTE: add controls for formats here in the future
  context->A_output.m = context->fA.m;
  context->A_output.n = context->fA.n;
  context->A_output.nz = context->fA.nz;
  context->A_output.format = context->fA.format;
  context->A_output.descr = context->fA.descr;
  context->A_output.data_type = context->fA.data_type;
  mpf_sparse_coo_alloc(&context->A_output);

  if (source == MPF_SP_FA)
  {
    mpf_sparse_csr_to_coo_convert(&context->fA, &context->A_output);
  }
  mpf_sparse_coo_write(&context->A_output, context->args.filename_fA,
    context->typecode_A);
}

void mpf_diag_write
(
  MPF_Context *context,
  MPF_Dense *fA
)
{
  FILE *file_handle = NULL;
  if ((file_handle = fopen(context->args.filename_fA, "w")) == NULL)
  {
    perror("fopen");
    return;
  }

  if (fA->data_type == MPF_REAL)
  {
    mpf_matrix_d_write(file_handle, context->typecode_fA, (double*)fA->data,
      fA->m, fA->n);
  }
  else if (fA->data_type == MPF_COMPLEX)
  {
    mpf_matrix_z_write(file_handle, context->typecode_fA, (MPF_ComplexDouble*)fA->data,
      context->A.m, context->solver.blk_fA);
  }
}

void mpf_context_write_fA
(
  MPF_ContextHandle mpf_handle
)
{
  printf("mpf_handle->solver.recon_target: %d\n", mpf_handle->solver.recon_target);
  printf("MPF_DIAG_FA: %d\n", MPF_DIAG_FA);
  printf("%d\n", mpf_handle->solver.recon_target);
  printf("%d\n", MPF_DIAG_FA);
  printf("%d\n", MPF_SP_FA);
  if (mpf_handle->solver.recon_target == MPF_DIAG_FA)
  {
    mpf_diag_write(mpf_handle, (MPF_Dense*)mpf_handle->fA_out);
  }
  else if (mpf_handle->solver.recon_target == MPF_SP_FA)
  {
    mpf_output_matrix_write(mpf_handle, MPF_SP_FA);
  }
}

void mpf_write_fA
(
  MPF_ContextHandle mpf_handle,
  char filename[]
)
{
  printf("writting sel(fA) to file: %s\n", filename);

  strcpy(mpf_handle->args.filename_fA, filename);
  if (mpf_handle->solver.recon_target == MPF_DIAG_FA)
  {
    mpf_diag_write(mpf_handle, (MPF_Dense*)mpf_handle->fA_out);
  }
  else if (mpf_handle->solver.recon_target == MPF_SP_FA)
  {
    mpf_output_matrix_write(mpf_handle, MPF_SP_FA);
  }
}
