#include "mpf.h"
void mpf_dense_init
(
  MPF_Dense* A,
  MPF_Int m,
  MPF_Int n,
  MPF_Layout layout
)
{
  A->m = m;
  A->n = n;
  A->layout = layout;
}


void mpf_dense_alloc
(
  void *A_in
)
{
  MPF_Dense *A = (MPF_Dense*) A_in;
  if (A->data_type == MPF_REAL)
  {
    A->data = mpf_malloc(sizeof(double)*A->m*A->n);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    A->data = mpf_malloc(sizeof(MPF_ComplexDouble)*A->m*A->n);
  }
  else if (A->data_type == MPF_INT)
  {
    A->data = mpf_malloc(sizeof(MPF_Int)*A->m*A->n);
  }
}

void mpf_dense_free
(
  MPF_Dense *A
)
{
  mpf_free(A->data);
}

void mpf_d_zeros
(
  MPF_Dense *A
)
{
  mpf_zeros_d_set(A->layout, A->m, A->n, (double*)A->data, A->m);
}

void mpf_z_zeros
(
  MPF_Dense *A
)
{
  mpf_zeros_z_set(A->layout, A->m, A->n, (MPF_ComplexDouble*)A->data, A->m);
}

/* --------------------------  I/O functions ---------------------------------*/

int mpf_dense_read
(
  char *filename_A,
  MPF_Dense* A,
  MPF_Layout layout
)
{
    printf("testing\n");
  A->layout = layout;

  /* reads matrix dimensions */
  MM_typecode matcode;
  mpf_dense_meta_read(A, filename_A, &matcode);

  printf("IN DENSE READ\n");
  printf("%li\n", mm_is_integer(matcode));

  if (mm_is_real(matcode))
  {
    A->data_type = MPF_REAL;
    mpf_dense_alloc(A);

    FILE* file_handle = fopen(filename_A, "r");
    MM_typecode typecode_A;
    mm_read_banner(file_handle, &typecode_A);
    mm_read_mtx_array_size(file_handle, &A->m, &A->n);

    int ret_code;

    if (A->layout == MPF_COL_MAJOR)
    {
      for (MPF_Int i = 0; i < A->n; ++i)
      {
        for (MPF_Int j = 0; j < A->m; ++j)
        {
          ret_code = fscanf(file_handle, "%lf ", &((double*)A->data)[A->m*i+j]);
        }
      }
    }
    else if (A->layout == MPF_ROW_MAJOR)
    {
      for (MPF_Int i = 0; i < A->m; ++i)
      {
        for (MPF_Int j = 0; j < A->n; ++j)
        {
          ret_code = fscanf(file_handle, "%lf ", &((double*)A->data)[A->m*i+j]);
        }
      }
    }
    else
    {
      printf("ERROR: in mpf_matrix_d_read, dimensions missmatch");
      printf("returned value; %d\n", ret_code);
      return MPF_ERROR_INVALID_ARGUMENT;
    }
    fclose(file_handle);
  }
  else if (mm_is_complex(matcode))
  {
    A->data_type = MPF_REAL;
  }
  else if (mm_is_integer(matcode)) {
    printf("---> HERE <---- \n");
    A->data_type = MPF_INT;
    mpf_dense_alloc(A);

    FILE* file_handle = fopen(filename_A, "r");
    MM_typecode typecode_A;
    mm_read_banner(file_handle, &typecode_A);
    mm_read_mtx_array_size(file_handle, &A->m, &A->n);

    int ret_code;
    printf("m: %li, n: %li\n", A->m, A->n);
    if (A->layout == MPF_COL_MAJOR)
    {
      for (MPF_Int i = 0; i < A->n; ++i)
      {
        for (MPF_Int j = 0; j < A->m; ++j)
        {
          ret_code = fscanf(file_handle, "%d ", &((MPF_Int*)A->data)[A->m*i+j]);

          if (j < 10) {
            printf("((MPF_Int*)A->data)[A->m*i+j]: %li\n", ((MPF_Int*)A->data)[A->m*i+j]);
          }
        }
      }
    }
    else if (A->layout == MPF_ROW_MAJOR)
    {
      for (MPF_Int i = 0; i < A->m; ++i)
      {
        for (MPF_Int j = 0; j < A->n; ++j)
        {
          ret_code = fscanf(file_handle, "%li ", &((MPF_Int*)A->data)[A->m*i+j]);
        }
      }
    }
    else
    {
      printf("ERROR: in mpf_matrix_d_read, dimensions missmatch");
      printf("returned value; %d\n", ret_code);
      return MPF_ERROR_INVALID_ARGUMENT;
    }
    fclose(file_handle);
  }

  


  return 0;
}

int mpf_dense_meta_read
(
  MPF_Dense* A,
  char *filename_A,
  MM_typecode* typecode_A
)
{
  FILE *file_handle = NULL;
  if ((file_handle = fopen(filename_A, "r")) == NULL)
  {
    return -1;
  }
  printf("filename_A: %s\n", filename_A);
  mm_read_banner(file_handle, typecode_A);
  mm_read_mtx_array_size(file_handle, &A->m, &A->n);

  fclose(file_handle);
  return 0;
}