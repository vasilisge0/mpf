#include "mpf.h"

/*=========================================*/
/*== dense matrix manipulation functions ==*/
/*=========================================*/

void mpf_zeros_d_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  double *A,
  const MPF_Int ld_A
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = 0.0;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; i++)
    {
      for (int j = 0; j < n_A; j++)
      {
        A[ld_A*i+j] = 0.0;
      }
    }
  }
}

void mpf_ones_d_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  double *A,
  const MPF_Int ld_A
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = 1.0;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; i++)
    {
      for (int j = 0; j < n_A; j++)
      {
        A[ld_A*i+j] = 1.0;
      }
    }
  }
}


void mpf_zeros_s_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  float *A,
  const MPF_Int ld_A
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = 1.0;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = 1.0;
      }
    }
  }
}

void mpf_zeros_z_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_ComplexDouble *A,
  const MPF_Int ld_A
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = ZERO_C;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = ZERO_C;
      }
    }
  }
}

void mpf_zeros_c_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Complex *A,
  const MPF_Int ld_A
)
{
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);

  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = ZERO_C;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = ZERO_C;
      }
    }
  }
}

void mpf_zeros_i_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Int *A,
  const MPF_Int ld_A
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = 0;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = 0;
      }
    }
  }
}

void mpf_matrix_i_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_Int *A,
  const MPF_Int ld_A,
  const MPF_Int val
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        //printf("ld_A*i+j: %d\n", ld_A*i+j);
        A[ld_A*i+j] = val;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = val;
      }
    }
  }
}

void mpf_matrix_d_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  double *A,
  const MPF_Int ld_A,
  const double val
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = val;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = val;
      }
    }
  }
}

void mpf_matrix_z_set
(
  const MPF_Layout layout,
  const MPF_Int m_A,
  const MPF_Int n_A,
  MPF_ComplexDouble *A,
  const MPF_Int ld_A,
  const MPF_ComplexDouble *val
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (int i = 0; i < n_A; ++i)
    {
      for (int j = 0; j < m_A; ++j)
      {
        A[ld_A*i+j] = *val;
      }
    }
  }
  else if (layout == MPF_ROW_MAJOR)
  {
    for (int i = 0; i < m_A; ++i)
    {
      for (int j = 0; j < n_A; ++j)
      {
        A[ld_A*i+j] = *val;
      }
    }
  }
}

/* ---------------------- matrix conversion functions ----------------------- */

void mpf_matrix_d_sy2b
(
  char mode,    /* indicates upper of lower triangular storage */
  MPF_Int m_H,  /* number of rows of matrix H */
  MPF_Int n_H,  /* number of columns of matrix H */
  double *H,    /* input banded matrix in standard dense storage scheme */
  MPF_Int ld_H, /* leading dimension of matrix H */
  MPF_Int k,    /* number of bands */
  double *h     /* output vector containing entries in banded storage scheme */
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  h[1] = H[0];
  if (mode == 'U')
  {
    for (i = 1; i < n_H; ++i)
    {
      for (j = 0; j < k; ++j)
      {
        h[k*i+j] = H[ld_H*i+(i-1)+j];
      }
    }
  }
}

void mpf_matrix_d_sy_diag_extract
(
  char mode,    /* indicates upper of lower triangular storage */
  MPF_Int m_H,  /* number of rows of matrix H */
  MPF_Int n_H,  /* number of columns of matrix H */
  double *H,    /* input banded matrix in standard dense storage scheme */
  MPF_Int ld_H, /* leading dimension of matrix H */
  MPF_Int k,    /* current band */
  double *h     /* output vector containing entries in banded storage scheme */
)
{
  MPF_Int count = 0;
  if (mode == 'U')
  {
    for (MPF_Int i = k; i < n_H; ++i)
    {
      h[count] = H[count*ld_H+i];
      count += 1;
    }
  }
}

MPF_Error mpf_matrix_d_read
(
  FILE *file_handle,
  MPF_Layout layout_A,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A
)
{
  int m = 0;
  int n = 0;
  int ret = 0;
  MM_typecode matcode;

  mm_read_banner(file_handle, &matcode);

  /* reads matrix dimensions */
  ret = fscanf(file_handle, "%d %d\n", &m, &n);

  if (((m == (int)m_A) && (n == (int)n_A))
     ||((m_A == -1) && (n_A == -1)))
  {
    if (layout_A == MPF_COL_MAJOR)
    {
      for (MPF_Int i = 0; i < n_A; ++i)
      {
        for (MPF_Int j = 0; j < m_A; ++j)
        {
          ret = fscanf(file_handle, "%lf ", &A[m_A*i+j]);
        }
      }
    }
    else if (layout_A == MPF_ROW_MAJOR)
    {
      for (MPF_Int i = 0; i < m_A; ++i)
      {
        for (MPF_Int j = 0; j < n_A; ++j)
        {
          ret = fscanf(file_handle, "%lf ", &A[n_A*i+j]);
        }
      }
    }
  }
  else
  {
    printf("ERROR: in mpf_matrix_d_read, dimensions missmatch");
    printf("returned value; %d\n", ret);
    return MPF_ERROR_INVALID_ARGUMENT;
  }
  return MPF_ERROR_NONE;
}

MPF_Error mpf_matrix_i_read
(
  FILE *file_handle,
  MPF_Layout layout_A,
  MPF_Int m_A,
  MPF_Int n_A,
  MKL_INT* A
)
{
  int m = 0;
  int n = 0;
  int ret = 0;
  MM_typecode matcode;

  //mm_read_banner(file_handle, &matcode);

  /* reads matrix dimensions */
  //ret = fscanf(file_handle, "%d %d\n", &m, &n);

  //if (((m == (int)m_A) && (n == (int)n_A))
  //   ||((m_A == -1) && (n_A == -1)))
  //{
    if (layout_A == MPF_COL_MAJOR)
    {
      for (MPF_Int i = 0; i < n_A; ++i)
      {
        for (MPF_Int j = 0; j < m_A; ++j)
        {
          ret = fscanf(file_handle, "%lli ", &A[m_A*i+j]);
        }
      }
    }
    else if (layout_A == MPF_ROW_MAJOR)
    {
      for (MPF_Int i = 0; i < m_A; ++i)
      {
        for (MPF_Int j = 0; j < n_A; ++j)
        {
          ret = fscanf(file_handle, "%lli ", &A[n_A*i+j]);
        }
      }
    }
  //}
  //else
  //{
  //  printf("ERROR: in mpf_matrix_d_read, dimensions missmatch");
  //  printf("returned value; %d\n", ret);
  //  return MPF_ERROR_INVALID_ARGUMENT;
  //}
  return MPF_ERROR_NONE;
}

int mpf_matrix_i_write
(
  FILE *handle_file,
  MM_typecode matcode,
  const int *handle,
  const MPF_Int m,
  const MPF_Int n
)
{
  /* writes banner*/
  mm_write_banner(handle_file, matcode);

  /* writes dimensions */
  fprintf(handle_file, "%d %d \n", m, n);

  /* writes matrix data */
  for (MPF_Int i = 0; i < m; ++i)
  {
    for (MPF_Int j = 0; j < n; ++j)
    {
      fprintf(handle_file, "%d ", (int)handle[j*m+i]);
    }
    fprintf(handle_file, "\n");
  }

  return 0;
}

int mpf_matrix_d_write
(
  FILE *handle_file,
  MM_typecode matcode,
  const double *handle,
  const MPF_Int m,
  const MPF_Int n
)
{
  /* writes banner*/
  mm_write_banner(handle_file, matcode);

  /* writes dimensions */
  fprintf(handle_file, "%d %d \n", m, n);

  /* writes matrix data */
  for (MPF_Int i = 0; i < m; ++i)
  {
    for (MPF_Int j = 0; j < n; ++j)
    {
      fprintf(handle_file, "%1.16e ", handle[j*m+i]);
    }
    fprintf(handle_file, "\n");
  }

  return 0;
}

int mpf_matrix_z_write
(
  FILE *handle_file,
  MM_typecode matcode,
  const MPF_ComplexDouble *handle,
  const MPF_Int m,
  const MPF_Int n
)
{
  /* writes banner*/
  mm_write_banner(handle_file, matcode);

  /* writes dimensions */
  fprintf(handle_file, "%d %d \n", m, n);

  /* writes matrix data */
  for (MPF_Int i = 0; i < m; ++i)
  {
    for (MPF_Int j = 0; j < n; ++j)
    {
      fprintf(handle_file, "%1.16E %1.16E ", handle[m*j+i].real,
        handle[m*j+i].imag);
    }
    fprintf(handle_file, "\n");
  }

  fclose(handle_file);
  return 0;
}

int mpf_matrix_d_write_2
(
  const char *filename,
  const double *handle,
  const MPF_Int m,
  const MPF_Int n,
  MPF_IOType io_type
)
{

  printf("in mpf_matrix_d_write_2\n");
  FILE *handle_file;
  if (io_type == MPF_IOWrite)
  {
    if ((handle_file = fopen(filename, "w")) == NULL)
    {
      perror ("fopen");
      return -1;
    }
  }
  else if (io_type == MPF_IOAppend)
  {
    if ((handle_file = fopen (filename, "a")) == NULL)
    {
      perror("fopen");
      return -1;
    }
  }
  else
  {
    return -1;
  }


  /* writes banner*/

  MM_typecode typecode_A;
  mm_initialize_typecode(&typecode_A);
  mm_set_real(&typecode_A);
  mm_set_matrix(&typecode_A);
  mm_set_coordinate(&typecode_A);
  mm_set_general(&typecode_A);
    printf("before write banner\n");
  mm_write_banner(handle_file, typecode_A);

  fprintf(handle_file, "%d %d \n", m, n);
  for (MPF_Int i = 0; i < m; ++i)
  {
    for (MPF_Int j = 0; j < n; ++j)
    {
      fprintf(handle_file, "%1.16E ", handle[j*m+i]);
    }
    fprintf(handle_file, "\n");
  }
  fclose(handle_file);
  return 0;
}

int mpf_matrix_meta_read
(
  FILE *file_handler,
  MM_typecode *matcode,
  MPF_Int *m,
  MPF_Int *n
)
{
  fseek(file_handler, 0, SEEK_SET);
  mm_read_banner(file_handler, matcode);
  mm_read_mtx_array_size(file_handler, m, n);
  fclose(file_handler);
  return 0;
}

int mpf_matrix_size_read
(
  MPF_Int *m,
  MPF_Int *n,
  char *filename
)
{
  FILE *file_handler;
  printf("opening file: %s\n", filename);
  if ((file_handler = fopen(filename, "r")) == NULL)
  {
    printf("exiting...\n");
    return -1;
  }
  mm_read_mtx_array_size(file_handler, m, n);
  printf("*m: %d\n", *m);
  fclose(file_handler);
  return 0;
}

/* --------------------------- printout functions --------------------------- */

void mpf_matrix_d_print
(
  const double *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
)
{
  for (int i = 0; i < m_A; ++i)
  {
    for (int j = 0; j < n_A; ++j)
    {
      printf ("%1.2E ", A[j*ld_A + i]);
    }
    printf ("\n");
  }
}

void mpf_matrix_i_print
(
  const MPF_Int *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
)
{
  for (int i = 0; i < m_A; ++i)
  {
    for (int j = 0; j < n_A; ++j)
    {
      printf ("%d ", A[j*ld_A + i]);
    }
    printf ("\n");
  }
}

void mpf_matrix_d_announce
(
  const double *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
)
{
  printf("\n%s:\n", filename);
  mpf_matrix_d_print(A, m_A, n_A, ld_A);
  printf("\n");
}

void mpf_matrix_i_announce
(
  const MPF_Int *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
)
{
  printf("\n%s:\n", filename);
  mpf_matrix_i_print(A, m_A, n_A, ld_A);
  printf("\n");
}

void mpf_matrix_z_announce
(
  const MPF_ComplexDouble *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A,
  char filename[100]
)
{
  printf("\n%s:\n", filename);
  mpf_matrix_z_print(A, m_A, n_A, ld_A);
  printf("\n");
}


void mpf_matrix_z_print
(
  const MPF_ComplexDouble *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
)
{
  for (uint i = 0; i < m_A; ++i)
  {
    for (uint j = 0; j < n_A; ++j)
    {
      if (A[ld_A*j + i].imag >= 0)
      {
        printf("%1.2E+%1.2Ei  ", A[ld_A*j + i].real, A[ld_A*j + i].imag);
      }
      else
      {
        printf("%1.2E - %1.2Ei  ", A[ld_A*j + i].real, -A[ld_A*j + i].imag);
      }
    }
    printf("\n");
  }
}

void mpf_matrix_c_print
(
  const MPF_Complex *A,
  const MPF_Int m_A,
  const MPF_Int n_A,
  const MPF_Int ld_A
)
{
  for (uint i = 0; i < m_A; ++i)
  {
    for (uint j = 0; j < n_A; ++j)
    {
      if (A[ld_A*j + i].imag >= 0)
        printf("%1.2E+%1.2Ei  ", A[ld_A*j + i].real, A[ld_A*j + i].imag);
      else
        printf("%1.2E - %1.2Ei  ", A[ld_A*j + i].real, -A[ld_A*j + i].imag);
    }
    printf("\n");
  }
}

//int mpf_matrix_V_read
//(
//  MPF_Context *context
//)
//{
//  /* opens file */
//  FILE *file_handle = NULL;
//  if ((file_handle = fopen(context->args.filename_V, "r")) == NULL)
//  {
//    printf("quits reading soon\n");
//    return MM_COULD_NOT_READ_FILE;
//  }
//
//  /* reads matrix dimensions */
//  mpf_matrix_size_read(&context->m_V, &context->n_V, context->args.filename_V);
//
//  /* reads dense matrix */
//  mpf_matrix_d_read(file_handle, MPF_COL_MAJOR, context->m_V, context->n_V,
//    context->V);
//
//  /* close file and exits */
//  fclose(file_handle);
//  return 0;
//}
