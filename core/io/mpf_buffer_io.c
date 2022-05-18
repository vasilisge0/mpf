#include "mpf.h"

void mp_write_buffer_byte
(
  FILE *file_h,
  MPF_Int n_bytes,
  void *buffer
)
{
  MPF_Int i = 0;
  for (i = 0; i < n_bytes; ++i)
  {
    fprintf(file_h, "%c ", ((char*)buffer)[i]);
  }
  fprintf(file_h, "\n");
}

void mp_write_buffer_int
(
  FILE *file_h,
  MPF_Int n_entries,
  void *buffer
)
{
  MPF_Int i = 0;
  for (i = 0; i < n_entries; ++i)
  {
    fprintf(file_h, "%d ", ((MPF_Int*)buffer)[i]);
  }
  fprintf(file_h, "\n");
}

void mp_write_buffer_double
(
  FILE *file_h,
  MPF_Int n_entries,
  void *buffer
)
{
  MPF_Int i = 0;
  for (i = 0; i < n_entries; ++i)
  {
    fprintf(file_h, "%e ", ((double*)buffer)[i]);
  }
  fprintf(file_h, "\n");
}
