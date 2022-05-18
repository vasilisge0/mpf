#include "mpf.h"

/* ----------------------------- bucket_array ------------------------------- */

void mpf_bucket_array_init
(
  MPF_BucketArray *H,
  MPF_Int n_values,
  MPF_Int n_bins
)
{
  H->max_n_values = n_values;
  H->max_n_bins = n_bins;
  H->n_values = 0;
  H->n_bins = 0;
}

/*============================================================================*/
/* Initializes structure MPF_BucketArray with the following fields:           */
/*       next: array that holds the next entry                                */
/* bins_start: array that holds the first entry of each bin                   */
/*   bins_end: array that holds the last entry of each bin                    */
/*  bins_size: array that holds the size of each bin                          */
/*============================================================================*/
void mpf_bucket_array_values_init
(
  MPF_BucketArray *H
)
{
  mpf_matrix_i_set(MPF_COL_MAJOR, H->max_n_values, 1, H->next,
    H->max_n_values, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, H->max_n_bins, 1, H->bins_start,
    H->max_n_bins, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, H->max_n_bins, 1, H->bins_end,
    H->max_n_bins, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, H->max_n_bins, 1, H->bins_size,
    H->max_n_bins, 0);
}

/*============================================================================*/
/* Allocates memory for fields of structure MPF_BucketArray:                  */
/*       next: array that holds the next entry                                */
/* bins_start: array that holds the first entry of each bin                   */
/*   bins_end: array that holds the last entry of each bin                    */
/*  bins_size: array that holds the size of each bin                          */
/*============================================================================*/
void mpf_bucket_array_alloc
(
  MPF_BucketArray *H
)
{
  if (H->max_n_bins != 0)
  {
    H->bins_start = (MPF_Int*) mpf_malloc((sizeof *H->bins_start)*H->max_n_bins);
    H->bins_end = (MPF_Int*) mpf_malloc((sizeof *H->bins_end)*H->max_n_bins);
    H->bins_size = (MPF_Int*) mpf_malloc((sizeof *H->bins_size)*H->max_n_bins);
  }

  if (H->max_n_values != 0)
  {
    H->values = (MPF_Int *) mpf_malloc((sizeof *H->values)*H->max_n_values);
    H->next = (MPF_Int *) mpf_malloc((sizeof *H->next)*H->max_n_values);
  }
}


/*============================================================================*/
/* mpf_bucket_array_insert                                                    */
/* Inserts value into selected bin of MPF_BucketArray H                       */
/*============================================================================*/
void mpf_bucket_array_insert
(
  MPF_BucketArray *H,
  MPF_Int bin,
  MPF_Int value
)
{
  if (H->n_values == H->max_n_values) /* reallocates data/next arrays */
  {
    H->max_n_values += H->values_mem_increment;
    H->values = (MPF_Int*)mkl_realloc(H->values, (sizeof *H->values)*H->max_n_values);
    H->next = (MPF_Int*)mkl_realloc(H->next, (sizeof *H->next)*H->max_n_values);

    /* sets new entries to -1 */
    mpf_matrix_i_set(MPF_COL_MAJOR, H->values_mem_increment, 1,
      &H->values[H->n_values], H->values_mem_increment, -1);
    mpf_matrix_i_set(MPF_COL_MAJOR, H->values_mem_increment, 1,
      &H->next[H->n_values], H->values_mem_increment, -1);
  }

  //if (H->n_bins == H->max_n_bins) /* reallocates bins_start/bins_size arrays */
  //{
  //    H->max_n_bins += H->bins_memory_increment;
  //    H->bins_start = mkl_realloc(H->bins_start, (sizeof *H->bins_start)*H->max_n_bins);
  //    H->bins_end = mkl_realloc(H->bins_end, (sizeof *H->bins_end)*H->max_n_bins);
  //    H->bins_size = mkl_realloc(H->bins_size, (sizeof *H->bins_size)*H->max_n_bins);
  //    mpf_matrix_i_set(MPF_COL_MAJOR, H->bins_memory_increment, 1, &H->bins_start[H->n_bins], H->bins_memory_increment, -1);
  //    mpf_matrix_i_set(MPF_COL_MAJOR, H->bins_memory_increment, 1, &H->bins_end[H->n_bins], H->bins_memory_increment, -1);
  //    mpf_matrix_i_set(MPF_COL_MAJOR, H->bins_memory_increment, 1, &H->bins_size[H->n_bins], H->bins_memory_increment, 0);
  //}

  H->values[H->n_values] = value;   /* adds entry */
  if (H->bins_size[bin] == 0)
  {
      H->bins_start[bin] = H->n_values;
      H->bins_size[bin] = 0; /* is not required if the array is initialized to [0, 0 ... 0] */
  }
  H->next[H->bins_end[bin]] = H->n_values;
  H->bins_end[bin] = H->n_values;
  H->bins_size[bin] += 1;
  H->n_values += 1;
}

/*============================================================================*/
/* mpf_bucket_array_find_max_bin_size                                         */
/* Performs linear search among bins o MPF_BucketArray object H to find the   */
/* maximum bin size.                                                          */
/*============================================================================*/
void mpf_bucket_array_find_max_bin_size
(
  MPF_BucketArray *H
)
{
  MPF_Int i = 0;
  for (i = 0; i < H->n_bins; ++i)
  {
    H->max_bin_size = (H->bins_size[i] > H->max_bin_size)
                    ? H->bins_size[i]
                    : H->max_bin_size;
  }
}

/*============================================================================*/
/* Frees memory for fields of MPF_BucketArray Object                          */
/*============================================================================*/
void mpf_bucket_array_free
(
  MPF_BucketArray *H
)
{
  if (H->max_n_values > 0)
  {
    mpf_free(H->bins_start);
    mpf_free(H->bins_end);
    mpf_free(H->next);
    mpf_free(H->values);
    mpf_free(H->bins_size);
  }
}

void mpf_bucket_array_find_min
(
  MPF_BucketArray *H,
  MPF_Int start,
  MPF_Int bin,
  MPF_Int *nentries,
  MPF_Int *min
)
{
  for (MPF_Int i = start; i < H->bins_end[bin]; ++i)
  {
    *min = (H->values[i] < *min) ? H->values[i] : *min;
    (*nentries) += 1;
  }
}

/* ---------------------------- I/O functions ------------------------------- */

void mpf_bucket_array_write
(
  FILE *file_handle,
  MPF_BucketArray *H
)
{
  MPF_Int i = 0;

  /* write metadata */
  fprintf(file_handle, "%d %d\n", (int)H->n_bins, (int)H->max_n_bins);
  fprintf(file_handle, "%d %d\n", (int)H->n_values, (int)H->max_n_values);
  fprintf(file_handle, "%d\n", (int)H->max_bin_size);
  fprintf(file_handle, "%d\n", (int)H->values_mem_increment);
  fprintf(file_handle, "%d\n", (int)H->bins_mem_increment);

  /* write bins_start */
  if (H->n_bins > 0)
  {
    for (MPF_Int i = 0; i < H->n_bins; ++i)
    {
      fprintf(file_handle, "%lli ", H->bins_start[i]);
    }
    fprintf(file_handle, "\n");
  }

  /* write bins_end */
  if (H->n_bins > 0)
  {
    for (MPF_Int i = 0; i < H->n_bins; ++i)
    {
      fprintf(file_handle, "%lli ", H->bins_end[i]);
    }
    fprintf(file_handle, "\n");
  }

  /* write bins_size */
  if (H->n_bins > 0)
  {
    for (i = 0; i < H->n_bins; ++i)
    {
      fprintf(file_handle, "%lli ", H->bins_size[i]);
    }
    fprintf(file_handle, "\n");
  }

  /* write bins_values */
  if (H->n_values > 0)
  {
    for (i = 0; i < H->n_values; ++i)
    {
      fprintf(file_handle, "%lli ", H->values[i]);
    }
    fprintf(file_handle, "\n");
  }

  /* write bins_values */
  if (H->n_values > 0)
  {
    for (i = 0; i < H->n_values; ++i)
    {
      fprintf(file_handle, "%lli ", H->next[i]);
    }
    fprintf(file_handle, "\n");
  }
}
