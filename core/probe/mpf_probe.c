#include "mpf.h"

/* ---------------- memory allocation functions for probing ----------------- */

void mpf_probe_alloc
(
  MPF_Probe *context
)
{
  MPF_Int m = context->n_nodes;
  context->bytes_buffer = sizeof(MPF_Int)*m*2;
  context->buffer = mpf_malloc(context->bytes_buffer);
  context->colorings_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*m);
}

void mpf_avg_probe_alloc
(
  MPF_Probe *context
)
{
  MPF_Int m = context->n_nodes;
  srand(time(0));
  MPF_Int stride = context->stride;
  context->bytes_buffer = sizeof(MPF_Int)*m/context->stride*2;
  context->buffer = mpf_malloc(context->bytes_buffer);

  context->mappings_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*stride*(stride+1)/2);
  context->endpoints_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*context->n_endpoints);
  context->colorings_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*m);
}

void mpf_probe_free_internal
(
  MPF_Probe *context
)
{
  if (context->buffer != NULL)
  {
    mpf_free(context->buffer);
    context->buffer = NULL;
  }

  if (context->colorings_array != NULL)
  {
    mpf_free(context->colorings_array);
    context->colorings_array = NULL;
  }
}

void mpf_probe_free
(
  MPF_Probe* context
)
{
  /* testing */
  //if (context->P.format == MPF_SPARSE_CSR)
  //{
  //  mpf_sparse_csr_free(&context->P);
  //}

  //else if c(ontext->P.type == MPF_SPARSE_CSC)
  //{
  //  //mpf_sparse_csc_free(&context->P.csc);
  //}
  //else if (ontext->P.type == MPF_SPARSE_COO)
  //{
  //  //mpf_sparse_coo_free(&context->P.coo);
  //}
  if (context->endpoints_array != NULL)
  {
    mpf_free(context->endpoints_array);
    context->endpoints_array = NULL;
  }

  if (context->mappings_array != NULL)
  {
    mpf_free(context->mappings_array);
    context->mappings_array = NULL;
  }

  if (context->buffer != NULL)
  {
    mpf_free(context->buffer);
    context->buffer = NULL;
  }

  //if (context->colorings_array != NULL)
  //{
  //  mpf_free(context->colorings_array);
  //  context->colorings_array = NULL;
  //}
}

void mpf_probe_colorings_alloc
(
  MPF_Probe *context
)
{
  context->colorings_array = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*context->P.m);
}


void mpf_probe_mem_free
(
  MPF_Probe *context
)
{
  if (context->buffer != NULL)
  {
    mpf_free(context->buffer);
    context->buffer = NULL;
  }
}

MPF_Int mpf_get_n_diags
(
  MPF_Int n_levels,
  MPF_Int degree
)
{
  MPF_Int i = 0;
  MPF_Int num_diags = 1;
  for (i = 0; i < n_levels; ++i)
  {
    num_diags = num_diags*degree;
  }
  return num_diags;
}

void mpf_probe_get_sampling_offsets
(
  MPF_Probe *context,
  MPF_Int *offset_rows,
  MPF_Int *offset_cols
)
{
  if (context->type == MPF_PROBE_PATH_SAMPLING)
  {
    *offset_rows = context->offset_rows;
    *offset_cols = context->offset_cols;
  }
  else if (context->type == MPF_PROBE_AVG_PATH_SAMPLING)
  {
    *offset_rows = context->offset_rows;
    *offset_cols = context->offset_cols;
  }
}

/* --------------------- unpacking memory for sorting ----------------------- */

void mpf_probe_unpack_sort_mem
(
  MPF_Probe *probe,
  MPF_Int *temp_array,
  MPF_Int *temp_i_array
)
{
  /* unpack */
  temp_array = (MPF_Int*)probe->buffer;     /* temp cols */
  temp_i_array = &(temp_array)[probe->n_nodes]; /* temp inverse table */

  /* initialization */
  mpf_matrix_i_set(MPF_COL_MAJOR, probe->n_nodes, 1, temp_array,
    probe->n_nodes, 0);
  mpf_matrix_i_set(MPF_COL_MAJOR, probe->n_nodes, 1, temp_i_array,
     probe->n_nodes, -1);
}

void mpf_probe_init
(
  MPF_Context *context
)
{
  context->probe.P.format = MPF_SPARSE_CSR;
  context->probe.endpoints_array = NULL;
  context->probe.mappings_array = NULL;
  context->probe.buffer = NULL;
}
