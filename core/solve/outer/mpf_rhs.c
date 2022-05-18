#include "mpf.h"

void mpf_set_B
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  solver->n_max_B = pow((double)probe->stride, probe->n_levels);
}

void mpf_get_max_nrhs
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  solver->max_blk_fA = (MPF_Int)pow((double)probe->stride, (double)probe->n_levels);
  solver->n_max_B = solver->max_blk_fA*probe->n_colors; 
}

/*============================================================================*/
/* num_batches: n_max_B / batch_size => number of higher level rhs blocks     */
/* batch_id   : n_max_B % batch_size => indexes of higher level rhs blocks    */
/* num_blocks : batch_size / blk => number of inner blocks inside batches     */
/* (used by block and global methods)                                         */
/* block_id   : batch_size % blk => indexes of lower level rhs blocks         */
/*                                                                            */
/* num_parallel_batches => batches running on separate threads, set by user   */
/* parallel_sessions: ceil((num_batches - num_parallel_batches)/batch_size)   */
/* => remaining rhs that have to be rearragned to executed in parallel        */
/*============================================================================*/
//void mpf_probing_rhs_meta_pthreads_initialize
//(
//  MPF_Context *context
//)
//{
//  if (context->solver_outer_type == MPF_BATCH)
//  {
//    if (context->probing_type == MPF_PROBING_BLOCKING)
//    {
//      context->n_B = ((MPF_Int) pow((double) context->probing.blocking.blk,
//        (double)context->n_levels))*context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_MULTILEVEL_SAMPF_LING)
//    {
//      context->n_B =
//      ((MPF_Int) pow((double)context->probing.multilevel.stride,
//      (double) context->n_levels))*context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_MULTIPATH_SAMPF_LING)
//    {
//
//    }
//    else if (context->probing_type == MPF_PROBING_AVERAGE_MULTIPATH_SAMPF_LING)
//    {
//      context->n_B =
//        ((MPF_Int)pow((double)context->probing.average_multipath.stride,
//        (double) context->n_levels))*context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_BLOCKING_MKL)
//    {
//      context->n_B = ((MPF_Int) pow((double) context->probing.blocking.blk,
//        (double)context->n_levels))*context->n_colors;
//      context->ld_B = context->m_B;
//    }
//  }
//  else if (context->solver_outer_type == MPF_SOLVER_OUTER_PTHREADS_BATCH)
//  {
//    if (context->probing_type == MPF_PROBING_BLOCKING)
//    {
//      context->n_B =
//        ((MPF_Int) pow((double) context->probing.blocking.blk,
//        (double) context->n_levels)) * context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_MULTILEVEL_SAMPF_LING)
//    {
//      context->n_B =
//        ((MPF_Int) pow((double) context->probing.multilevel.stride,
//        (double) context->n_levels)) * context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_MULTIPATH_SAMPF_LING)
//    {
//
//    }
//    else if (context->probing_type == MPF_PROBING_AVERAGE_MULTIPATH_SAMPF_LING)
//    {
//      context->n_B =
//        ((MPF_Int) pow((double)context->probing.average_multipath.stride,
//        (double) context->n_levels))*context->n_colors;
//      context->ld_B = context->m_B;
//    }
//    else if (context->probing_type == MPF_PROBING_BLOCKING_MKL)
//    {
//      context->n_B =
//        ((MPF_Int) pow((double) context->probing.blocking.blk,
//        (double) context->n_levels)) * context->n_colors;
//      context->ld_B = context->m_B;
//    }
//  }
//}

/* ----------------------- rhs generator (threaded) ------------------------- */


//void mpf_probing_vectors_d_pthreads_generate
//(
//  MPF_Context *context,
//  MPF_Int start,
//  MPF_Int end,
//  MPF_MatrixDense *B
//)
//{
//  MPF_Int i = 0;
//  MPF_Int j = 0;
//  MPF_Int blk_max_fA = mpf_blk_max_fA_get(context);
//  MPF_Int color_size = blk_max_fA * context->n_colors;
//  double *data_handle = (double *) B;
//
//  //printf("end: %d\n", end);
//  if ((start >= 0) && (end <= color_size))
//  {
//    i = 0;
//    j = 0;
//    for (i = 0; i < end; ++i)
//    {
//      for (j = 0; j < context->m_B; ++j)
//      {
//        if ((j-i-start) % (color_size) == 0)
//        {
//          data_handle[context->m_B*i+j] = 1.0;
//        }
//        else
//        {
//          data_handle[context->m_B*i+j] = 0.0;
//        }
//      }
//    }
//  }
//}

/* ----------------------------- rhs generators ----------------------------- */


//void mpf_probing_vectors_z_apply_dynamic_5_diag_blocks(MPF_Context *context,
//MPF_ComplexDouble *X, MPF_Int cols_start, MPF_Int cols_offset)
/* Similar to apply_dynamic_5 but for diag blocks 

      cols_start  offset+current_rhs
            V        V
       [    |        |      ]
       [    |        |      ] <-- i
       [    |        |      ]
   V = [    |        |      ]
       [    |        |      ]
       [    |        |      ]
       [    |        |      ]
                          ^
                          |
                          j

    INVARIANT:
    (i+cols_start)/part_size => partiition
    context->memory_colorings[j/blk_max_fA] => coloring of block_node
    if blk_fA == parition_size
*/
void mpf_z_blk_select_X_dynamic
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  MPF_ComplexDouble *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
)
{
  MPF_ComplexDouble ZERO_COMPF_LEX = mpf_scalar_z_init(0.0, 0.0);
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int row_bound = 0;

  for (i = 0; i < cols_offset; ++i)
  {
    for (j = 0; j < m_X; ++j)
    {
      row_bound = (j/blk_fA)*blk_fA + blk_fA;
      if
      (
        ((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
        (j < row_bound) &&
        ((j%blk_max_fA)/blk_fA == ((i+cols_start)%blk_max_fA)/blk_fA)
      )
      {
        /* do nothing */
      }
      else
      {
        X[m_X*i+j] = ZERO_COMPF_LEX;
      }
    }
  }
}

void mpf_d_generate_B
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Dense *B
)
{
  for (MPF_Int i = 0; i < B->n; ++i)
  {
    for (MPF_Int j = 0; j < B->m; ++j)
    {
      if (((i+solver->current_rhs)/solver->max_blk_fA == probe->colorings_array[j/solver->max_blk_fA]) &&
          (j%solver->max_blk_fA == (i+solver->current_rhs)%solver->max_blk_fA))
      {
        ((double*)B->data)[B->m*i+j] = 1.0;
      }
      else
      {
        ((double*)B->data)[B->m*i+j] = 0.0;
      }
    }
  }
}

void mpf_z_generate_B
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  MPF_Dense *B
)
{
  MPF_ComplexDouble ONE = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO = mpf_scalar_z_init(.0, 0.0);

  for (MPF_Int i = 0; i < B->n; ++i)
  {
    for (MPF_Int j = 0; j < B->m; ++j)
    {
      //printf("colorings_array[j/blk_max_fA]: %d\n",
      //  colorings_array[j/blk_max_fA]);
      if (((i+solver->current_rhs)/solver->max_blk_fA == probe->colorings_array[j/solver->max_blk_fA]) &&
          (j%solver->max_blk_fA == (i+solver->current_rhs)%solver->max_blk_fA))
      {
        ((MPF_ComplexDouble*)B->data)[B->m*i+j] = ONE;
      }
      else
      {
        ((MPF_ComplexDouble*)B->data)[B->m*i+j] = ZERO;
      }
    }
  }
}

void mpf_d_xy_generate_B
(
  MPF_Int blk_max_fA,
  MPF_Int n_levels,
  MPF_Int *colorings_array,
  MPF_Int cols_start,
  MPF_Int m_B,
  MPF_Int n_B,
  double *B
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int ml = m_B%(MPF_Int)pow((double)blk_max_fA, (double)n_levels);
  MPF_Int mf = (MPF_Int)pow(blk_max_fA, n_levels)-ml;

  for (i = 0; i < n_B; ++i)
  {
    for (j = 0; j < mf; ++j)
    {
      if (((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
          (j%blk_max_fA == (i+cols_start)%blk_max_fA))
      {
        B[m_B*i+j] = 1.0;
      }
      else
      {
        B[m_B*i+j] = 0.0;
      }
    }
  }

  blk_max_fA = ml;
  for (i = 0; i < n_B; ++i)
  {
    for (j = mf; j < m_B; ++j)
    {
      if (((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
          (j%blk_max_fA == (i+cols_start)%blk_max_fA))
      {
        B[m_B*i+j] = 1.0;
      }
      else
      {
        B[m_B*i+j] = 0.0;
      }
    }
  }
}

//void mpf_z_generate_B
//(
//  MPF_Int blk_max_fA,
//  MPF_Int *colorings_array,
//  MPF_Int cols_start,
//  MPF_Int m_B,
//  MPF_Int n_B,
//  MPF_ComplexDouble *B
//)
//{
//  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
//  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
//  MPF_Int i = 0;
//  MPF_Int j = 0;
//  for (i = 0; i < n_B; ++i)
//  {
//    for (j = 0; j < m_B; ++j)
//    {
//      if (((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
//          (j%blk_max_fA == (i+cols_start)%blk_max_fA))
//      {
//        B[m_B*i+j] = ONE_C;
//      }
//      else
//      {
//        B[m_B*i+j] = ZERO_C;
//      }
//    }
//  }
//}


/* --------------------- probing vectors select functions -------------------- */

/* stores only the upper triangular part of each block */
void mpf_d_select
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  double *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int row_bound = 0;
  MPF_Int ml = m_X%blk_fA;
  MPF_Int mf = m_X-ml;

  for (i = 0; i < cols_offset; ++i)
  {
    for (j = 0; j < mf; ++j)
    {
      row_bound = (j/blk_fA)*blk_fA + blk_fA;
      if
      (
        ((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
        (j < row_bound) &&
        ((j%blk_max_fA)/blk_fA == ((i+cols_start)%blk_max_fA)/blk_fA) &&
        ((i+cols_start)%blk_max_fA >= j%blk_max_fA)
      )
      {
        /* do nothing */
      }
      else
      {
        X[m_X*i+j] = 0.0;
      }
    }
  }

  blk_fA = ml;
  for (i = 0; i < cols_offset; ++i)
  {
    for (j = mf; j < m_X; ++j)
    {
      row_bound = (j/blk_fA)*blk_fA + blk_fA;
      if
      (
        ((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
        (j < row_bound) &&
        ((j%blk_max_fA)/blk_fA == ((i+cols_start)%blk_max_fA)/blk_fA) &&
        ((i+cols_start)%blk_max_fA >= j%blk_max_fA)
      )
      {
        /* do nothing */
      }
      else
      {
        X[m_X*i+j] = 0.0;
      }
    }
  }
}

void mpf_z_select
(
  MPF_Int blk_max_fA,
  MPF_Int *colorings_array,
  MPF_Int m_X,
  MPF_ComplexDouble *X,
  MPF_Int blk_fA,
  MPF_Int cols_start,
  MPF_Int cols_offset
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_Int row_bound = 0;
  MPF_Int ml = m_X%blk_fA;
  MPF_Int mf = m_X-ml;
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);

  for (i = 0; i < cols_offset; ++i)
  {
    for (j = 0; j < mf; ++j)
    {
      row_bound = (j/blk_fA)*blk_fA + blk_fA;
      if
      (
        ((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
        (j < row_bound) &&
        ((j%blk_max_fA)/blk_fA == ((i+cols_start)%blk_max_fA)/blk_fA) &&
        ((i+cols_start)%blk_max_fA >= j%blk_max_fA)
      )
      {
        /* do nothing */
      }
      else
      {
        X[m_X*i+j] = ZERO_C;
      }
    }
  }

  blk_fA = ml;
  for (i = 0; i < cols_offset; ++i)
  {
    for (j = mf; j < m_X; ++j)
    {
      row_bound = (j/blk_fA)*blk_fA + blk_fA;
      if
      (
        ((i+cols_start)/blk_max_fA == colorings_array[j/blk_max_fA]) &&
        (j < row_bound) &&
        ((j%blk_max_fA)/blk_fA == ((i+cols_start)%blk_max_fA)/blk_fA) &&
        ((i+cols_start)%blk_max_fA >= j%blk_max_fA)
      )
      {
        /* do nothing */
      }
      else
      {
        X[m_X*i+j] = ZERO_C;
      }
    }
  }
}

/* --------------------- processing extracte entries ------------------------ */


void mpf_diag_d_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B_in
)
{
  MPF_Dense *B = (MPF_Dense*)B_in;
  for (MPF_Int i = 0; i < B->n; ++i)
  {
    for (MPF_Int j = 0; j < B->m; ++j)
    {
      if
      (
        (j%solver->blk_fA) > (i%solver->blk_fA)
      )
      {
        ((double*)B->data)[B->m*i+j] = ((double*)B->data)[B->m*(j%B->n)+i+(j/solver->blk_fA)*solver->blk_fA];
      }
    }
  }
}

void mpf_diag_zhe_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B_in
)
{
  MPF_Dense *B = (MPF_Dense*)B_in;
  for (MPF_Int i = 0; i < B->n; ++i)
  {
    for (MPF_Int j = 0; j < B->m; ++j)
    {
      if
      (
        (j%solver->blk_fA) > (i%solver->blk_fA)
      )
      {
        ((MPF_ComplexDouble*)B->data)[B->m*i+j].real = ((MPF_ComplexDouble*)B->data)[B->m*(j%solver->blk_fA)+i+(j/solver->blk_fA)*solver->blk_fA].real;
        ((MPF_ComplexDouble*)B->data)[B->m*i+j].imag = -((MPF_ComplexDouble*)B->data)[B->m*(j%solver->blk_fA)+i+(j/solver->blk_fA)*solver->blk_fA].imag;
      }
    }
  }
}

void mpf_diag_zsy_sym2gen
(
  MPF_Probe *probe,
  MPF_Solver *solver,
  void *B_in
)
{
  MPF_Dense *B = (MPF_Dense*)B_in;
  for (MPF_Int i = 0; i < B->n; ++i)
  {
    for (MPF_Int j = 0; j < B->m; ++j)
    {
      if
      (
        (j%solver->blk_fA) > (i%solver->blk_fA)
      )
      {
        ((MPF_ComplexDouble*)B->data)[B->m*i+j] = ((MPF_ComplexDouble*)B->data)[B->m*(j%solver->blk_fA)+i+(j/solver->blk_fA)*solver->blk_fA];
      }
    }
  }
}
