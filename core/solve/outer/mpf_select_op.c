#include "mpf.h"

/* ------------- assignment based diagonal extraction functions ------------- */

void mpf_matrix_d_diag_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A,
  MPF_Int ld_A,
  double value
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (MPF_Int i = 0; i < n_A; ++i)
    {
      for (MPF_Int j = 0; j < m_A; ++j)
      {
        if ((i+j) % (n_A) == 0)
        {
          A[ld_A*i+j] = value;
        }
        else
        {
          A[ld_A*i+j] = 0.0;
        }
      }
    }
  }
}

void mpf_matrix_z_diag_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  MPF_ComplexDouble *A,
  MPF_Int ld_A,
  MPF_ComplexDouble value
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  if (layout == MPF_COL_MAJOR)
  {
    for (MPF_Int i = 0; i < n_A; ++i)
    {
      for (MPF_Int j = 0; j < m_A; ++j)
      {
        if ((i+j) % (n_A) == 0)
        {
          A[ld_A*i+j] = value;
        }
        else
        {
          A[ld_A*i+j] = ZERO_C;
        }
      }
    }
  }
}

void mpf_diag_d_set
(
  const MPF_Layout layout,
  MPF_Int m_A,
  MPF_Int n_A,
  double *A,
  MPF_Int ld_A,
  double *values
)
{
  if (layout == MPF_COL_MAJOR)
  {
    for (MPF_Int i = 0; i < n_A; ++i)
    {
      for (MPF_Int j = 0; j < m_A; ++j)
      {
        if ((i+j) % (n_A) == 0)
        {
          A[ld_A*i+j] = values[i];
        }
        else
        {
          A[ld_A*i+j] = 0.0;
        }
      }
    }
  }
}
