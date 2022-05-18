#include "mpf.h"

/*================================*/
/*== qr direct solver functions ==*/
/*================================*/

/* general qr */
//void mpf_qr_givens_dge_factorize
//(
//  const MPF_Int n_H,
//  const MPF_Int n_B,
//  double *H,
//  const MPF_Int ld_H,
//  double *b,
//  MPF_Int ld_br,
//  double *tempf_matrix
//)
void mpf_qr_givens_dge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
)
{
  MPF_Int j;
  for (j = 0; j < n_H; ++j)
  {
    double tempf_vector[2];
    double R[4];
    double swap[2];
    int n_givens_H;

    /* computes givens rotation matrix R and updates H */
    n_givens_H = n_H-j;
    R[0] = H[ld_H*j+j]/
      sqrt(H[ld_H*j+j]*H[ld_H*j+j]+H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1]/
      sqrt(H[ld_H*j+j]*H[ld_H*j+j]+H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[2] = - R[1];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
      1.0, R, 2, &H[ld_H*j+j], ld_H, 0.0, tempf_matrix, 2);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, tempf_matrix, 2,
      &H[ld_H*j+j], ld_H);

    /* updates rhs */
    tempf_vector[0] = R[0]*b[j];
    swap[0]        = R[2]*b[j+1];
    tempf_vector[1] = R[1]*b[j];
    swap[1]        = R[3]*b[j+1];
    b[j]   = tempf_vector[0] + swap[0];
    b[j+1] = tempf_vector[1] + swap[1];
  }
}

/*
    General QR factorization. Needs to add pivoting
*/
void mpf_qr_givens_dge_2
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
)
{
  MPF_Int j;
  MPF_Int i;
  for (j = 0; j < n_H; ++j)
  {
    double tempf_vector[2];
    double R[4];
    double swap[2];
    int n_givens_H;

    for (i = m_H-2; i >= j; --i)
    {
      if (fabs(H[ld_H*j+i]) > 1e-10) // arbitrary threshold for zero. TESTING
      {
        /* computes givens rotation matrix R */
        n_givens_H = n_H-j;
        R[0] = H[ld_H*j+i]
             / sqrt(H[ld_H*j+i]*H[ld_H*j+i]+H[ld_H*j+i+1]*H[ld_H*j+i+1]);
        R[3] = R[0];
        R[1] = -H[ld_H*j+i+1]
             / sqrt(H[ld_H*j+i]*H[ld_H*j+i]+H[ld_H*j+i+1]*H[ld_H*j+i+1]);
        R[2] = -R[1];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
          1.0, R, 2, &H[ld_H*j+i], ld_H, 0.0, tempf_matrix, 2);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, tempf_matrix, 2,
          &H[ld_H*j+i], ld_H);

        /* updates rhs */
        tempf_vector[0] = R[0]*b[i];
        swap[0]        = R[2]*b[i+1];
        tempf_vector[1] = R[1]*b[i];
        swap[1]        = R[3]*b[i+1];
        b[i]   = tempf_vector[0] + swap[0];
        b[i+1] = tempf_vector[1] + swap[1];
      }
    }
  }
}

/*
    General QR factorization. Needs to add pivoting. Based on
    mpf_qr_givens_dge_factorize_2.
*/

void mpf_qr_givens_mrhs_dge_2
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
)
{
  MPF_Int j = 0;
  MPF_Int i = 0;
  MPF_Int k = 0;
  for (j = 0; j < n_H; ++j)
  {
    double tempf_vector[2];
    double R[4];
    double swap[2];
    int n_givens_H;

    for (i = m_H-2; i >= j; --i)
    {
      if (fabs(H[ld_H*j+i]) > 1e-10) // arbitrary threshold for zero. TESTING
      {
        n_givens_H = n_H-j;
        R[0] = H[ld_H*j+i]
             / sqrt(H[ld_H*j+i]*H[ld_H*j+i] + H[ld_H*j+i+1]*H[ld_H*j+i+1]);
        R[3] = R[0];
        R[1] = - H[ld_H*j+i+1]
             / sqrt(H[ld_H*j+i]*H[ld_H*j+i] + H[ld_H*j+i+1]*H[ld_H*j+i+1]);
        R[2] = - R[1];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
          1.0, R, 2, &H[ld_H*j+i], ld_H, 0.0, tempf_matrix, 2);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, tempf_matrix, 2,
          &H[ld_H*j+i], ld_H);

        for (k = 0; k < n_B; ++k)
        {
          tempf_vector[0] = R[0]*b[m_H*k+i];
          swap[0]        = R[2]*b[m_H*k+i+1];
          tempf_vector[1] = R[1]*b[m_H*k+i];
          swap[1]        = R[3]*b[m_H*k+i+1];
          b[m_H*k+i]   = tempf_vector[0] + swap[0];
          b[m_H*k+i+1] = tempf_vector[1] + swap[1];
        }
      }
    }
  }
}

void mpf_qr_givens_mrhs_dge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  MPF_Int ld_br,
  double *tempf_matrix
)
{
  MPF_Int j = 0;
  MPF_Int i = 0;
  for (j = 0; j < n_H; ++j)
  {
    double tempf_vector[2];
    double R[4];
    double swap[2];
    int n_givens_H;

    n_givens_H = n_H - j;
    R[0] = H[ld_H*j+j]
         / sqrt(H[ld_H*j+j]*H[ld_H*j+j] + H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1]
         / sqrt(H[ld_H*j+j]*H[ld_H*j+j] + H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[2] = - R[1];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
      1.0, R, 2, &H[ld_H*j+j], ld_H, 0.0, tempf_matrix, 2);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, tempf_matrix, 2,
      &H[ld_H*j+j], ld_H);

    for (i = 0; i < n_B; ++i)
    {
      /* update */
      tempf_vector[0] = R[0]*b[n_H*i+j];
      swap[0] = R[2]*b[n_H*i+j+1];
      tempf_vector[1] = R[1]*b[n_H*i+j];
      swap[1] = R[3]*b[n_H*i+j+1];
      /* copy result to b */
      b[n_H*i+j] = tempf_vector[0] + swap[0];
      b[n_H*i+j+1] = tempf_vector[1] + swap[1];
    }
  }
}

void mpf_qr_givens_sge
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *b,
  MPF_Int ld_br,
  float *tempf_matrix
)
{
  MPF_Int j;
  for (j = 0; j < n_H; ++j)
  {
    float tempf_vector[2];
    float R[4];
    float swap[2];
    int n_givens_H;

    /* computes rotation matrix R and and apply to H */
    n_givens_H = n_H - j;
    R[0] = H[ld_H*j+j]
         / sqrt(H[ld_H*j+j]*H[ld_H*j+j] + H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1]
         / sqrt(H[ld_H*j+j]*H[ld_H*j+j] + H[ld_H*j+j+1]*H[ld_H*j+j+1]);
    R[2] = - R[1];
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
      1.0, R, 2, &H[ld_H*j+j], ld_H, 0.0, tempf_matrix, 2);
    LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, tempf_matrix, 2,
      &H[ld_H*j+j], ld_H);

    /* updates rhs */
    tempf_vector[0] = R[0]*b[j];
    swap[0]        = R[2]*b[j+1];
    tempf_vector[1] = R[1]*b[j];
    swap[1]        = R[3]*b[j+1];
    b[j]   = tempf_vector[0] + swap[0];
    b[j+1] = tempf_vector[1] + swap[1];
  }
}

/* general block qr */

void mpf_block_qr_dge_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B,
  const MPF_Int blk
)
{
  MPF_Int j, i;
  MPF_Int n_givens_H = n_H;

  double *memory_handle = (double*)mpf_malloc((sizeof *memory_handle) * (2 * n_H + 2 * n_B)); //@OPTIMIZE: this should be allocated beforehand
  double *tempf_matrix = memory_handle;
  double *tempf_vecblk = &memory_handle[2*n_H];

  for (j = 0; j < n_H; ++j)
  {
    MPF_Int k = ld_H*j + blk*(j/blk) + blk + j%blk;
    MPF_Int z = blk*(j/blk) + blk + j%blk;
    double R[4];
    double norm_temp;
    n_givens_H = n_H-j;
    for (i = 0; i < blk; ++i)
    {
      k = k - 1;
      z = z - 1;

      norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
      R[0] = H[k] / norm_temp;
      R[3] = R[0];
      R[1] = - H[k+1] / norm_temp;
      R[2] = - R[1];
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, &H[k], ld_H,
        tempf_matrix, 2);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H, 2,
        1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
        tempf_vecblk, 2);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
        2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory_handle);
}

void mpf_block_qr_sge_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *B,
  const MPF_Int ld_B,
  const MPF_Int blk
)
{
  MPF_Int j = 0;
  MPF_Int i = 0;
  MPF_Int n_givens_H = n_H;
  float *memory_handle = (float*)mpf_malloc((sizeof *memory_handle)*(2*n_H+2*n_B));  //@OPTIMIZE this should be alocated beforehand
  float *tempf_matrix = memory_handle;
  float *tempf_vecblk = &memory_handle[2*n_H];

  for (j = 0; j < n_H; ++j)
  {
    MPF_Int k = ld_H*j + blk*(j / blk) + blk + (j % blk);
    MPF_Int z = blk*(j / blk) + blk + j % blk;
    float R[4];
    float norm_temp;
    n_givens_H = n_H - j;
    for (i = 0; i < blk; ++i)
    {
      k = k - 1;
      z = z - 1;

      norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
      R[0] = H[k] / norm_temp;
      R[3] = R[0];
      R[1] = - H[k+1] / norm_temp;
      R[2] = - R[1];
      LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_givens_H, &H[k], ld_H,
        tempf_matrix, 2);
      mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_givens_H,
        2, 1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
      LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z],
        ld_H, tempf_vecblk, 2);
      mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R, 2,
        tempf_vecblk, 2, 0.0, &B[z], ld_H);
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory_handle);
}

/* symmetric qr */

void mpf_qr_dsy_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b
)
{
  MPF_Int j = 0;
  double temp = 0.0;
  double norm_temp = 0.0;
  double R[4];
  double Hblk[6];
  double b_swap_1 = 0.0;
  double b_swap_2 = 0.0;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    norm_temp = cblas_dnrm2(2, &H[ld_H*j+j], 1);
    R[0] = H[ld_H*j+j] / norm_temp;
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1] / norm_temp;
    R[2] = - R[1];

    Hblk[0] = H[ld_H*j+j];
    Hblk[1] = H[ld_H*j+j+1];
    Hblk[2] = H[ld_H*(j+1)+j];
    Hblk[3] = H[ld_H*(j+1)+j+1];
    Hblk[4] = H[ld_H*(j+2)+j];
    Hblk[5] = H[ld_H*(j+2)+j+1];

    H[ld_H*j+j]       = R[0]*Hblk[0];
    temp              = R[2]*Hblk[1];
    H[ld_H*j+j]       = H[ld_H*j+j] + temp;
    H[ld_H*j+j+1]     = R[1]*Hblk[0];
    temp              = R[3]*Hblk[1];
    H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
    H[ld_H*(j+1)+j]   = R[0]*Hblk[2];
    temp              = R[2]*Hblk[3];
    H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
    H[ld_H*(j+1)+j+1] = R[1]*Hblk[2];
    temp              = R[3]*Hblk[3];
    H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;
    H[ld_H*(j+2)+j]   = R[0]*Hblk[4];
    temp              = R[2]*Hblk[5];
    H[ld_H*(j+2)+j]   = H[ld_H*(j+2)+j] + temp;
    H[ld_H*(j+2)+j+1] = R[1]*Hblk[4];
    temp              = R[3]*Hblk[5];
    H[ld_H*(j+2)+j+1] = H[ld_H*(j+2)+j+1] + temp;

    b_swap_1 = R[0]*b[j];
    temp     = R[2]*b[j+1];
    b_swap_1 = b_swap_1 + temp;
    b_swap_2 = R[1]*b[j];
    temp     = R[3]*b[j+1];
    b_swap_2 = b_swap_2 + temp;
    b[j]     = b_swap_1;
    b[j+1]   = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R */
  /* and applies givens rotations                                    */

  norm_temp = cblas_dnrm2(2, &H[ld_H*j+j], 1);
  R[0] = H[ld_H*j+j] / norm_temp;
  R[3] = R[0];
  R[1] = - H[ld_H*j+j+1] / norm_temp;
  R[2] = - R[1];

  Hblk[0] = H[ld_H*j+j];
  Hblk[1] = H[ld_H*j+j+1];
  Hblk[2] = H[ld_H*(j+1)+j];
  Hblk[3] = H[ld_H*(j+1)+j+1];

  H[ld_H*j+j]       = R[0]*Hblk[0];
  temp              = R[2]*Hblk[1];
  H[ld_H*j+j]       = H[ld_H*j+j] + temp;
  H[ld_H*j+j+1]     = R[1]*Hblk[0];
  temp              = R[3]*Hblk[1];
  H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
  H[ld_H*(j+1)+j]   = R[0]*Hblk[2];
  temp              = R[2]*Hblk[3];
  H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
  H[ld_H*(j+1)+j+1] = R[1]*Hblk[2];
  temp              = R[3]*Hblk[3];
  H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;

  b_swap_1      = R[0]*b[j];
  temp          = R[2]*b[j+1];
  b_swap_1      = b_swap_1 + temp;
  b_swap_2      = R[1]*b[j];
  temp          = R[3]*b[j+1];
  b_swap_2      = b_swap_2 + temp;
  b[j]   = b_swap_1;
  b[j+1] = b_swap_2;
}

/* new qr_dsy_givens to that saves reflectors */
void mpf_qr_dsy_ref_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
)
{
  MPF_Int j = 0;
  double temp = 0.0;
  double norm_temp = 0.0;
  double R[4];
  double Hblk[6];
  double b_swap_1 = 0.0;
  double b_swap_2 = 0.0;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    norm_temp = cblas_dnrm2(2, &H[ld_H*j+j], 1);
    R[0] = H[ld_H*j+j] / norm_temp;
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1] / norm_temp;
    R[2] = - R[1];

    /* save refs_array */
    refs_array[j] = acos(R[0]);

    Hblk[0] = H[ld_H*j+j];
    Hblk[1] = H[ld_H*j+j+1];
    Hblk[2] = H[ld_H*(j+1)+j];
    Hblk[3] = H[ld_H*(j+1)+j+1];
    Hblk[4] = H[ld_H*(j+2)+j];
    Hblk[5] = H[ld_H*(j+2)+j+1];

    H[ld_H*j+j]       = R[0]*Hblk[0];
    temp              = R[2]*Hblk[1];
    H[ld_H*j+j]       = H[ld_H*j+j] + temp;
    H[ld_H*j+j+1]     = R[1]*Hblk[0];
    temp              = R[3]*Hblk[1];
    H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
    H[ld_H*(j+1)+j]   = R[0]*Hblk[2];
    temp              = R[2]*Hblk[3];
    H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
    H[ld_H*(j+1)+j+1] = R[1]*Hblk[2];
    temp              = R[3]*Hblk[3];
    H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;
    H[ld_H*(j+2)+j]   = R[0]*Hblk[4];
    temp              = R[2]*Hblk[5];
    H[ld_H*(j+2)+j]   = H[ld_H*(j+2)+j] + temp;
    H[ld_H*(j+2)+j+1] = R[1]*Hblk[4];
    temp              = R[3]*Hblk[5];
    H[ld_H*(j+2)+j+1] = H[ld_H*(j+2)+j+1] + temp;

    b_swap_1 = R[0]*b[j];
    temp     = R[2]*b[j+1];
    b_swap_1 = b_swap_1 + temp;
    b_swap_2 = R[1]*b[j];
    temp     = R[3]*b[j+1];
    b_swap_2 = b_swap_2 + temp;
    b[j]     = b_swap_1;
    b[j+1]   = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R */
  /* and applies givens rotations                                    */

  norm_temp = cblas_dnrm2(2, &H[ld_H*j+j], 1);
  R[0] = H[ld_H*j+j] / norm_temp;
  R[3] = R[0];
  R[1] = - H[ld_H*j+j+1] / norm_temp;
  R[2] = - R[1];

  /* save refs_array */
  refs_array[j] = acos(R[0]);

  Hblk[0] = H[ld_H*j+j];
  Hblk[1] = H[ld_H*j+j+1];
  Hblk[2] = H[ld_H*(j+1)+j];
  Hblk[3] = H[ld_H*(j+1)+j+1];

  H[ld_H*j+j]       = R[0]*Hblk[0];
  temp              = R[2]*Hblk[1];
  H[ld_H*j+j]       = H[ld_H*j+j] + temp;
  H[ld_H*j+j+1]     = R[1]*Hblk[0];
  temp              = R[3]*Hblk[1];
  H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
  H[ld_H*(j+1)+j]   = R[0]*Hblk[2];
  temp              = R[2]*Hblk[3];
  H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
  H[ld_H*(j+1)+j+1] = R[1]*Hblk[2];
  temp              = R[3]*Hblk[3];
  H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;

  b_swap_1 = R[0]*b[j];
  temp     = R[2]*b[j+1];
  b_swap_1 = b_swap_1 + temp;
  b_swap_2 = R[1]*b[j];
  temp     = R[3]*b[j+1];
  b_swap_2 = b_swap_2 + temp;
  b[j]     = b_swap_1;
  b[j+1]   = b_swap_2;
}

void mpf_qr_dsy_rhs_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
)
{
  MPF_Int j = 0;
  double temp = 0.0;
  double R[4];
  double b_swap_1 = 0.0;
  double b_swap_2 = 0.0;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    R[0] = cos(refs_array[j]);
    R[3] = R[0];
    R[1] = - sin(refs_array[j]);
    R[2] = - R[1];

    b_swap_1 = R[0]*b[j];
    temp     = R[2]*b[j+1];
    b_swap_1 = b_swap_1 + temp;
    b_swap_2 = R[1]*b[j];
    temp     = R[3]*b[j+1];
    b_swap_2 = b_swap_2 + temp;
    b[j]     = b_swap_1;
    b[j+1]   = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R */
  /* and applies givens rotations                                    */

  R[0] = cos(refs_array[j]);
  R[3] = R[0];
  R[1] = - sin(refs_array[j]);
  R[2] = - R[1];

  b_swap_1 = R[0]*b[j];
  temp     = R[2]*b[j+1];
  b_swap_1 = b_swap_1 + temp;
  b_swap_2 = R[1]*b[j];
  temp     = R[3]*b[j+1];
  b_swap_2 = b_swap_2 + temp;
  b[j]     = b_swap_1;
  b[j+1]   = b_swap_2;
}

void mpf_qr_dsy_mrhs_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  double *H,
  const MPF_Int ld_H,
  double *b,
  double *refs_array
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  double temp = 0.0;
  double R[4];
  double b_swap_1 = 0.0;
  double b_swap_2 = 0.0;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    R[0] = cos(refs_array[j]);
    R[3] = R[0];
    R[1] = - sin(refs_array[j]);
    R[2] = - R[1];

    for (i = 0; i < n_B; ++i)
    {
      b_swap_1 = R[0]*b[m_H*i+j];
      temp     = R[2]*b[m_H*i+j+1];
      b_swap_1 = b_swap_1 + temp;
      b_swap_2 = R[1]*b[m_H*i+j];
      temp     = R[3]*b[m_H*i+j+1];
      b_swap_2 = b_swap_2 + temp;
      b[m_H*i+j]     = b_swap_1;
      b[m_H*i+j+1]   = b_swap_2;
    }
  }

  /* last iteration, constructs and applies givens rotation matrix R */
  /* and applies givens rotations                                    */

  R[0] = cos(refs_array[j]);
  R[3] = R[0];
  R[1] = - sin(refs_array[j]);
  R[2] = - R[1];

  for (i = 0; i < n_B; ++i)
  {
    b_swap_1 = R[0]*b[m_H*i+j];
    temp     = R[2]*b[m_H*i+j+1];
    b_swap_1 = b_swap_1 + temp;
    b_swap_2 = R[1]*b[m_H*i+j];
    temp     = R[3]*b[m_H*i+j+1];
    b_swap_2 = b_swap_2 + temp;
    b[m_H*i+j]     = b_swap_1;
    b[m_H*i+j+1]   = b_swap_2;
  }
}

void mpf_qr_ssy_givens
(
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  float *H,
  const MPF_Int ld_H,
  float *b
)
{
  MPF_Int j = 0;
  float temp = 0.0;
  float norm_temp = 0.0;
  float R[4];
  float Hblk[6];
  float b_swap_1 = 0.0;
  float b_swap_2 = 0.0;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    norm_temp = cblas_snrm2(2, &H[ld_H*j+j], 1);
    R[0] = H[ld_H*j+j] / norm_temp;
    R[3] = R[0];
    R[1] = - H[ld_H*j+j+1] / norm_temp;
    R[2] = - R[1];

    Hblk[0] = H[ld_H*j+j];
    Hblk[1] = H[ld_H*j+j+1];
    Hblk[2] = H[ld_H*(j+1)+j];
    Hblk[3] = H[ld_H*(j+1)+j+1];
    Hblk[4] = H[ld_H*(j+2)+j];
    Hblk[5] = H[ld_H*(j+2)+j+1];

    H[ld_H*j+j]       = R[0]*Hblk[0];
    temp              = R[2]*Hblk[1];
    H[ld_H*j+j]       = H[ld_H*j+j] + temp;
    H[ld_H*j+j+1]     = R[1]*Hblk[0];
    temp              = R[3]*Hblk[1];
    H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
    H[ld_H*(j+1)+j]   = R[0]*Hblk[2];
    temp              = R[2]*Hblk[3];
    H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
    H[ld_H*(j+1)+j+1] = R[1]*Hblk[2];
    temp              = R[3]*Hblk[3];
    H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;
    H[ld_H*(j+2)+j]   = R[0]*Hblk[4];
    temp              = R[2]*Hblk[5];
    H[ld_H*(j+2)+j]   = H[ld_H*(j+2)+j] + temp;
    H[ld_H*(j+2)+j+1] = R[1]*Hblk[4];
    temp              = R[3]*Hblk[5];
    H[ld_H*(j+2)+j+1] = H[ld_H*(j+2)+j+1] + temp;

    b_swap_1 = R[0]*b[j];
    temp     = R[2]*b[j+1];
    b_swap_1 = b_swap_1 + temp;
    b_swap_2 = R[1] * b[j];
    temp     = R[3] * b[j+1];
    b_swap_2 = b_swap_2 + temp;
    b[j]     = b_swap_1;
    b[j+1]   = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R and */
  /* applies givens rotations                                            */

  norm_temp = mpf_snrm2(2, &H[ld_H*j+j], 1);
  R[0] = H[ld_H*j+j] / norm_temp;
  R[3] = R[0];
  R[1] = - H[ld_H*j+j+1] / norm_temp;
  R[2] = - R[1];

  Hblk[0] = H[ld_H*j+j];
  Hblk[1] = H[ld_H*j+j+1];
  Hblk[2] = H[ld_H*(j+1)+j];
  Hblk[3] = H[ld_H*(j+1)+j+1];

  H[ld_H*j+j]       = R[0]*Hblk[0];
  temp              = R[2] * Hblk[1];
  H[ld_H*j+j]       = H[ld_H*j+j] + temp;
  H[ld_H*j+j+1]     = R[1] * Hblk[0];
  temp              = R[3] * Hblk[1];
  H[ld_H*j+j+1]     = H[ld_H*j+j+1] + temp;
  H[ld_H*(j+1)+j]   = R[0] * Hblk[2];
  temp              = R[2] * Hblk[3];
  H[ld_H*(j+1)+j]   = H[ld_H*(j+1)+j] + temp;
  H[ld_H*(j+1)+j+1] = R[1] * Hblk[2];
  temp              = R[3] * Hblk[3];
  H[ld_H*(j+1)+j+1] = H[ld_H*(j+1)+j+1] + temp;

  b_swap_1 = R[0]*b[j];
  temp     = R[2]*b[j+1];
  b_swap_1 = b_swap_1 + temp;
  b_swap_2 = R[1]*b[j];
  temp     = R[3]*b[j+1];
  b_swap_2 = b_swap_2 + temp;
  b[j]     = b_swap_1;
  b[j+1]   = b_swap_2;
}

/* complex symmetric block */

void mpf_qr_block_givens_dsy
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B
)
{
  MPF_Int j, i;
  MPF_Int counter = 1;
  MPF_Int n_H_givens = 3 * blk;
  double *memory_handle = (double*)mpf_malloc((sizeof *memory_handle) * (2*3*blk+2*n_B)); /* @OPTIMIZE: should be allocated beforehand */
  double *tempf_matrix = memory_handle;
  double *tempf_vecblk = memory_handle + 3*blk;

  if (n_H >= blk*3)
  {
    for (j = 0; j < n_H - 3*blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j /(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; i++)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
    }

    for (j = n_H-3*blk; j < n_H-blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = n_H-blk; j < n_H-1; j++)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j /(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk * 2)
  {
    for (j = 0; j < blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = blk; j < blk*2-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j / (n_H - blk)) * (j % (n_H - blk) + 1);
      MPF_Int z = j + blk - 1 - (j / (n_H - blk)) * (j % (n_H - blk) + 1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk-counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk)
  {
    for (j = 0; j < blk-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk-counter; i++)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory_handle);
}

void mpf_block_qr_ssy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  float *H,
  const MPF_Int ld_H,
  float *B,
  const MPF_Int ld_B
)
{
  MPF_Int j, i;
  MPF_Int counter = 1;
  MPF_Int n_H_givens = 3 * blk;
  float *memory_handle = (float*)mpf_malloc((sizeof *memory_handle)*(2*3*blk + 2*n_B));
  float *tempf_matrix = memory_handle;
  float *tempf_vecblk = memory_handle + 3 * blk;

  if (n_H >= blk * 3)
  {
    for (j = 0; j < n_H - 3*blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
    }

    for (j = n_H-3*blk; j < n_H-blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = n_H-blk; j < n_H-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk-counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk * 2)
  {
    for (j = 0; j < blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j /(n_H-blk))*(j%(n_H-blk)+1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = blk; j < blk*2-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+ 1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk-counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk)
  {
    for (j = 0; j < blk-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      float R[4];
      float norm_temp;
      for (i = 0; i < blk-counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        mpf_slacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B, tempf_vecblk, 2);
        mpf_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory_handle);
}

/* complex symmetric qr */

void mpf_qr_zge_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
    MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
    MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
    MPF_ComplexDouble tempf_complex = ZERO_C;
    MPF_Int j = 0;

    for (j = 0; j < n_H; ++j)
    {
      MPF_ComplexDouble tempf_vector[2];
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble swap[2];
      int n_H_givens;

      /* computes rotation matrix R */
      n_H_givens = n_H - j;
      R[0]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
      tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
      R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
      mpf_vectorized_z_sqrt(1, &R[0], &R[0]);
      R[0]  = mpf_scalar_z_divide(H[m_H*j+j], R[0]);
      R[3]  = R[0];
      R[1]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
      tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
      R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
      mpf_vectorized_z_sqrt(1, &R[1], &R[1]);
      R[1]  = mpf_scalar_z_divide(H[m_H*j+j+1], R[1]);
      R[1]  = mpf_scalar_z_invert_sign(R[1]);
      R[2]  = mpf_scalar_z_invert_sign(R[1]);

      /* applie R to H */
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens,
        2, &ONE_C, R, 2, &H[m_H*j+j], m_H, &ZERO_C, tempf_matrix, 2);
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
        &H[m_H*j+j], m_H);

      /* applie R to B */
      tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[j]);
      swap[0]        = mpf_scalar_z_multiply(R[2], b[j+1]);
      tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[j]);
      swap[1]        = mpf_scalar_z_multiply(R[3], b[j+1]);
      b[j]    = mpf_scalar_z_add(tempf_vector[0], swap[0]);
      b[j+1]  = mpf_scalar_z_add(tempf_vector[1], swap[1]);
    }
}

/* @NOTE: updates previous version by enabling H to have different lead  */
/* dimension than m_H. The formating is still ordered by columns however */
/* the leading dimenion is now and can be ld_H != m_H.                   */

void mpf_qr_givens_zge_2
(
  MPF_ComplexDouble *H,
  MPF_Int ld_H,
  MPF_ComplexDouble *b,
  MPF_Int ld_b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0); 
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_Int j = 0;

  for (j = 0; j < n_H; ++j)
  {
    MPF_ComplexDouble tempf_vector[2];
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble swap[2];
    int n_H_givens;

    /* creates rotation matrix R */
    n_H_givens = n_H - j;
    R[0]  = mpf_scalar_z_multiply(H[ld_H*j+j], H[ld_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+j+1], H[ld_H*j+j+1]);
    R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
    mpf_vectorized_z_sqrt(1, &R[0], &R[0]);
    R[0]  = mpf_scalar_z_divide(H[ld_H*j+j], R[0]);
    R[3]  = R[0];
    R[1]  = mpf_scalar_z_multiply(H[ld_H*j+j], H[ld_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+j+1], H[ld_H*j+j+1]);
    R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
    mpf_vectorized_z_sqrt(1, &R[1], &R[1]);
    R[1]  = mpf_scalar_z_divide(H[ld_H*j+j+1], R[1]);
    R[1]  = mpf_scalar_z_invert_sign(R[1]);
    R[2]  = mpf_scalar_z_invert_sign(R[1]);

    /* applies R to H */
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
      &ONE_C, R, 2, &H[ld_H*j+j], m_H, &ZERO_C, tempf_matrix, 2);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
      &H[ld_H*j+j], m_H);

    /* applies R to b*/
    tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[j]);
    swap[0]        = mpf_scalar_z_multiply(R[2], b[j+1]);
    tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[j]);
    swap[1]        = mpf_scalar_z_multiply(R[3], b[j+1]);
    b[j]   = mpf_scalar_z_add(tempf_vector[0], swap[0]);
    b[j+1] = mpf_scalar_z_add(tempf_vector[1], swap[1]);
  }
}

/* UPDATES mpf_qr_givens_zge_factorize_2 to make QR general (as opposed to QR */
/* for tridiagonal matrices). Similar to simple and _2                       */
/* version for dge matrices. Have to add pivoting at some point.             */

void mpf_qr_zge_givens_3
(
  MPF_ComplexDouble *H,
  MPF_Int ld_H,
  MPF_ComplexDouble *b,
  MPF_Int ld_b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_Int j = 0;
  MPF_Int i = 0;

  for (j = 0; j < n_H; ++j)
  {
    MPF_ComplexDouble tempf_vector[2];
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble swap[2];
    int n_H_givens = 0;

    for (i = m_H-2; i >= j; --i)
    {
      /* creates rotation matrix R */
      n_H_givens = n_H - j;
      R[0]  = mpf_scalar_z_multiply(H[ld_H*j+i], H[ld_H*j+i]);
      tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+i+1], H[ld_H*j+i+1]);
      R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
      mpf_vectorized_z_sqrt(1, &R[0], &R[0]);
      R[0]  = mpf_scalar_z_divide(H[ld_H*j+i], R[0]);
      R[3]  = R[0];
      R[1]  = mpf_scalar_z_multiply(H[ld_H*j+i], H[ld_H*j+i]);
      tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+i+1], H[ld_H*j+i+1]);
      R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
      mpf_vectorized_z_sqrt(1, &R[1], &R[1]);
      R[1]  = mpf_scalar_z_divide(H[ld_H*j+i+1], R[1]);
      R[1]  = mpf_scalar_z_invert_sign(R[1]);
      R[2]  = mpf_scalar_z_invert_sign(R[1]);

      /* applied R to H*/
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
        &ONE_C, R, 2, &H[ld_H*j+i], m_H, &ZERO_C, tempf_matrix, 2);
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
        &H[ld_H*j+i], m_H);

      /* applies R to b */
      tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[i]);
      swap[0]        = mpf_scalar_z_multiply(R[2], b[i+1]);
      tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[i]);
      swap[1]        = mpf_scalar_z_multiply(R[3], b[i+1]);
      b[i]    = mpf_scalar_z_add(tempf_vector[0], swap[0]);
      b[i+1]  = mpf_scalar_z_add(tempf_vector[1], swap[1]);
    }
  }
}

void mpf_zge_qr_givens_mrhs
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_Int j = 0;
  MPF_Int i = 0;

  for (j = 0; j < n_H; ++j)
  {
    MPF_ComplexDouble tempf_vector[2];
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble swap[2];
    int n_H_givens;

    /* creates rotation matrix R */
    n_H_givens = n_H - j;
    R[0]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
    R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
    mpf_vectorized_z_sqrt(1, &R[0], &R[0]);
    R[0]  = mpf_scalar_z_divide(H[m_H*j+j], R[0]);
    R[3]  = R[0];
    R[1]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+2], H[m_H*j+j+1]);
    R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
    mpf_vectorized_z_sqrt(1, &R[1], &R[1]);
    R[1]  = mpf_scalar_z_divide(H[m_H*j+j+1], R[1]);
    R[1]  = mpf_scalar_z_invert_sign(R[1]);
    R[2]  = mpf_scalar_z_invert_sign(R[1]);

    /* applies R to H */
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
      &ONE_C, R, 2, &H[m_H*j+j], m_H, &ZERO_C, tempf_matrix, 2);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
      &H[m_H*j+j], m_H);

    /* applies R to all columns of B */
    for (i = 0; i < n_B; ++i)
    {
      tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[n_B*i+j]);
      swap[0]        = mpf_scalar_z_multiply(R[2], b[n_B*i+j+1]);
      tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[n_B*i+j]);
      swap[1]        = mpf_scalar_z_multiply(R[3], b[n_B*i+j+1]);
      b[n_B*i+j]    = mpf_scalar_z_add(tempf_vector[0], swap[0]);
      b[n_B*i+j+1]  = mpf_scalar_z_add(tempf_vector[1], swap[1]);
    }
  }
}

void mpf_qr_zge_givens_mrhs_2
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_Int ld_H = m_H;
  MPF_Int j = 0;
  MPF_Int i = 0;
  MPF_Int k = 0;

  for (j = 0; j < n_H; ++j)
  {
    MPF_ComplexDouble tempf_vector[2];
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble swap[2];
    int n_H_givens;

    for (i = m_H-2; i >= j; --i)
    {
      /* creates rotation matrix R */
      n_H_givens = n_H - j;
      R[0]  = mpf_scalar_z_multiply(H[ld_H*j+i], H[ld_H*j+i]);
      tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+i+1], H[ld_H*j+i+1]);
      R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
      mpf_vectorized_z_sqrt(1, &R[0], &R[0]);
      R[0]  = mpf_scalar_z_divide(H[ld_H*j+i], R[0]);
      R[3]  = R[0];
      R[1]  = mpf_scalar_z_multiply(H[ld_H*j+i], H[ld_H*j+i]);
      tempf_complex = mpf_scalar_z_multiply(H[ld_H*j+i+1], H[ld_H*j+i+1]);
      R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
      mpf_vectorized_z_sqrt(1, &R[1], &R[1]);
      R[1]  = mpf_scalar_z_divide(H[ld_H*j+i+1], R[1]);
      R[1]  = mpf_scalar_z_invert_sign(R[1]);
      R[2]  = mpf_scalar_z_invert_sign(R[1]);

      /* applies R to H */
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
        &ONE_C, R, 2, &H[ld_H*j+i], m_H, &ZERO_C, tempf_matrix, 2);
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
        &H[ld_H*j+i], m_H);

      /* applie R to B*/
      for (k = 0; k < n_B; ++k)
      {
        tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[ld_H*k+i]);
        swap[0]        = mpf_scalar_z_multiply(R[2], b[ld_H*k+i+1]);
        tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[ld_H*k+i]);
        swap[1]        = mpf_scalar_z_multiply(R[3], b[ld_H*k+i+1]);
        b[ld_H*k+i]   = mpf_scalar_z_add(tempf_vector[0], swap[0]);
        b[ld_H*k+i+1] = mpf_scalar_z_add(tempf_vector[1], swap[1]);
      }
    }
  }
}

/* @BUG: assumes m_H == ld_H */
void mpf_qr_givens_cge
(
  MPF_Complex *H,
  MPF_Complex *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  MPF_Complex *tempf_matrix
)
{
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex ONE_C  = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Int j = 0;

  for (j = 0; j < n_H; ++j)
  {
    MPF_Complex tempf_vector[2];
    MPF_Complex R[4];
    MPF_Complex swap[2];
    int n_H_givens;

    /* creates rotation matrix R */
    n_H_givens = n_H - j;
    R[0]  = mpf_scalar_c_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_c_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
    R[0]  = mpf_scalar_c_add(tempf_complex, R[0]);
    mpf_vectorized_c_sqrt(1, &R[0], &R[0]);
    R[0]  = mpf_scalar_c_divide(H[m_H*j+j], R[0]);
    R[3]  = R[0];
    R[1]  = mpf_scalar_c_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_c_multiply(H[m_H*j+j+2], H[m_H*j+j+1]);
    R[1]  = mpf_scalar_c_add(R[1], tempf_complex);
    mpf_vectorized_c_sqrt(1, &R[1], &R[1]);
    R[1]  = mpf_scalar_c_divide(H[m_H*j+j+1], R[1]);
    R[1]  = mpf_scalar_c_invert_sign(R[1]);
    R[2]  = mpf_scalar_c_invert_sign(R[1]);

    /* applies rotation matrix R to H */
    mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
      &ONE_C, R, 2, &H[m_H*j+j], m_H, &ZERO_C, tempf_matrix, 2);
    mpf_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
      &H[m_H*j+j], m_H);

    /* applies rotattion matrix R to b */
    tempf_vector[0] = mpf_scalar_c_multiply(R[0], b[j]);
    swap[0]        = mpf_scalar_c_multiply(R[2], b[j+1]);
    tempf_vector[1] = mpf_scalar_c_multiply(R[1], b[j]);
    swap[1]        = mpf_scalar_c_multiply(R[3], b[j+1]);
    b[j]    = mpf_scalar_c_add(tempf_vector[0], swap[0]);
    b[j+1]  = mpf_scalar_c_add(tempf_vector[1], swap[1]);
  }
}

void mpf_qr_zsy_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B
)
{
  MPF_Int j = 0;
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble R[4];
  MPF_ComplexDouble Hblk[6];
  MPF_ComplexDouble b_swap_1 = ZERO_C;
  MPF_ComplexDouble b_swap_2 = ZERO_C;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    /* creates rotation matrix R */
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
      &H[m_H*j+j], m_H, &H[m_H*j+j], m_H, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
    R[0]  = mpf_scalar_z_divide(H[m_H*j+j], tempf_complex);
    R[3]  = R[0];
    tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
    R[1]  = mpf_scalar_z_divide(H[m_H*j+j+1], tempf_complex);
    R[2]  = mpf_scalar_z_invert_sign(R[1]);

    /* applies R to H */
    Hblk[0] = H[m_H*j+j];
    Hblk[1] = H[m_H*j+j+1];
    Hblk[2] = H[m_H*(j+1)+j];
    Hblk[3] = H[m_H*(j+1)+j+1];
    Hblk[4] = H[m_H*(j+2)+j];
    Hblk[5] = H[m_H*(j+2)+j+1];
    H[m_H*j+j]       = mpf_scalar_z_multiply(R[0], Hblk[0]);
    tempf_complex     = mpf_scalar_z_multiply(R[2], Hblk[1]);
    H[m_H*j+j]       = mpf_scalar_z_add(H[m_H*j+j], tempf_complex);
    H[m_H*j+j+1]     = mpf_scalar_z_multiply(R[1], Hblk[0]);
    tempf_complex     = mpf_scalar_z_multiply(R[3], Hblk[1]);
    H[m_H*j+j+1]     = mpf_scalar_z_add(H[m_H*j+j+1], tempf_complex);
    H[m_H*(j+1)+j]   = mpf_scalar_z_multiply(R[0], Hblk[2]);
    tempf_complex     = mpf_scalar_z_multiply(R[2], Hblk[3]);
    H[m_H*(j+1)+j]   = mpf_scalar_z_add(H[m_H*(j+1)+j], tempf_complex);
    H[m_H*(j+1)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[2]);
    tempf_complex     = mpf_scalar_z_multiply(R[3], Hblk[3]);
    H[m_H*(j+1)+j+1] = mpf_scalar_z_add(H[m_H*(j+1)+j+1], tempf_complex);
    H[m_H*(j+2)+j]   = mpf_scalar_z_multiply(R[0], Hblk[4]);
    tempf_complex     = mpf_scalar_z_multiply(R[2], Hblk[5]);
    H[m_H*(j+2)+j]   = mpf_scalar_z_add(H[m_H*(j+2)+j], tempf_complex);
    H[m_H*(j+2)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[4]);
    tempf_complex     = mpf_scalar_z_multiply(R[3], Hblk[5]);
    H[m_H*(j+2)+j+1] = mpf_scalar_z_add(H[m_H*(j+2)+j+1], tempf_complex);

    /* applies R to b */
    b_swap_1      = mpf_scalar_z_multiply(R[0], b[j]);
    tempf_complex  = mpf_scalar_z_multiply(R[2], b[j+1]);
    b_swap_1      = mpf_scalar_z_add(b_swap_1, tempf_complex);
    b_swap_2      = mpf_scalar_z_multiply(R[1], b[j]);
    tempf_complex  = mpf_scalar_z_multiply(R[3], b[j+1]);
    b_swap_2      = mpf_scalar_z_add(b_swap_2, tempf_complex);
    b[j]   = b_swap_1;
    b[j+1] = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R and */
  /* applies givens rotations                                            */

  /* creates rotation matrix R */
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
    &H[m_H*j+j], m_H, &H[m_H*j+j], m_H, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
  R[0]  = mpf_scalar_z_divide(H[m_H*j+j], tempf_complex);
  R[3]  = R[0];
  tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
  R[1]  = mpf_scalar_z_divide(H[m_H*j+j+1], tempf_complex);
  R[2]  = mpf_scalar_z_invert_sign(R[1]);

  /* applies R to H */
  Hblk[0] = H[m_H*j+j];
  Hblk[1] = H[m_H*j+j+1];
  Hblk[2] = H[m_H*(j+1)+j];
  Hblk[3] = H[m_H*(j+1)+j+1];
  H[m_H*j+j]       = mpf_scalar_z_multiply(R[0], Hblk[0]);
  tempf_complex     = mpf_scalar_z_multiply(R[2], Hblk[1]);
  H[m_H*j+j]       = mpf_scalar_z_add(H[m_H*j+j], tempf_complex);
  H[m_H*j+j+1]     = mpf_scalar_z_multiply(R[1], Hblk[0]);
  tempf_complex     = mpf_scalar_z_multiply(R[3], Hblk[1]);
  H[m_H*j+j+1]     = mpf_scalar_z_add(H[m_H*j+j+1], tempf_complex);
  H[m_H*(j+1)+j]   = mpf_scalar_z_multiply(R[0], Hblk[2]);
  tempf_complex     = mpf_scalar_z_multiply(R[2], Hblk[3]);
  H[m_H*(j+1)+j]   = mpf_scalar_z_add(H[m_H*(j+1)+j], tempf_complex);
  H[m_H*(j+1)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[2]);
  tempf_complex     = mpf_scalar_z_multiply(R[3], Hblk[3]);
  H[m_H*(j+1)+j+1] = mpf_scalar_z_add(H[m_H*(j+1)+j+1], tempf_complex);

  /* applies R to b */
  b_swap_1      = mpf_scalar_z_multiply(R[0], b[j]);
  tempf_complex  = mpf_scalar_z_multiply(R[2], b[j+1]);
  b_swap_1      = mpf_scalar_z_add(b_swap_1, tempf_complex);
  b_swap_2      = mpf_scalar_z_multiply(R[1], b[j]);
  tempf_complex  = mpf_scalar_z_multiply(R[3], b[j+1]);
  b_swap_2      = mpf_scalar_z_add(b_swap_2, tempf_complex);
  b[j]   = b_swap_1;
  b[j+1] = b_swap_2;
}

/* NOTE: minor change relative to mpf_qr_givens_zsy_factorize in order to */
/* work when m_H != ld_H.                                                */
void mpf_qr_zsy_givens_2
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  const MPF_Int ld_H,
  const MPF_Int n_H,
  const MPF_Int n_B
)
{
  MPF_Int j = 0;
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble R[4];
  MPF_ComplexDouble Hblk[6];
  MPF_ComplexDouble b_swap_1 = ZERO_C;
  MPF_ComplexDouble b_swap_2 = ZERO_C;
  MPF_Int m_H = n_H;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    /* creates rotiation matrix R */
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
      &H[m_H*j+j], ld_H, &H[m_H*j+j], ld_H, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
    R[0]  = mpf_scalar_z_divide(H[ld_H*j+j], tempf_complex);
    R[3]  = R[0];
    tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
    R[1]  = mpf_scalar_z_divide(H[ld_H*j+j+1], tempf_complex);
    R[2]  = mpf_scalar_z_invert_sign(R[1]);

    /* applies R to H */
    Hblk[0] = H[ld_H*j+j];
    Hblk[1] = H[ld_H*j+j+1];
    Hblk[2] = H[ld_H*(j+1)+j];
    Hblk[3] = H[ld_H*(j+1)+j+1];
    Hblk[4] = H[ld_H*(j+2)+j];
    Hblk[5] = H[ld_H*(j+2)+j+1];
    H[m_H*j+j]        = mpf_scalar_z_multiply(R[0], Hblk[0]);
    tempf_complex      = mpf_scalar_z_multiply(R[2], Hblk[1]);
    H[ld_H*j+j]       = mpf_scalar_z_add(H[ld_H*j+j], tempf_complex);
    H[ld_H*j+j+1]     = mpf_scalar_z_multiply(R[1], Hblk[0]);
    tempf_complex      = mpf_scalar_z_multiply(R[3], Hblk[1]);
    H[ld_H*j+j+1]     = mpf_scalar_z_add(H[m_H*j+j+1], tempf_complex);
    H[ld_H*(j+1)+j]   = mpf_scalar_z_multiply(R[0], Hblk[2]);
    tempf_complex      = mpf_scalar_z_multiply(R[2], Hblk[3]);
    H[ld_H*(j+1)+j]   = mpf_scalar_z_add(H[ld_H*(j+1)+j], tempf_complex);
    H[ld_H*(j+1)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[2]);
    tempf_complex      = mpf_scalar_z_multiply(R[3], Hblk[3]);
    H[ld_H*(j+1)+j+1] = mpf_scalar_z_add(H[ld_H*(j+1)+j+1], tempf_complex);
    H[ld_H*(j+2)+j]   = mpf_scalar_z_multiply(R[0], Hblk[4]);
    tempf_complex      = mpf_scalar_z_multiply(R[2], Hblk[5]);
    H[ld_H*(j+2)+j]   = mpf_scalar_z_add(H[ld_H*(j+2)+j], tempf_complex);
    H[ld_H*(j+2)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[4]);
    tempf_complex      = mpf_scalar_z_multiply(R[3], Hblk[5]);
    H[ld_H*(j+2)+j+1] = mpf_scalar_z_add(H[ld_H*(j+2)+j+1], tempf_complex);

    /* applies R to b */
    b_swap_1      = mpf_scalar_z_multiply(R[0], b[j]);
    tempf_complex  = mpf_scalar_z_multiply(R[2], b[j+1]);
    b_swap_1      = mpf_scalar_z_add(b_swap_1, tempf_complex);
    b_swap_2      = mpf_scalar_z_multiply(R[1], b[j]);
    tempf_complex  = mpf_scalar_z_multiply(R[3], b[j+1]);
    b_swap_2      = mpf_scalar_z_add(b_swap_2, tempf_complex);
    b[j]   = b_swap_1;
    b[j+1] = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R and */
  /* applies givens rotations                                            */

  /* creates rotation matrix R*/
  mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
    &H[ld_H*j+j], ld_H, &H[ld_H*j+j], ld_H, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_z_sqrt(1, &tempf_complex, &tempf_complex);
  R[0]  = mpf_scalar_z_divide(H[ld_H*j+j], tempf_complex);
  R[3]  = R[0];
  tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
  R[1]  = mpf_scalar_z_divide(H[ld_H*j+j+1], tempf_complex);
  R[2]  = mpf_scalar_z_invert_sign(R[1]);

  /* applies R to H */
  Hblk[0] = H[ld_H*j+j];
  Hblk[1] = H[ld_H*j+j+1];
  Hblk[2] = H[ld_H*(j+1)+j];
  Hblk[3] = H[ld_H*(j+1)+j+1];
  H[m_H*j+j]       = mpf_scalar_z_multiply(R[0], Hblk[0]);
  tempf_complex                   = mpf_scalar_z_multiply(R[2], Hblk[1]);
  H[m_H*j+j]       = mpf_scalar_z_add(H[ld_H*j+j], tempf_complex);
  H[m_H*j+j+1]     = mpf_scalar_z_multiply(R[1], Hblk[0]);
  tempf_complex                   = mpf_scalar_z_multiply(R[3], Hblk[1]);
  H[m_H*j+j+1]     = mpf_scalar_z_add(H[ld_H*j+j+1], tempf_complex);
  H[m_H*(j+1)+j]   = mpf_scalar_z_multiply(R[0], Hblk[2]);
  tempf_complex                   = mpf_scalar_z_multiply(R[2], Hblk[3]);
  H[m_H*(j+1)+j]   = mpf_scalar_z_add(H[ld_H*(j+1)+j], tempf_complex);
  H[m_H*(j+1)+j+1] = mpf_scalar_z_multiply(R[1], Hblk[2]);
  tempf_complex                   = mpf_scalar_z_multiply(R[3], Hblk[3]);
  H[m_H*(j+1)+j+1] = mpf_scalar_z_add(H[ld_H*(j+1)+j+1], tempf_complex);

  /* applies R to b */
  b_swap_1     = mpf_scalar_z_multiply(R[0], b[j]);
  tempf_complex = mpf_scalar_z_multiply(R[2], b[j+1]);
  b_swap_1     = mpf_scalar_z_add(b_swap_1, tempf_complex);
  b_swap_2     = mpf_scalar_z_multiply(R[1], b[j]);
  tempf_complex = mpf_scalar_z_multiply(R[3], b[j+1]);
  b_swap_2     = mpf_scalar_z_add(b_swap_2, tempf_complex);
  b[j]   = b_swap_1;
  b[j+1] = b_swap_2;
}

void mpf_qr_csy_givens
(
  MPF_Complex *H,
  MPF_Complex *b,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B
)
{
  MPF_Int j = 0;
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex ONE_C  = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Complex R[4];
  MPF_Complex Hblk[6];
  MPF_Complex b_swap_1 = ZERO_C;
  MPF_Complex b_swap_2 = ZERO_C;
  MPF_Int ld_H = m_H;

  for (j = 0; j < m_H-2; ++j)
  {
    /* constructs and applies givens rotation matrix R and applies givens */
    /* rotations to H and right-hand side b                               */

    /* creates rotation matrix R */
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
      &H[ld_H*j+j], m_H, &H[ld_H*j+j], ld_H, &ZERO_C, &tempf_complex, 1);
    mpf_vectorized_c_sqrt(1, &tempf_complex, &tempf_complex);
    R[0]  = mpf_scalar_c_divide(H[ld_H*j+j], tempf_complex);
    R[3]  = R[0];
    tempf_complex = mpf_scalar_c_invert_sign(tempf_complex);
    R[1]  = mpf_scalar_c_divide(H[ld_H*j+j+1], tempf_complex);
    R[2]  = mpf_scalar_c_invert_sign(R[1]);

    /* applies rotation R to H */
    Hblk[0] = H[m_H*j+j];
    Hblk[1] = H[m_H*j+j+1];
    Hblk[2] = H[m_H*(j+1)+j];
    Hblk[3] = H[m_H*(j+1)+j+1];
    Hblk[4] = H[m_H*(j+2)+j];
    Hblk[5] = H[m_H*(j+2)+j+1];
    H[ld_H*j+j]       = mpf_scalar_c_multiply(R[0], Hblk[0]);
    tempf_complex      = mpf_scalar_c_multiply(R[2], Hblk[1]);
    H[ld_H*j+j]       = mpf_scalar_c_add(H[m_H*j+j], tempf_complex);
    H[ld_H*j+j+1]     = mpf_scalar_c_multiply(R[1], Hblk[0]);
    tempf_complex      = mpf_scalar_c_multiply(R[3], Hblk[1]);
    H[ld_H*j+j+1]     = mpf_scalar_c_add(H[m_H*j+j+1], tempf_complex);
    H[ld_H*(j+1)+j]   = mpf_scalar_c_multiply(R[0], Hblk[2]);
    tempf_complex      = mpf_scalar_c_multiply(R[2], Hblk[3]);
    H[ld_H*(j+1)+j]   = mpf_scalar_c_add(H[m_H*(j+1)+j], tempf_complex);
    H[ld_H*(j+1)+j+1] = mpf_scalar_c_multiply(R[1], Hblk[2]);
    tempf_complex      = mpf_scalar_c_multiply(R[3], Hblk[3]);
    H[ld_H*(j+1)+j+1] = mpf_scalar_c_add(H[m_H*(j+1)+j+1], tempf_complex);
    H[ld_H*(j+2)+j]   = mpf_scalar_c_multiply(R[0], Hblk[4]);
    tempf_complex      = mpf_scalar_c_multiply(R[2], Hblk[5]);
    H[ld_H*(j+2)+j]   = mpf_scalar_c_add(H[m_H*(j+2)+j], tempf_complex);
    H[ld_H*(j+2)+j+1] = mpf_scalar_c_multiply(R[1], Hblk[4]);
    tempf_complex      = mpf_scalar_c_multiply(R[3], Hblk[5]);
    H[ld_H*(j+2)+j+1] = mpf_scalar_c_add(H[m_H*(j+2)+j+1], tempf_complex);

    /* applies R to b */
    b_swap_1      = mpf_scalar_c_multiply(R[0], b[j]);
    tempf_complex  = mpf_scalar_c_multiply(R[2], b[j+1]);
    b_swap_1      = mpf_scalar_c_add(b_swap_1, tempf_complex);
    b_swap_2      = mpf_scalar_c_multiply(R[1], b[j]);
    tempf_complex  = mpf_scalar_c_multiply(R[3], b[j+1]);
    b_swap_2      = mpf_scalar_c_add(b_swap_2, tempf_complex);
    b[j]   = b_swap_1;
    b[j+1] = b_swap_2;
  }

  /* last iteration, constructs and applies givens rotation matrix R and */
  /* applies givens rotations                                            */

  /* creates rotation matrix R */
  mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 2, &ONE_C,
    &H[ld_H*j+j], ld_H, &H[ld_H*j+j], ld_H, &ZERO_C, &tempf_complex, 1);
  mpf_vectorized_c_sqrt(1, &tempf_complex, &tempf_complex);
  R[0] = mpf_scalar_c_divide(H[ld_H*j+j], tempf_complex);
  R[3] = R[0];
  tempf_complex = mpf_scalar_c_invert_sign(tempf_complex);
  R[1] = mpf_scalar_c_divide(H[ld_H*j+j+1], tempf_complex);
  R[2] = mpf_scalar_c_invert_sign(R[1]);

  /* applies R to H */
  Hblk[0] = H[m_H*j+j];
  Hblk[1] = H[m_H*j+j+1];
  Hblk[2] = H[m_H*(j+1)+j];
  Hblk[3] = H[m_H*(j+1)+j+1];
  H[ld_H*j+j]       = mpf_scalar_c_multiply(R[0], Hblk[0]);
  tempf_complex      = mpf_scalar_c_multiply(R[2], Hblk[1]);
  H[ld_H*j+j]       = mpf_scalar_c_add(H[m_H*j+j], tempf_complex);
  H[ld_H*j+j+1]     = mpf_scalar_c_multiply(R[1], Hblk[0]);
  tempf_complex      = mpf_scalar_c_multiply(R[3], Hblk[1]);
  H[ld_H*j+j+1]     = mpf_scalar_c_add(H[m_H*j+j+1], tempf_complex);
  H[ld_H*(j+1)+j]   = mpf_scalar_c_multiply(R[0], Hblk[2]);
  tempf_complex      = mpf_scalar_c_multiply(R[2], Hblk[3]);
  H[ld_H*(j+1)+j]   = mpf_scalar_c_add(H[m_H*(j+1)+j], tempf_complex);
  H[ld_H*(j+1)+j+1] = mpf_scalar_c_multiply(R[1], Hblk[2]);
  tempf_complex      = mpf_scalar_c_multiply(R[3], Hblk[3]);
  H[ld_H*(j+1)+j+1] = mpf_scalar_c_add(H[m_H*(j+1)+j+1], tempf_complex);

  /* applies R to b */
  b_swap_1     = mpf_scalar_c_multiply(R[0], b[j]);
  tempf_complex = mpf_scalar_c_multiply(R[2], b[j+1]);
  b_swap_1     = mpf_scalar_c_add(b_swap_1, tempf_complex);
  b_swap_2     = mpf_scalar_c_multiply(R[1], b[j]);
  tempf_complex = mpf_scalar_c_multiply(R[3], b[j+1]);
  b_swap_2     = mpf_scalar_c_add(b_swap_2, tempf_complex);
  b[j]   = b_swap_1;
  b[j+1] = b_swap_2;
}

void mpf_qr_zsy_hessen_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *b,
  MPF_Int m_H,
  MPF_Int n_H,
  MPF_Int n_B,
  MPF_ComplexDouble *tempf_matrix
)
{
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_Int j = 0;
  for (j = 0; j < n_H; ++j)
  {
    MPF_ComplexDouble tempf_vector[2];
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble swap[2];
    int n_H_givens;

    /* creates rotation matrix R */
    n_H_givens = n_H - j;
    R[0]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
    R[0]  = mpf_scalar_z_add(tempf_complex, R[0]);
    vzSqrt(1, &R[0], &R[0]);
    R[0]  = mpf_scalar_z_divide(H[m_H*j+j], R[0]);
    R[3]  = R[0];
    R[1]  = mpf_scalar_z_multiply(H[m_H*j+j], H[m_H*j+j]);
    tempf_complex = mpf_scalar_z_multiply(H[m_H*j+j+1], H[m_H*j+j+1]);
    R[1]  = mpf_scalar_z_add(R[1], tempf_complex);
    vzSqrt(1, &R[1], &R[1]);
    R[1]  = mpf_scalar_z_divide(H[m_H*j+j+1], R[1]);
    R[1]  = mpf_scalar_z_invert_sign(R[1]);
    R[2]  = mpf_scalar_z_invert_sign(R[1]);

    /* applies R to H */
    mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
      &ONE_C, R, 2, &H[m_H*j+j], m_H, &ZERO_C, tempf_matrix, 2);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, tempf_matrix, 2,
      &H[m_H*j+j], m_H);

    /* applies R to b */
    tempf_vector[0] = mpf_scalar_z_multiply(R[0], b[j]);
    swap[0]        = mpf_scalar_z_multiply(R[2], b[j+1]);
    tempf_vector[1] = mpf_scalar_z_multiply(R[1], b[j]);
    swap[1]        = mpf_scalar_z_multiply(R[3], b[j+1]);
    b[j]    = mpf_scalar_z_add(tempf_vector[0], swap[0]);
    b[j+1]  = mpf_scalar_z_add(tempf_vector[1], swap[1]);
  }
}

/* complex general block qr zge */

void mpf_block_qr_zge_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
)
{
  MPF_Int j, i;
  MPF_Int n_H_givens = n_H;
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;

  MPF_ComplexDouble *memory = (MPF_ComplexDouble*)mpf_malloc((sizeof*memory)*(2*n_H+2*n_B)); /* @OPTIMIZE: should be allocated beforehand */
  MPF_ComplexDouble *tempf_matrix = memory;
  MPF_ComplexDouble *tempf_vecblk = &memory[2*n_H];

  for (j = 0; j < n_H; ++j)
  {
    MPF_Int k = m_H*j + blk*(j/blk) + blk + j%blk;
    MPF_Int z = blk*(j/blk) + blk + j%blk;
    MPF_ComplexDouble R[4];
    MPF_ComplexDouble norm_temp;
    n_H_givens = n_H - j;

    for (i = 0; i < blk; ++i)
    {
      k = k - 1;
      z = z - 1;

      /* creates rotation matrix R */
      norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
      tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
      norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
      mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
      R[0] = mpf_scalar_z_divide(H[k], norm_temp);
      R[3] = R[0];
      R[1] = mpf_scalar_z_invert_sign(H[k+1]);
      R[1] = mpf_scalar_z_divide(R[1], norm_temp);
      R[2] = mpf_scalar_z_invert_sign(R[1]);

      /* applies R to H */
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], m_H,
        tempf_matrix, 2);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
        &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], m_H);
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], m_H, tempf_vecblk, 2);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C, R,
        2, tempf_vecblk, 2, &ZERO_C, &B[z], m_H);
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

void mpf_block_qr_zhe_givens
(
  MPF_ComplexDouble *H,
  MPF_ComplexDouble *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
)
{
  MPF_Int j = 0;
  MPF_Int i = 0;
  MPF_Int n_H_givens = n_H;
  MPF_ComplexDouble ONE_C  = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble *memory = (MPF_ComplexDouble*)mpf_malloc((sizeof *memory)*(2*n_H + 2*n_B));  /* @OPTIMIZE: should be allocated beforehand */
  MPF_ComplexDouble *tempf_matrix = memory;
  MPF_ComplexDouble *tempf_vecblk = memory + 2*n_H;

  for (j = 0; j < n_H; ++j)
  {
    MPF_Int k = m_H*j + blk*(j / blk) + blk + j % blk;
    MPF_Int z = blk*(j / blk) + blk + j % blk;
    MPF_ComplexDouble R[4];
    double norm_temp;
    n_H_givens = n_H - j;

    for (i = 0; i < blk; ++i)
    {
      k = k - 1;
      z = z - 1;
      //norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
      //tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
      //norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
      //mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
      /* creates rotation matrix R */
      norm_temp = mpf_dznrm2(2, H, 1);
      R[0] = mpf_scalar_z_normalize(H[k], norm_temp);
      R[3] = R[0];
      R[1] = mpf_scalar_z_invert_sign(H[k+1]);
      R[1] = mpf_scalar_z_normalize(R[1], norm_temp);
      R[2] = mpf_scalar_z_invert_sign(R[1]);

      /* applies R to H */
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], m_H,
        tempf_matrix, 2);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
        &ONE_C, R, 2, tempf_matrix, 2,
                  &ZERO_C, &H[k], m_H);
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], m_H, tempf_vecblk, 2);
      mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C, R,
        2, tempf_vecblk, 2, &ZERO_C, &B[z], m_H);
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

void mpf_block_qr_cge_givens
(
  MPF_Complex *H,
  MPF_Complex *B,
  const MPF_Int m_H,
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk
)
{
  MPF_Int j, i;
  MPF_Int n_H_givens = n_H;
  MPF_Complex ONE_C  = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Complex *memory = (MPF_Complex*)mpf_malloc((sizeof *memory)*(2*n_H + 2*n_B));
  MPF_Complex *tempf_matrix = memory;
  MPF_Complex *tempf_vecblk = memory + 2*n_H;

  for (j = 0; j < n_H; ++j)
  {
    MPF_Int k = m_H*j + blk*(j/blk) + blk + j%blk;
    MPF_Int z = blk*(j/blk) + blk + j%blk;
    MPF_Complex R[4];
    MPF_Complex norm_temp;
    n_H_givens = n_H - j;

    for (i = 0; i < blk; ++i)
    {
      k = k - 1;
      z = z - 1;

      /* creates rotation matrix R */
      norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
      tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
      norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
      mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
      R[0] = mpf_scalar_c_divide(H[k], norm_temp);
      R[3] = R[0];
      R[1] = mpf_scalar_c_invert_sign(H[k+1]);
      R[1] = mpf_scalar_c_divide(R[1], norm_temp);
      R[2] = mpf_scalar_c_invert_sign(R[1]);

      /* applies R to H */
      LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], m_H,
        tempf_matrix, 2);
      mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
        &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], m_H);

      /* applies R to B */
      LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], m_H, tempf_vecblk, 2);
      mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C, R,
        2, tempf_vecblk, 2, &ZERO_C, &B[z], m_H);
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

/* real symmetric block qr */

void mpf_block_qr_dsy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  double *H,
  const MPF_Int ld_H,
  double *B,
  const MPF_Int ld_B
)
{
  MPF_Int j = 0;
  MPF_Int i = 0;
  MPF_Int n_H_givens = 3 * blk;
  MPF_Int counter = 1;
  double *memory = (double*)mpf_malloc((sizeof *memory)*(2*3*blk + 2*n_B));
  double *tempf_matrix = memory;
  double *tempf_vecblk = memory+3*blk;

  if (n_H >= blk*3)
  {
    for (j = 0; j < n_H - 3*blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];

        /* applies R to H */
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0,
          R, 2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
    }

    for (j = n_H-3*blk; j < n_H-blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = n_H-blk; j < n_H-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk*2)
  {
    for (j = 0; j < blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = blk; j < blk*2-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j /(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }

    for (j = 0; j < blk-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      double R[4];
      double norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        norm_temp = sqrt(H[k]*H[k] + H[k+1]*H[k+1]);
        R[0] =   H[k] / norm_temp;
        R[3] =   R[0];
        R[1] = - H[k+1] / norm_temp;
        R[2] = - R[1];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          1.0, R, 2, tempf_matrix, 2, 0.0, &H[k], ld_H);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, 1.0, R,
          2, tempf_vecblk, 2, 0.0, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }

  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

/* complex symmetric block qr */

void mpf_block_qr_zsy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  MPF_ComplexDouble *H,
  const MPF_Int ld_H,
  MPF_ComplexDouble *B,
  const MPF_Int ld_B
)
{
  MPF_Int j, i;
  MPF_Int counter = 1;
  MPF_Int n_H_givens = blk*3;

  MPF_ComplexDouble *memory = (MPF_ComplexDouble*)mpf_malloc((sizeof *memory)*(2*3*blk + 2*n_B));
  MPF_ComplexDouble *tempf_matrix = memory;
  MPF_ComplexDouble *tempf_vecblk = memory + blk*3;
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;

  if (n_H >= blk*3)
  {
    for (j = 0; j < n_H - 3*blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens,
          2, &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2,
          &ONE_C, R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
    }

    for (j = n_H-3*blk; j < n_H-blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        vzSqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens,
          2, &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2,
          &ONE_C, R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = n_H-blk; j < n_H-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk*2)
  {
    for (j = 0; j < blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = blk; j < blk*2-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk)
  {
    for (j = 0; j < blk - 1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_ComplexDouble R[4];
      MPF_ComplexDouble norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        norm_temp = mpf_scalar_z_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_z_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_z_add(norm_temp, tempf_complex);
        mpf_vectorized_z_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_z_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_z_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_z_invert_sign(R[1]);
        R[2] = mpf_scalar_z_invert_sign(R[1]);
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);
        LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

void mpf_block_qr_csy_givens
(
  const MPF_Int n_H,
  const MPF_Int n_B,
  const MPF_Int blk,
  MPF_Complex *H,
  const MPF_Int ld_H,
  MPF_Complex *B,
  const MPF_Int ld_B
)
{
  MPF_Int j, i;
  MPF_Int counter = 1;
  MPF_Int n_H_givens = blk * 3;

  MPF_Complex *memory = (MPF_Complex*)mpf_malloc((sizeof *memory)*(2*3*blk + 2*n_B));
  MPF_Complex *tempf_matrix = memory;
  MPF_Complex *tempf_vecblk = memory + blk*3;
  MPF_Complex ONE_C = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex tempf_complex = ZERO_C;

  if (n_H >= blk*3)
  {
    for (j = 0; j < n_H - 3*blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2,
          &ONE_C, R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
    }

    for (j = n_H-3*blk; j < n_H-blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        vcSqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2,
          &ONE_C, R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = n_H-blk; j < n_H-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk*2)
  {
    for (j = 0; j < blk; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
    }

    for (j = blk; j < blk*2-1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  else if (n_H == blk)
  {
    for (j = 0; j < blk - 1; ++j)
    {
      MPF_Int k = ld_H*j + j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Int z = j + blk - 1 - (j/(n_H-blk))*(j%(n_H-blk)+1);
      MPF_Complex R[4];
      MPF_Complex norm_temp;
      for (i = 0; i < blk - counter; ++i)
      {
        /* creates rotation matrix R */
        norm_temp = mpf_scalar_c_multiply(H[k], H[k]);
        tempf_complex = mpf_scalar_c_multiply(H[k+1], H[k+1]);
        norm_temp = mpf_scalar_c_add(norm_temp, tempf_complex);
        mpf_vectorized_c_sqrt(1, &norm_temp, &norm_temp);
        R[0] = mpf_scalar_c_divide(H[k], norm_temp);
        R[3] = R[0];
        R[1] = mpf_scalar_c_divide(H[k+1], norm_temp);
        R[1] = mpf_scalar_c_invert_sign(R[1]);
        R[2] = mpf_scalar_c_invert_sign(R[1]);

        /* applies R to H */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_H_givens, &H[k], ld_H,
          tempf_matrix, 2);
        mpf_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_H_givens, 2,
          &ONE_C, R, 2, tempf_matrix, 2, &ZERO_C, &H[k], ld_H);

        /* applies R to B */
        LAPACKE_clacpy(LAPACK_COL_MAJOR, 'A', 2, n_B, &B[z], ld_B,
          tempf_vecblk, 2);
        mpf_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, n_B, 2, &ONE_C,
          R, 2, tempf_vecblk, 2, &ZERO_C, &B[z], ld_B);
        k = k - 1;
        z = z - 1;
      }
      --n_H_givens;
      ++counter;
    }
  }
  tempf_matrix = NULL;
  tempf_vecblk = NULL;
  mpf_free(memory);
}

void mpf_gram_schmidt_zge  /* @BUG: ld_B is missing */
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_ComplexDouble* B,
  MPF_ComplexDouble *H,
  const MPF_Int m_H
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble h_temp = ZERO_C;

  for (i = 0; i < n_B; ++i)
  {
    for (j = 0; j < i; ++j)
    {
      mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
        &B[m_B*j], m_B, &B[m_B*i], m_B, &ZERO_C, &tempf_complex, 1);
      H[m_H*i+j] = tempf_complex;
      tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
      mpf_zaxpy(m_B, &tempf_complex, &B[m_B*j], 1, &B[m_B*i], 1);
    }
    mpf_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      &B[m_B*i], m_B, &B[m_B*i], m_B, &ZERO_C, &h_temp, 1);
    mpf_vectorized_z_sqrt(1, &h_temp, &h_temp);
    H[m_H*i+i] = h_temp;
    h_temp = mpf_scalar_z_divide(ONE_C, h_temp);
    mpf_zscal(m_B, &h_temp, &B[m_B*i], 1);
  }
}

void mpf_gram_schmidt_zhe
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_ComplexDouble* B,
  MPF_ComplexDouble *H,
  const MPF_Int m_H
)
{
  MPF_Int i = 0;
  MPF_Int j = 0;
  MPF_ComplexDouble ZERO_C = mpf_scalar_z_init(0.0, 0.0);
  MPF_ComplexDouble ONE_C = mpf_scalar_z_init(1.0, 0.0);
  MPF_ComplexDouble tempf_complex = ZERO_C;
  MPF_ComplexDouble h_temp = ZERO_C;

  for (i = 0; i < n_B; ++i)
  {
    for (j = 0; j < i; ++j)
    {
      mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
        &B[m_B*j], m_B, &B[m_B*i], m_B, &ZERO_C, &tempf_complex, 1);
      H[m_H*i+j] = tempf_complex;
      tempf_complex = mpf_scalar_z_invert_sign(tempf_complex);
      mpf_zaxpy(m_B, &tempf_complex, &B[m_B*j], 1, &B[m_B*i], 1);
    }
    mpf_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      &B[m_B*i], m_B, &B[m_B*i], m_B, &ZERO_C, &h_temp, 1);
    mpf_vectorized_z_sqrt(1, &h_temp, &h_temp);
    H[m_H*i+i] = h_temp;
    h_temp = mpf_scalar_z_divide(ONE_C, h_temp);
    mpf_zscal(m_B, &h_temp, &B[m_B*i], 1);
  }
}

void mpf_gram_schmidt_cge
(
  const MPF_Int m_B,
  const MPF_Int n_B,
  MPF_Complex *B,
  MPF_Complex *H,
  const MPF_Int m_H
)
{
  MPF_Int i, j;
  MPF_Complex ZERO_C = mpf_scalar_c_init(0.0, 0.0);
  MPF_Complex ONE_C = mpf_scalar_c_init(1.0, 0.0);
  MPF_Complex tempf_complex = ZERO_C;
  MPF_Complex h_temp = ZERO_C;

  for (i = 0; i < n_B; ++i)
  {
    for (j = 0; j < i; ++j)
    {
      mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
        &B[m_B*j], m_B, &B[m_B*i], m_B, &ZERO_C, &tempf_complex, 1);
      H[m_H*i+j] = tempf_complex;
      tempf_complex = mpf_scalar_c_invert_sign(tempf_complex);
      mpf_caxpy(m_B, &tempf_complex, &B[m_B*j], 1, &B[m_B*i], 1);
    }
    mpf_cgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, m_B, &ONE_C,
      &B[m_B*i], m_B, &B[m_B*i], m_B, &ZERO_C, &h_temp, 1);
    mpf_vectorized_c_sqrt(1, &h_temp, &h_temp);
    H[m_H*i+i] = h_temp;
    h_temp = mpf_scalar_c_divide(ONE_C, h_temp);
    mpf_cscal(m_B, &h_temp, &B[m_B*i], 1);
  }
}
