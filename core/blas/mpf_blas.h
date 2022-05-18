#ifndef MPF_F_BLAS_H /* MPF_F_BLAS_h -- start */
#define MPF_F_BLAS_H

#if BLAS == MPF_F_BLAS_MKL
    #include "mpf_blas_mkl_internal.h"
#endif

extern sparse_status_t mpf_sparse_d_mv
(
  const MPF_SparseTranspose transpose_A,
  const double alpha,
  const MPF_MatrixSparseInternal A_matrix,
  const MPF_MatrixSparseDescriptor descriptor_A,
  const double *x_vector,
  const double beta,
  double *y_vector
);

extern sparse_status_t mpf_sparse_z_mv
(
  const MPF_SparseTranspose transpose_A,
  const MPF_Complex16 alpha,
  const MPF_MatrixSparseInternal A_matrix,
  const MPF_MatrixSparseDescriptor descriptor_A,
  const MPF_Complex16 *x_vector,
  const MPF_Complex16 beta,
  MPF_Complex16 *y_vector
);

extern sparse_status_t mpf_sparse_d_mm
(
  const MPF_SparseTranspose transpose_A,
  const double alpha,
  const MPF_MatrixSparseInternal A_matrix,
  const MPF_MatrixSparseDescriptor descriptor_A,
  const MPF_SparseLayout layout_A,
  const double *B_matrix,
  const MPF_Int num_cols_X,
  const MPF_Int lead_dim_B,
  const double beta,
  double *C_matrix,
  const MPF_Int lead_dim_C
);

extern sparse_status_t mpf_sparse_z_mm(
  const MPF_SparseTranspose transpose_A,
  const MPF_Complex16 alpha,
  const MPF_MatrixSparseInternal A_matrix,
  const MPF_MatrixSparseDescriptor descriptor_A,
  const MPF_SparseLayout layout_A,
  const MPF_Complex16 *B_matrix,
  const MPF_Int num_cols_X,
  const MPF_Int lead_dim_B,
  const MPF_Complex16 beta,
  MPF_Complex16 *C_matrix,
  const MPF_Int lead_dim_C
);

extern void mpf_dtrsm(const MPF_Layout layout_A,
                     const MPF_Side side_A,
                     const MPF_Uplo uplo_A,
                     const MPF_Transpose transpose_A,
                     const MPF_Diag diag_A,
                     const MPF_Int num_rows_A,
                     const MPF_Int num_cols_B,
                     double alpha,
                     const double *A_matrix,
                     const MPF_Int lead_dim_A,
                     double *B_matrix,
                     const MPF_Int lead_dim_B) NOTHROW;

extern MPF_LapackInt mpf_dgesv(int layout_A,
                            MPF_LapackInt num_rows_A,
                            MPF_LapackInt num_rhs,
                            double *A_matrix,
                            MPF_LapackInt lead_dim_A,
                            MPF_LapackInt *pivots_vector,
                            double *B_vecblk,
                            MPF_LapackInt lead_dim_B);

extern MPF_LapackInt mpf_zgesv(int layout_A,
                            MPF_LapackInt num_rows_A,
                            MPF_LapackInt num_rhs,
                            MPF_LapackComplexDouble *A_matrix,
                            MPF_LapackInt lead_dim_A,
                            MPF_LapackInt *pivots_vector,
                            MPF_LapackComplexDouble *B_vecblk,
                            MPF_LapackInt lead_dim_B);

extern void mpf_dgemv(const MPF_Layout layout_A,
                     const MPF_Transpose transpose_A,
                     const MPF_Int num_rows_A,
                     const MPF_Int num_cols_X,
                     const double alpha,
                     const double *A_matrix,
                     const MPF_Int lead_dim_A,
                     const double *x_vector,
                     const MPF_Int inc_x,
                     const double beta,
                     double *y_vector,
                     const MPF_Int inc_y) NOTHROW;

extern MPF_LapackInt mpf_dgeqrf(const MPF_LapackLayout layout_A,
                             const MPF_LapackInt num_rows_A,
                             const MPF_LapackInt num_cols_A,
                             double *A_matrix,
                             const MPF_LapackInt lead_dim_A,
                             double *reflectors);

extern MPF_LapackInt mpf_dlacpy(const MPF_LapackLayout layout_A,
                             const MPF_LapackUplo uplo_A,
                             const MPF_LapackInt num_rows_A,
                             const MPF_LapackInt num_cols_A,
                             const double *A_matrix,
                             const MPF_LapackInt lead_dim_A,
                             double *B_matrix,
                             const MPF_LapackInt lead_dim_B);

extern MPF_LapackInt mpf_dorgqr(MPF_LapackLayout,
                             MPF_LapackInt num_rows_A,
                             MPF_LapackInt num_cols_A,
                             MPF_LapackInt num_reflectors,
                             double *A_matrix,
                             MPF_LapackInt lead_dim_A,
                             const double *reflectors_vector);

extern void mpf_dgemm(const MPF_BlasLayout layout,
                     const MPF_BlasTranspose transpose_A,
                     const MPF_BlasTranspose transpose_B,
                     const MPF_Int num_rows_A,
                     const MPF_Int num_cols_B,
                     const MPF_Int num_cols_A,
                     const double alpha,
                     const double *A_matrix,
                     const MPF_Int lead_dim_A,
                     const double *B_matrix,
                     const MPF_Int lead_dim_B,
                     const double beta,
                     double *C_matrix,
                     const MPF_Int lead_dim_C) NOTHROW;

extern void mpf_zgemm(const MPF_BlasLayout layout,
                     const MPF_BlasTranspose transpose_A,
                     const MPF_BlasTranspose transpose_B,
                     const MPF_Int num_rows_A,
                     const MPF_Int num_cols_B,
                     const MPF_Int num_cols_A,
                     const void *alpha,
                     const void *A_matrix,
                     const MPF_Int lead_dim_A,
                     const void *B_matrix,
                     const MPF_Int lead_dim_B,
                     const void *beta,
                     void *C_matrix,
                     const MPF_Int lead_dim_C) NOTHROW;

extern double mpf_dnrm2(const MPF_Int num_rows_x,
                       const double *x_vector,
                       const MPF_Int inc_x) NOTHROW;

extern double mpf_dznrm2(const MPF_Int num_rows_x,
                        const void *x_vector,
                        const MPF_Int inc_x) NOTHROW;

extern void mpf_daxpy(const MPF_Int num_rows_A,
                     const double alpha,
                     const double *x_vector,
                     const MPF_Int inc_X,
                     double *y_vector,
                     const MPF_Int inc_y) NOTHROW;

extern double mpf_ddot(const MPF_Int num_rows_x,
                      const double *x_vector,
                      const MPF_Int inc_x,
                      const double *y_vector,
                      const MPF_Int inc_y) NOTHROW;

extern void mpf_dscal(const MPF_Int num_rows_x,
                     const double alpha,
                     double *x_vector,
                     const MPF_Int inc_x) NOTHROW;

extern double LAPACKE_dlange(MPF_LapackLayout layout_A,
                             char norm,
                             MPF_LapackInt num_rows_A,
                             MPF_LapackInt num_cols_A,
                             const double *A_matrix,
                             MPF_LapackInt lead_dim_A);

extern double LAPACKE_zlange(MPF_LapackLayout layout_A,
                             char norm,
                             MPF_LapackInt num_rows_A,
                             MPF_LapackInt num_cols_A,
                             const lapack_complex_double *A_matrix,
                             MPF_LapackInt lead_dim_A);

extern MPF_LapackInt mpf_zlacpy(const MPF_LapackLayout layout_A,
                             const MPF_LapackUplo uplo_A,
                             const MPF_LapackInt num_rows_A,
                             const MPF_LapackInt num_cols_A,
                             const lapack_complex_double *A_matrix,
                             const MPF_LapackInt lead_dim_A,
                             lapack_complex_double *B_matrix,
                             const MPF_LapackInt lead_dim_B);

extern MPF_LapackInt mpf_clacpy(const MPF_LapackLayout layout_A,
                              const MPF_LapackUplo uplo_A,
                              const MPF_LapackInt num_rows_A,
                              const MPF_LapackInt num_cols_A,
                              const lapack_complex_float *A_matrix,
                              const MPF_LapackInt lead_dim_A,
                              lapack_complex_float *B_matrix,
                              const MPF_LapackInt lead_dim_B);

extern void mpf_dimatcopy(const char ordering,
                         const char transpose_A,
                         size_t num_rows_A,
                         size_t num_cols_A,
                         const double alpha,
                         double *AB_matrix,
                         size_t lead_dim_A,
                         size_t lead_dim_B);

extern void mpf_zimatcopy(const char ordering,
                         const char transpose_A,
                         size_t num_rows_A,
                         size_t num_cols_A,
                         const MPF_Complex16 alpha,
                         MPF_LapackComplexDouble *AB_matrix,
                         size_t lead_dim_A,
                         size_t lead_dim_B);

extern void mpf_domatcopy(char ordering,
                         char trans,
                         size_t rows,
                         size_t cols,
                         const double alpha,
                         const double *A_matrix,
                         size_t lead_dim_A,
                         double *B_matrix,
                         size_t lead_dim_B);

extern void mpf_zomatcopy(char ordering,
                         char trans,
                         size_t rows,
                         size_t cols,
                         const MPF_Complex16 alpha,
                         const MPF_Complex16 *A_matrix,
                         size_t lead_dim_A,
                         MPF_Complex16 *B_matrix,
                         size_t lead_dim_B);

extern sparse_status_t mpf_sparse_d_export_csr(const MPF_MatrixSparseInternal source,
                                              MPF_SparseIndexBase *indexing,
                                              MPF_Int *num_rows,
                                              MPF_Int *num_cols,
                                              MPF_Int **rows_start,
                                              MPF_Int **rows_end,
                                              MPF_Int **col_ind,
                                              double **values);

#endif /* MPF_F_BLAS_h -- end */
