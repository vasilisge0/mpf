#ifndef MPF_BLAS_MKL_INTERNAL_H
#define MPF_BLAS_MKL_INTERNAL_H

#include "mkl.h"
#include "mkl_types.h"
#include "mkl_spblas.h"
#include "mkl_cblas.h"

typedef CBLAS_LAYOUT    MPF_Layout;
typedef CBLAS_TRANSPOSE MPF_Transpose;
typedef CBLAS_SIDE      MPF_Side;
typedef CBLAS_UPLO      MPF_Uplo;
typedef CBLAS_DIAG      MPF_Diag;
typedef sparse_layout_t MPF_LayoutSparse;

typedef void                (*MPF_FuncPtr)();
typedef void                (*MPF_FunctionPtr)();
typedef struct matrix_descr MPF_MatrixSparseDescriptor;
typedef struct matrix_descr MPF_SparseDescr;

typedef struct matrix_descr MPF_SparseDescr;

typedef CBLAS_TRANSPOSE     MPF_Transpose;
typedef sparse_operation_t  MPF_SparseTranspose;
typedef CBLAS_SIDE          MPF_Side;
typedef CBLAS_UPLO          MPF_Uplo;
typedef CBLAS_DIAG          MPF_Diag;
typedef void                MPF_Meta;
typedef void                *MPF_MetaHandle;
typedef void                MPF_Matrix;
typedef MKL_INT             MPF_Int;

/* === LAPACK types === */

typedef char       MPF_LapackUplo;
typedef lapack_int MPF_LapackInt;
typedef int        MPF_LapackLayout;

/* === BLAS types === */

typedef CBLAS_LAYOUT        MPF_BlasLayout;
typedef CBLAS_TRANSPOSE     MPF_BlasTranspose;
typedef sparse_operation_t  MPF_BlasSparseTranspose;
typedef CBLAS_SIDE          MPF_BlasSide;
typedef CBLAS_UPLO          MPF_BlasUplo;
typedef CBLAS_DIAG          MPF_BlasDiag;

typedef void                MPF_Matrix;
typedef MPF_Matrix*           MPF_MatrixHandle;

typedef sparse_layout_t     MPF_SparseLayout;
typedef sparse_matrix_t     MPF_MatrixSparseInternal;
typedef sparse_index_base_t MPF_SparseIndexBase;
typedef void*               MPF_Handle;
//typedef double*             MPF_MatrixDense;
typedef sparse_matrix_t     MPF_MatrixSparseHandle;
typedef sparse_matrix_t     MPF_CsrHandle;
typedef sparse_matrix_t     MPF_SparseHandle;

typedef MKL_Complex16       MPF_ComplexDouble;
typedef MKL_Complex8        MPF_Complex;
typedef int                 CusparseInt;

#define MPF_SPARSE_COL_MAJOR SPARSE_LAYOUT_COLUMN_MAJOR
#define MPF_SPARSE_ROW_MAJOR SPARSE_LAYOUT_ROW_MAJOR
#define MPF_COL_MAJOR CblasColMajor
#define MPF_BLAS_COL_MAJOR CblasColMajor
#define MPF_BLAS_COL_MAJOR CblasColMajor
#define MPF_ROW_MAJOR CblasRowMajor
#define MPF_COL_MAJOR_SPARSE SPARSE_LAYOUT_COLUMN_MAJOR
#define MPF_SPARSE_COL_MAJOR SPARSE_LAYOUT_COLUMN_MAJOR

#define MPF_COL_MAJOR CblasColMajor
#define MPF_ROW_MAJOR CblasRowMajor

/* === SPARSE === */

#define MPF_SPARSE_TRANSPOSE SPARSE_OPERATION_TRANSPOSE
#define MPF_SPARSE_NON_TRANSPOSE SPARSE_OPERATION_NON_TRANSPOSE

/* === DENSE === */

#define MPF_TRANSPOSE      CblasTrans
#define MPF_NON_TRANSPOSE  CblasNoTrans

#define MPF_Trans  CblasTrans
#define MPF_NoTrans CblasNoTrans
#define MPF_ConjTrans  CblasConjNoTrans

#define MPF_BLAS_CONJ_TRANS CblasConjTrans
#define MPF_BLAS_TRANS CblasTrans
#define MPF_BLAS_NO_TRANS CblasNoTrans

#define MPF_ALIGN  16
#define INDEXING SPARSE_INDEX_BASE_ZERO

//typedef SPARSE_OPERATION_TRANSPOSE MPF_SPARSE_TRANPOSE;

#define mpf_sparse_d_mv mkl_sparse_d_mv
#define mpf_sparse_d_mm mkl_sparse_d_mm
#define mpf_sparse_z_mm mkl_sparse_z_mm
#define mpf_sparse_z_mm mkl_sparse_z_mm
#define mpf_dtrsm cblas_dtrsm
#define mpf_dgemv cblas_dgemv
#define mpf_dgeqrf LAPACKE_dgeqrf
#define mpf_dlacpy LAPACKE_dlacpy
#define mpf_dorgqr LAPACKE_dorgqr
#define mpf_dgemm cblas_dgemm
#define mpf_dnrm2 cblas_dnrm2
#define mpf_daxpy cblas_daxpy
#define mpf_ddot cblas_ddot
#define mpf_dscal cblas_dscal
#define mpf_dlange LAPACKE_dlange
#define mpf_zlange LAPACKE_zlange
#define mpf_domatcopy mkl_domatcopy
#define mpf_dgesv LAPACKE_dgesv
#define mpf_dimatcopy mkl_dimatcopy
#define mpf_zimatcopy mkl_zimatcopy
#define mpf_sparse_d_export_csr mkl_sparse_d_export_csr
#define mpf_sparse_z_export_csr mkl_sparse_z_export_csr

#define mpf_sparse_s_mv mkl_sparse_s_mv
#define mpf_sparse_s_mm mkl_sparse_s_mm
#define mpf_sparse_c_mm mkl_sparse_c_mm
#define mpf_sparse_c_mm mkl_sparse_c_mm
#define mpf_strsm cblas_strsm
#define mpf_sgemv cblas_sgemv
#define mpf_sgeqrf LAPACKE_sgeqrf
#define mpf_slacpy LAPACKE_slacpy
#define mpf_sorgqr LAPACKE_sorgqr
#define mpf_sgemm cblas_sgemm
#define mpf_snrm2 cblas_snrm2
#define mpf_saxpy cblas_saxpy
#define mpf_sdot cblas_sdot
#define mpf_sscal cblas_sscal
#define mpf_slange LAPACKE_slange
#define mpf_somatcopy mkl_somatcopy
#define mpf_sgesv LAPACKE_sgesv
#define mpf_simatcopy mkl_simatcopy
#define mpf_cimatcopy mkl_cimatcopy
#define mpf_sparse_s_export_csr mkl_sparse_s_export_csr

/* sorting algorithms */
#define mpf_dlasrt LAPACKE_dlasrt

/* complex functions */

#define mpf_sparse_z_mv mkl_sparse_z_mv
#define mpf_sparse_d_mm mkl_sparse_d_mm
#define mpf_zgemm cblas_zgemm
#define mpf_zaxpy cblas_zaxpy
#define mpf_zscal cblas_zscal
#define mpf_ztrsm cblas_ztrsm
#define mpf_zgemv cblas_zgemv
#define mpf_zlacpy LAPACKE_zlacpy
#define mpf_zomatcopy mkl_zomatcopy
#define mpf_dznrm2 cblas_dznrm2
#define mpf_zgesv LAPACKE_zgesv

#define mpf_sparse_c_mv mkl_sparse_c_mv
#define mpf_sparse_s_mm mkl_sparse_s_mm
#define mpf_cgemm cblas_cgemm
#define mpf_caxpy cblas_caxpy
#define mpf_cscal cblas_cscal
#define mpf_ctrsm cblas_ctrsm
#define mpf_cgemv cblas_cgemv
#define mpf_clacpy LAPACKE_clacpy
#define mpf_comatcopy mkl_comatcopy
#define mpf_sznrm2 cblas_sznrm2
#define mpf_cgesv LAPACKE_cgesv

#define mpf_vectorized_z_abs vzAbs
#define mpf_vectorized_z_sqrt vzSqrt

#define mpf_vectorized_c_abs vcAbs
#define mpf_vectorized_c_sqrt vcSqrt

#define mpf_malloc(size)             mkl_malloc(size, MPF_ALIGN)
#define mpf_calloc(num_items, size)  mkl_calloc(num_items, size, MPF_ALIGN)
#define mpf_free(source)             mkl_free(source)

typedef MKL_Complex16 MPF_Complex16;
typedef lapack_complex_double MPF_LapackComplexDouble;

#endif
