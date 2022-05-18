#ifndef MPF_MMIO_H
#define MPF_MMIO_H

#define MM_TYPECODE_SIZE 4

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64
//#define ALIGN               MPF__ALIGN

#include "mpf.h"
#include "mpf_blas_mkl_internal.h"
#include "string.h"
#include "mkl.h"
#include "ctype.h"

typedef char          MM_typecode[MM_TYPECODE_SIZE];

char *mm_typecode_to_str(MM_typecode matcode);

// old
//int mm_read_banner(FILE *handle_file,
//                   MM_typecode *matrix_code);

int mm_read_banner(FILE *handle_file, MM_typecode *matrix_code);

int mm_read_mtx_crd_size(FILE  *handle_file,
                         MPF_Int *num_rows,
                         MPF_Int *num_cols,
                         MPF_Int *num_nonzero_entries);

int mm_read_mtx_array_size(FILE *handler_file,
                           MPF_Int *num_rows_A,
                           MPF_Int *num_cols_A);

int mm_write_banner(FILE        *handle_file,
                    MM_typecode matcode);


int mm_write_mtx_crd_size(FILE *f,
                          MPF_Int num_rows,
                          MPF_Int num_cols,
                          MPF_Int num_nonzero_entries);

int mm_write_mtx_crd (char fname[],
                      MPF_Int M,
                      MPF_Int N,
                      MPF_Int nz,
                      MPF_Int I[],
                      MPF_Int J[],
                      void *val,
                      MM_typecode matcode);

int mm_read_mtx_array_size(FILE  *handler_file,
                           MPF_Int *num_rows_A,
                           MPF_Int *num_cols_A);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode);   // too complex for a macro

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	((*typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)
#define mm_set_sparse(typecode)	mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)	((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)	((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
                                    (*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)
#define mm_typecode_init(typecode) mm_initialize_typecode(typecode)

/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17

/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

ojbect  sparse/   data        storage 
        dense     type        scheme

   string position:	 [0]        [1]			[2]         [3]

   Matrix typecode:  M(atrix)  C(oord)		R(eal)   	G(eneral)
						        A(array)	C(omplex)   H(ermitian)
											P(attern)   S(ymmetric)
								    		I(nteger)	K(kew)

 ***********************************************************************/

#define MM_MTX_STR		"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate" 
#define MM_SPARSE_STR	"coordinate"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR		"real"
#define MM_INT_STR		"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR		"symmetric"
#define MM_HERM_STR		"hermitian"
#define MM_SKEW_STR		"skew-symmetric"
#define MM_PATTERN_STR  "pattern"

/*  high level routines */

int mm_read_mtx_crd_ext (char        *filename,
                         MPF_Int       num_rows_A,
                         MPF_Int       num_cols_A,
                         MPF_Int       num_nonzero_entries_A,
                         MPF_Int       *rows_A,
                         MPF_Int       *cols_A,
                         void        *values_A,
                         MM_typecode *matrix_code);

int mm_read_mtx_crd_data(FILE *f, MPF_Int M, MPF_Int N, MPF_Int nz, MPF_Int I[], MPF_Int J[],
		double val[], MM_typecode matcode);

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *img,
			MM_typecode matcode);

int mm_read_unsymmetric_sparse(const char *fname, MPF_Int *M_, MPF_Int *N_, MPF_Int *nz_,
                double **val_, MPF_Int **I_, MPF_Int **J_);

int mm_read_mtx_crd(char        *fname,
                    MPF_Int       *M,
                    MPF_Int       *N,
                    MPF_Int       *nz,
                    MPF_Int       **I,
                    MPF_Int       **J,
                    double      **val,
                    MM_typecode *matcode);


int mm_read_mtx_crd_data_ext(FILE *handler_file,
                             MPF_Int num_rows_A , MPF_Int num_cols_A, MPF_Int num_nonzero_entries,
                             MPF_Int rows_A[]   , MPF_Int cols_A[]  , void *values_A            ,
                             MM_typecode matrix_code);

char *mm_strdup(const char *s);

int mm_read_mtx_crd_data_ext(FILE *handle_file,
                             MPF_Int num_rows_A,
                             MPF_Int num_cols_A,
                             MPF_Int num_nonzero_entries,
                             MPF_Int rows_A[],
                             MPF_Int cols_A[],
                             void *values_A,
                             MM_typecode matrix_code);


//void read_matrix_coo (char              *filename_in,
//                      MPF_Int            num_rows_A,
//                      MPF_Int            num_cols_A,
//                      MPF_Int            num_nz_A,
//                      MatrixCoo         *matrix_A,
//                      MM_typecode  *io_matrix_code);
//
//void write_matrix_coo (char             *filename_out,
//                       MPF_Int           num_rows_A,
//                       MPF_Int           num_cols_A,
//                       MPF_Int           num_nonzero_entries_A,
//                       MatrixCoo        *matrix_A,
//                       MM_typecode io_matrix_code);
//
//void write_complex_matrix_coo (char             *filename_out,
//                               MPF_Int           num_rows_A,
//                               MPF_Int           num_cols_A,
//                               MPF_Int           num_nonzero_entries_A,
//                               MatrixCooComplex *matrix_A,
//                               MM_typecode io_matrix_code);
//
//void convert_real_to_complex_coo (char *filename_in, char *filename_out);
//
//void convert_real_to_complex_dense (char *filename_in, char *filename_out);
//
//void complex_to_real_dense (double *matrix_real_A, MPF_ComplexDouble *matrix_complex_A, MPF_Int num_rows_A, MPF_Int num_cols_A);

//int read_dense_matrix (char               *filename_in,
//                       blas_layout_t      layout,
//                       double             *matrix_A,
//                       MPF_Int              num_rows,
//                       MPF_Int              num_cols);
//
//int write_dense_matrix(char   *filename,
//                       double *matrix_A,
//                       MPF_Int num_rows,
//                       MPF_Int num_cols);
//
//int read_dense_matrix_complex (char             *filename_in,
//                               blas_layout_t    layout,
//                               MPF_ComplexDouble *matrix_A,
//                               MPF_Int           num_rows,
//                               MPF_Int           num_cols);
//
//int write_dense_matrix_complex (char             *filename,
//                                MPF_ComplexDouble *matrix_A,
//                                MPF_Int           num_rows_A,
//                                MPF_Int           num_cols_A);

//void read_matrix_complex_coo (char             *filename_in,
//                              MPF_Int           num_rows_A,
//                              MPF_Int           num_cols_A,
//                              MPF_Int           num_nonzero_entries_A,
//                              MatrixCooComplex *matrix_A,
//                              MM_typecode *io_matrix_code);

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/



#endif
