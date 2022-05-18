#include "mpf_mmio.h"

char* mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];
    char *mm_strdup(const char *);
    //int error =0;


    /* check for MTX type */
    if (mm_is_matrix(matcode))
        types[0] = MM_MTX_STR;
    //else {
    //    error=1;
    //}

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;

    /* check for symmetry type */

    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);
}

int mm_read_banner(FILE *handler_file, MM_typecode *matrix_code)
{
    /* declares variables */

    char line           [MM_MAX_LINE_LENGTH];
    char banner         [MM_MAX_TOKEN_LENGTH];
    char mtx            [MM_MAX_TOKEN_LENGTH];
    char crd            [MM_MAX_TOKEN_LENGTH];
    char data_type      [MM_MAX_TOKEN_LENGTH];
    char storage_scheme [MM_MAX_TOKEN_LENGTH];

    char *p;

    /* reads data as lines */

    mm_clear_typecode(matrix_code);

    if (fgets(line, MM_MAX_LINE_LENGTH, handler_file) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5) {
        return MM_PREMATURE_EOF;
    }

    for (p = mtx; *p!='\0' ; *p  = tolower(*p), p++);  // convert to lower case
    for (p = crd; *p!='\0' ; *p  = tolower(*p), p++);
    for (p = data_type     ; *p !='\0';         *p = tolower(*p), p++);
    for (p = storage_scheme; *p !='\0';         *p = tolower(*p), p++);

    /* check for banner */

    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;


    /* first field should be "mtx" */

    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;

    mm_set_matrix(matrix_code);

    /* second field describes whether this is a sparse matrix (in coordinate storage) or a dense array */

    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matrix_code);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
        mm_set_dense(matrix_code);
    else
        return MM_UNSUPPORTED_TYPE;

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matrix_code);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matrix_code);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matrix_code);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matrix_code);
    else
        return MM_UNSUPPORTED_TYPE;

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matrix_code);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matrix_code);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matrix_code);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matrix_code);
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

int mm_read_mtx_crd_size(FILE *f, MPF_Int *M, MPF_Int *N, MPF_Int *nz) {

    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;
    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
    {
        return 0;
    }
    else
    {
        do
        {
            num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
            printf("waiting to reach size (mm_read_mtx_crd_size)\n");
            //printf("(%d, %d, %d)\n", *M, *N, *nz);
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
        }
        while (num_items_read != 3);
    }

    return 0;
}

int mm_write_banner
(
  FILE *f,
  MM_typecode matcode
)
{
  char *str = mm_typecode_to_str(matcode);
  int ret_code;

    printf("    >> in mm_write_banner\n");

  printf("output header: \n");
  printf("%s %s\n", MatrixMarketBanner, str);
  ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
  printf("after write banner\n");
//free(str);    // testing
  printf("after free\n");
  if (ret_code !=2 )
      return MM_COULD_NOT_WRITE_FILE;
  else
      return 0;
}

int mm_write_mtx_crd_size(FILE *f, MPF_Int M, MPF_Int N, MPF_Int nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

int mm_read_mtx_array_size
(
  FILE *handler_file,
  MPF_Int *num_rows_A,
  MPF_Int *num_cols_A
)
{
  char line[MM_MAX_LINE_LENGTH];
  int  num_items_read;

  /* sets return null parameter values, in case we exit with errors */

  *num_rows_A = 0;
  *num_cols_A = 0;

  /* now continues scanning until you reach the end-of-comments */

  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, handler_file) == NULL)
    {
        return MM_PREMATURE_EOF;
    }
  } while (line[0] == '%');

  /* line[] is either blank or has num_rows_A, num_cols_A, num_nonzero_entries */

  if (sscanf(line, "%d %d", num_rows_A, num_cols_A) == 2)
  {
    return 0;
  }

  do {
      num_items_read = fscanf(handler_file, "%d %d\n", num_rows_A, num_cols_A);

      if (num_items_read == EOF)
          return MM_PREMATURE_EOF;

  } while (num_items_read != 2);

  return 0;
}

int mm_is_valid(MM_typecode matcode)
{
  if (!mm_is_matrix(matcode)) return 0;
  if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
  if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
  if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
              mm_is_skew(matcode))) return 0;
  return 1;
}

int mm_read_mtx_crd_ext(char        *filename,
                        MPF_Int       num_rows_A,
                        MPF_Int       num_cols_A,
                        MPF_Int       num_nonzero_entries_A,
                        MPF_Int       *rows_A,
                        MPF_Int       *cols_A,
                        void        *values_A,
                        MM_typecode *matrix_code) {
    int  return_code;
    FILE *handler_file;
    printf("in crd_ext\n");
    /* metadata configuration */

    if (strcmp(filename, "stdin") == 0) {
        handler_file = stdin;
    }
    else if ((handler_file = fopen(filename, "r")) == NULL) {
        return MM_COULD_NOT_READ_FILE;
    }
    printf("after first test\n");

    if ((return_code = mm_read_banner(handler_file, matrix_code)) != 0) {
        return return_code;
    }
    if (!(mm_is_valid (*matrix_code) && mm_is_sparse(*matrix_code) && mm_is_matrix(*matrix_code))) {
        return MM_UNSUPPORTED_TYPE;
    }

    MPF_Int tmp_rows;
    MPF_Int tmp_cols;
    MPF_Int tmp_nnz;

    if ((return_code = mm_read_mtx_crd_size(handler_file, &tmp_rows, &tmp_cols, &tmp_nnz)) != 0) {
        return return_code;
    }

    /* reads data */

    printf("until here (out)\n");
    if (mm_is_complex(*matrix_code)) {
        printf("until here\n");
        printf("complex\n");
        return_code = mm_read_mtx_crd_data_ext(handler_file, num_rows_A, num_cols_A, num_nonzero_entries_A, rows_A, cols_A, values_A,
                                               *matrix_code);
        // checks return code

        if (return_code != 0) {
            return return_code;
        }
    }
    else if (mm_is_real(*matrix_code)) {

        printf("real\n");
        return_code = mm_read_mtx_crd_data_ext(handler_file,
                                               num_rows_A  , num_cols_A, num_nonzero_entries_A,
                                               rows_A      , cols_A    , values_A             ,
                                               *matrix_code);
        // checks return code

        if (return_code != 0) {
            return return_code;
        }
    }
    else if (mm_is_pattern(*matrix_code)) {
        printf(" $$$$$$$$$$$$$$$$$$$$$$$ IN PATTERN $$$$$$$$$$$$$$$$$$$$$$$\n");
        return_code = mm_read_mtx_crd_data_ext(handler_file,
                                               num_rows_A  , num_cols_A, num_nonzero_entries_A,
                                               rows_A      , cols_A    , values_A             ,
                                               *matrix_code);
        if (return_code != 0) {
            return return_code;
        }
    }

    /* closes file */

    if (handler_file != stdin) {
        fclose(handler_file);
    }

    return 0;
}

int mm_read_mtx_crd_data(FILE *f, 
                         MPF_Int M, MPF_Int N, MPF_Int nz, 
                         MPF_Int I[], MPF_Int J[], double val[], 
                         MM_typecode matcode) {


    int i;


    if (mm_is_complex(matcode)) {                                               // complex data
        for (i = 0; i< nz; i++) {
            if (fscanf(f, "%d %d %lg %lg", 
                &I[i], &J[i], &val[2*i], &val[2*i+1]) != 4) {
                    return MM_PREMATURE_EOF;
            }
        }
    }
    else if (mm_is_real(matcode)) {                                             // real data
        for (i = 0; i< nz; i++) {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) { 
                return MM_PREMATURE_EOF;
            }

            // convert to 1-based indexing
            I[i]--;
            J[i]--;
        }
    }
    else if (mm_is_pattern(matcode)) {                                          // binary data

        for (i=0; i<nz; i++) {
            if (fscanf(f, "%d %d", &I[i], &J[i]) != 2) {
                return MM_PREMATURE_EOF;
            }

            // convert to 1-based indexing
            I[i]--;
            J[i]--;

            // assigns value "1" as the default value for entries in the pattern
            val[i] = 1;
        }
    }
    else {                                                                      // non real, complex or binary data type
        return MM_UNSUPPORTED_TYPE;
    }

    // printf("after val[0]: %d\n",val[i]);

    return 0;
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J,
        double *real, double *imag, MM_typecode matcode)
{
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) return MM_PREMATURE_EOF;

    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

int mm_read_unsymmetric_sparse(const char *fname, MPF_Int *M_, MPF_Int *N_, MPF_Int *nz_,
                double **val_, MPF_Int **I_, MPF_Int **J_) {
    FILE *f;
    MM_typecode matcode;
    MPF_Int M, N, nz;
    int i;
    double *val;
    MPF_Int *I, *J;
    MPF_Int ret = 0;

    if ((f = fopen(fname, "r")) == NULL)
            return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }



    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }
    *M_ = M;
    *N_ = N;
    *nz_ = nz;

    /* reseve memory for matrices */

    I = (MPF_Int *) mkl_malloc(nz * sizeof(MPF_Int), 64);
    J = (MPF_Int *) mkl_malloc(nz * sizeof(MPF_Int), 64);
    val = (double *) mkl_malloc(nz * sizeof(double), 64);

    *val_ = val;
    *I_ = I;
    *J_ = J;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        ret = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);
    printf("ret: %d\n", ret);

    return 0;
}

//int mm_read_mtx_crd(char  *fname,
//                    MPF_Int M,
//                    MPF_Int N,
//                    MPF_Int nz,
//                    MPF_Int *I,
//                    MPF_Int *J,
//                    double *val,
//                    MM_typecode *matcode) {
//
//    /* initialization */
//
//    int return_code;
//    FILE *file_handle;
//
//    /* metadata configuration */
//
//    if (strcmp(fname, "stdin") == 0) {                                          // opens file
//        f=stdin;
//    }
//    else if ((f = fopen(fname, "r")) == NULL) {
//        return MM_COULD_NOT_READ_FILE;
//    }
//
//    if ((ret_code = mm_read_banner(f, matcode)) != 0) {                         // reads file header
//        return ret_code;
//    }
//
//    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&                    // verifies matcode corretness
//            mm_is_matrix(*matcode))) {
//
//        return MM_UNSUPPORTED_TYPE;
//    }
//
//    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0) {                  // reads input matrix dimensions
//        return ret_code;
//    }
//
//    if (mm_is_complex(matrix_code)) {                                               // complex data
//
//        MPF_ComplexDouble *values = (MPF_ComplexDouble *) values_A;
//
//        printf("%1.4E, %1.4E\n", values[0].real, values[0].imag);
//        printf("num_rows_A: %d, num_cols_A: %d, num_nonzero_entries: %d\n", num_rows_A, num_cols_A, num_nonzero_entries);
//
//        for (i = 0; i < num_nonzero_entries; i++) {
//            printf("i: %d\n", i);
//            printf("%d\n", rows_A[i]);
//            printf("%d\n", cols_A[i]);
//            if (fscanf(handle_file, "%d %d %lg %lg", &rows_A[i], &cols_A[i], &(values[i].real), &(values[i].imag)) != 4) {
//                printf("test\n");
//                return MM_PREMATURE_EOF;
//            }
//
//            // converts to 0-based indexing
//
//            rows_A[i]--;
//            cols_A[i]--;
//        }
//    }
//    else if (mm_is_real(matrix_code)) {                                             // real data
//
//        double *values = (double *) values_A;
//
//        for (i = 0; i < num_nonzero_entries; i++) {
//            if (fscanf(handle_file, "%d %d %lg\n", &rows_A[i], &cols_A[i], &values[i]) != 3) { 
//                return MM_PREMATURE_EOF;
//            }
//
//            // converts to 0-based indexing
//
//            rows_A[i]--;
//            cols_A[i]--;
//        }
//    }
//    else if (mm_is_pattern(matrix_code)) {                                          // binary data
//
//        double *values = (double *) values_A;
//
//        for (i = 0; i < num_nonzero_entries; i++) {
//
//            if (fscanf(handle_file, "%d %d", &rows_A[i], &cols_A[i]) != 2) {
//                return MM_PREMATURE_EOF;
//            }
//
//            // convert to 0-based indexing
//
//            rows_A[i]--;
//            cols_A[i]--;
//
//            // assigns value "1" as the default value for entries in the pattern
//
//            values[i] = 1;
//        }
//    }
//    else {                                                                      // non real, complex or binary data type
//        return MM_UNSUPPORTED_TYPE;
//    }
//
//
//    if (f != stdin) {
//        fclose(f);
//    }
//
//    return 0;
//}

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/
char *mm_strdup(const char *s)
{
	int len = strlen(s);
	char *s2 = (char *) malloc((len+1)*sizeof(char));
	return strcpy(s2, s);
}

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data_ext(FILE *handle_file, MPF_Int num_rows_A, MPF_Int num_cols_A, MPF_Int num_nonzero_entries,
                             MPF_Int rows_A[], MPF_Int cols_A[], void *values_A, MM_typecode matrix_code)
{

    int i;

    if (mm_is_complex(matrix_code)) {                                               // complex data

        MPF_ComplexDouble *values = (MPF_ComplexDouble *) values_A;

        for (i = 0; i < num_nonzero_entries; i++) {

            if (fscanf(handle_file, "%d %d %lg %lg",
                &rows_A[i], &cols_A[i], &values[i].real, &values[i].imag) != 4) {
                //&rows_A[i], &cols_A[i], &values.real[2*i], &values_A[2*i+1]) != 4) {
                    return MM_PREMATURE_EOF;
            }

            // converts to 0-based indexing

            rows_A[i]--;
            cols_A[i]--;

        }
    }
    else if (mm_is_real(matrix_code)) {                                             // real data
        double *values = (double *) values_A;

        for (i = 0; i < num_nonzero_entries; i++) {
            if (fscanf(handle_file, "%d %d %lg\n", &rows_A[i], &cols_A[i], &values[i]) != 3) { 
                return MM_PREMATURE_EOF;
            }

            // converts to 0-based indexing

            rows_A[i]--;
            cols_A[i]--;
        }
    }
    else if (mm_is_pattern(matrix_code)) {                                          // binary data
        double *values = (double *) values_A;

        for (i = 0; i < num_nonzero_entries; i++) {

            if (fscanf(handle_file, "%d %d", &rows_A[i], &cols_A[i]) != 2) {
                return MM_PREMATURE_EOF;
            }

            // convert to 0-based indexing

            rows_A[i]--;
            cols_A[i]--;

            // assigns value "1" as the default value for entries in the pattern

            values[i] = 1;
        }
    }
    else {                                                                      // non real, complex or binary data type
        return MM_UNSUPPORTED_TYPE;
    }

    return 0;
}

int mm_write_mtx_crd
(
  char fname[],
  MPF_Int M,
  MPF_Int N,
  MPF_Int nz,
  MPF_Int I[],
  MPF_Int J[],
  void *val,
  MM_typecode matcode
)
{
    /* initialization */
    FILE *f;
    int i;
    printf("calling function write_mtx_crd\n");

    /* writes matrix */

    // opens file
    if (strcmp (fname, "stdout") == 0)
        f = stdout;
    else if ((f = fopen (fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;

    printf("here\n");
    // prints banner followed by typecode
    //fprintf(f, "%s ", MatrixMarketBanner);
    mm_write_banner(f, matcode);
    printf("and here\n");

    // prints matrix sizes and nonzeros
    printf("M: %d, N: %d, nz: %d\n", M, N, nz);
    fprintf(f, "%d %d %d\n", M, N, nz);

    printf("IN COO WRITE\n");
    printf("matcode: %s\n", matcode);
    // print values
    if (mm_is_pattern(matcode))
    {
        printf("PATTERN\n");
        for (i = 0; i < nz; i++)
        {
            fprintf (f, "%d %d\n", I[i]+1, J[i]+1);
        }
    }
    else if (mm_is_real (matcode)) {
        printf("Writting double values\n");
        double *values = (double *) val;

        for (i = 0; i < nz; i++)
        {
            //fprintf (f, "%d %d %20.16g\n", I[i]+1, J[i]+1, values[i]);
            //printf("%lf\n", values[i]);
            //printf("i/nz: %d/%d\n", i, nz);
            fprintf (f, "%d %d %1.16E\n", I[i]+1, J[i]+1, values[i]);
        }
    }
    else if (mm_is_complex(matcode)) { 

        MPF_ComplexDouble *values = (MPF_ComplexDouble *) val;

        for (i = 0; i < nz; i++) {
            //fprintf (f, "%d %d %20.16g %20.16g\n", 
            fprintf (f, "%d %d %lf %lf\n", I[i]+1, J[i]+1, values[i].real, values[i].imag);
        }
    }
    else {
        if (f != stdout)
            fclose (f);

        return MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout)
        fclose (f);

    return 0;
}
