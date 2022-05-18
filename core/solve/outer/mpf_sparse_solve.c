// #include "mpf.h"
//
// void mpf_d_sparse_solve
// (
//   MPF_Solver *context,
//   MPF_Sparse *A,
//   MPF_Sparse *B
// )
// {
//   MPF_Sparse *X;
//
//   mkl_sparse_d_create_csr(A->handle, INDEXING, A->m, A->m, A->mem.csr.rs,
//     A->mem.csr.re, A->mem.csr.cols, A->mem.csr.data);
//
//   /* requires CSC format */
//   //for (i = 0; i < m; ++i)
//   //for (i = 0; i < 1; ++i)
//   i = 0;
//   {
//     B->mem.csr.rs = mp_malloc(sizeof(MPF_Int)*1);
//     B->mem.csr.re = mp_malloc(sizeof(MPF_Int)*1);
//     B->mem.csr.cols = mp_malloc(sizeof(MPF_Int)*1);
//     B->mem.csr.data = mp_malloc(sizeof(double)*1);
//     B->mem.csr.rs[0] = 0;
//     B->mem.csr.re[0] = 1;
//     B->mem.csr.cols[0] = 0;
//     ((double*)B.data)[0] = 1.0;
//     B.descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//     mkl_sparse_d_create_csr(&B->handle, INDEXING, 1, B->m, B->mem.csr.rs,
//       B->mem.csr.re, B->mem.csr.cols, B->mem.csr.data);
//
//     X->mem.csr.rs = mp_malloc(sizeof(MPF_Int)*1);
//     X->mem.csr.re = mp_malloc(sizeof(MPF_Int)*1);
//     X->mem.csr.cols = mp_malloc(sizeof(MPF_Int)*1);
//     X->mem.csr.data = mp_malloc(sizeof(double)*1);
//     X->mem.csr.rs[0] = 0;
//     X->mem.csr.re[0] = 0;
//     X->mem.csr.cols[0] = 0;
//     ((double*)X->mem.csr.data)[0] = 0.0;
//     X->descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//     mkl_sparse_d_create_csr(&X->handle, INDEXING, 1, X->m, X->mem.csr.rs,
//       X->mem.csr.re, X->mem.csr.cols, X->mem.csr.data);
//   }
//
//   A->descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//   fsolve = &mp_dsy_sparse_cg;
//
//   for (MPF_Int i = 0; i < A->m; ++i)
//   {
//     //mp_dsy_sparse_cg
//     fsolve(meta, m, A->descr, A->handle, B->data, X->data, mem, NULL); /* csc handle */
//     //fsolve(meta, m, A->descr, A_handle, &B, &X, mem, NULL);  /* csr handle */
//   }
//
//   /* frees memory */
//
//   mkl_sparse_destroy(A->handle);
//   mkl_sparse_destroy(B->handle);
//   mkl_sparse_destroy(X->handle);
//
//   mp_free(B->mem.csr.rs);
//   mp_free(B->mem.csr.re);
//   mp_free(B->mem.csr.cols);
//   mp_free(B->mem.csr.data);
//   //mp_free(X.rs);
//   //mp_free(X.re);
//   //mp_free(X.cols);
//   //mp_free(X.data);
// }
