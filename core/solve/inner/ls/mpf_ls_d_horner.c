//#include "mpf.h"
//
///*==============================================================================
//Uses the least squares estimation method to estimate the diagonal of
// A^{-1}. This is accomplised by:
//
//(1) solving the least squares problem
//
//    Vx = e where
//
//V = [d(A) d(A^2} d(A^3) ... d(A^{k})]
//e = [1 1 1 ... 1]^T = ones(n,1);
//
//(2) reconstructing approximation as
//
//d ~= [e d(A) d(A^2) ... d(A^{k-1}]*x
//
//d(A^i) are extracted in exact fashion by applying k-distance multilevel
// probing.
//
//(1) colorings are the inputs, A^{i}*probing_vectors are computed via Horner's
//method:
//
//Illustration of the way loops are arragned:
//-------------------------------------------
//num_batches
//|
//|
//|   num_blocks
//|   |
//|   |
//|   |   num_levels  (computes (A^k)*V via Horner's method)
//|   |   |
//|   |   |
// --  --  --
//
//(2) solves [d(A) d(A^2} d(A^3) ... d(A^{k})]*x = e using qr_mrhs_dge
//
//and reconstructs by computing d ~= [I d(A) d(A^2} ... d(A^{k-1})]*x.
//
//(3) Used only fr diag(A)
//
//applies horner iterations to current batch of B
//
//(4) HORNER's method for evaluating matrix polynomial.
//
//A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1}
//    = a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//      -------------------------------------------------------------
//            restructuring that is used in Horner's method
//
//(5) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
//A^p*V(:, I) = [a0I + A(a1*I + A(a2*I + A(a3*I + ...
//              + A(a{p-1}I + ap*A))...)]*V(:, I)
//            = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... +A(a{p-1}V(:,I)
//              + ap*A*V(:,I))...)
//
//INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//----------
//INITIAL:  i = 0:p-2, with V(:,I)^(0) <- ap-2*V(:,I)+ap-1*A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//In following code ai = 1.0 so:
//INITIAL: starting from i = 0 -> p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//
//==============================================================================*/
//
//void mpf_ls_dsy_horner
//(
//  MPF_Solver *solver,
//  MPF_Sparse *A,
//  MPF_Dense *B,
//  MPF_Dense *X
//)
//{
////  MPF_Dense *diag_fA = (MPF_Dense *)fA_out;
////  MPF_Int n_max_B = solver->n_max_B;
////  MPF_Int n_diags = (MPInt) (pow(2.0, (double)solver->n_levels)+0.5);
////  MPF_Int n_batches = (MPF_Int)((double)n_max_B / (double)solver->batch+0.5);
////
////  /* allocates memory */
////  solver->inner_mem = solver->inner_alloc(solver->inner_bytes);
////
////  /* unpacks array variables */
////  solver->V = (double*)solver->inner_mem;
////  double *temp_X = &((double*)solver->V)[solver->X.m*solver->batch*n_diags];
////  double *temp_V = &((double*)temp_X)[solver->X.m*solver->batch];
////  double *e_vector = &((double*)temp_V)[solver->X.m*solver->batch*n_diags];
////  double *temp_matrix = &((double*)e_vector)[solver->X.m];
////
////  /* used for swapping solver->B and solver->X */
////  double *swap = NULL;
////
////  /*---------------------------------------------------*/
////  /* Extracts diagonals in matrix V = [d0 d1 ... dp-1] */
////  /*---------------------------------------------------*/
////
////  /* sets first column of V to [1 1 ... 1]^T */
////  mp_matrix_d_set(MPF_COL_MAJOR, solver->B.m, 1, solver->V, solver->B.m, 1.0);
////
////  /* forall batches */
////  for (MPF_Int j = 0; j < n_batches; ++j)  
////  {
////    MPF_Int current_rhs = solver->batch*j;
////    MPF_Int current_blk =
////        (1-j/(n_batches-1))*solver->batch + (j/(n_batches-1))*(n_max_B-current_rhs);
////    mp_d_generate_B(solver->blk_max_fA, solver->colorings-array,
////      current_rhs, solver->B.m, solver->batch, solver->B.data);
////
////    /* forall diagonals */
////    for (MPF_Int i = 1; i < n_diags; ++i) 
////    {
////      /* updates rhs X <- A*B */
////      mp_sparse_d_mm(
////        SPARSE_OPERATION_NON_TRANSPOSE,
////        1.0,
////        A->handle,
////        A->descr,
////        MPF_SPARSE_COL_MAJOR,
////        solver->B.data,  /* ok computes solver->B = A*solver->X */
////        current_blk,
////        solver->B.m,
////        0.0,
////        solver->X.data,
////        solver->X.m);
////
////      /* extracts nonzeros entrie of X into temp_X */
////      memcpy(temp_X, solver->X, (sizeof *temp_X)*solver->B.m*current_blk);
////      mp_d_select_X_dynamic(solver->blk_max_fA, solver->colorings_array,
////        solver->B.m, temp_X, current_rhs, current_blk);
////
////      /* sums columns of rhs */
////      for (MPF_Int p = 0; p < current_blk; ++p)
////      {
////        mp_daxpy(
////          solver->X.m,
////          1.0,
////          &temp_X[solver->X.m*p],      /* holds diag on all of S(A^2p) */
////          1,                            /* incx */
////          &((double*)solver->V)[solver->X.m*i],/* V = [d0 d1 ... dp-1] */
////          1);                           /* incy */
////      }
////
////      /* swaps B and X */
////      swap = solver->B.data;
////      solver->B.data = solver->X.data;
////      solver->X.data = swap;
////    }
//  }
//
//  /*-------------------------------------*/
//  /* solves least squares problem VX = W */
//  /*-------------------------------------*/
//
//  //memcpy(temp_V, solver->V, (sizeof *temp_V)*solver->B.m*(n_diags-1));
//  //mp_matrix_d_set(MPF_COL_MAJOR, solver->B.m, 1, e_vector, solver->B.m, 1.0);
//  //mp_matrix_d_set(MPF_COL_MAJOR, diag_FA->m, 1, diag_fA.data, diag_fA->m, 1.0);
//
//  ///* computes qr factorization of thin and tall matrix V */
//  //mp_qr_givens_dge_2(solver->B.m, n_diags-1, 1,
//  //  &((double*)solver->V)[solver->B.m], solver->B.m,
//  //  e_vector, solver->B.m, temp_matrix);
//
//  ///* solves system of equqation */
//  //mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//  //  n_diags-1, 1, 1.0, &((double*)solver->V)[m_B], solver->B.m, e_vector,
//  //  solver->B.m);
//
//  ///* reconstruction */
//  //mp_dgemm(
//  //  CblasColMajor,      /* ordering */
//  //  MPF_BLAS_NO_TRANS,  /* transpose operator V */
//  //  MPF_BLAS_NO_TRANS,  /* transpose operator e_vector */
//  //  solver->B.m,       /* rows of V */
//  //  1,                  /* num_cols of e_vector */
//  //  n_diags-1,          /* */
//  //  1.0,                /* multiplier */
//  //  temp_V,             /* */
//  //  solver->B.m,       /* lead dimension of V */
//  //  e_vector,
//  //  solver->B.m,       /* lead dimension of e_vector */
//  //  0.0,
//  //  solver->diag_fA,   /* diagonal vector result (output) */
//  //  solver->B.m);      /* lead dimension of output */
//}
