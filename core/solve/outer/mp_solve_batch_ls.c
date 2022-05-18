//#include "mpf.h"
//
///* -------------------- least squares solvers (outer) ----------------------- */
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
//void mpf_batch_d_ls_horner
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA,
//) //@USED
//{
//
//  MPF_Int j = 0; /* for blocks */
//  MPF_Int i = 0; /* for diags */
//  MPF_Int p = 0; /* for reduction */
//
//  MPF_Int n_max_B = context->n_max_B;
//
//  MPF_Int n_diags = (MPInt) (pow(2.0, (double)context->n_levels)+0.5);
//
//  /* initializes n_blocks */
//  MPF_Int n_batches = (MPF_Int)((double)n_max_B / (double)context->batch+0.5);
//
//  if (context->data_type == MPF_REAL)
//  {
//    double *swap = NULL;        /* swaps context->B and context->X */
//    double *temp_X = NULL;      /* holds sparsified version of X */
//    double *temp_V = NULL;      /* holds V after QR factorization */
//    double *e_vector = NULL;    /* rhs and solution of LS problem */
//    double *temp_matrix = NULL; /* internal memory used by QR function */
//
//    /* unpacks array variables */
//    context->B.data = context->mem_outer;
//    context->X.data = &((double*)context->B)[context->B.m*context->batch];
//    context->V = &((double*)context->X)[context->X.m*context->batch];
//    temp_X = &((double*)context->V)[context->X.m*context->batch*n_diags];
//    temp_V = &((double*)temp_X)[context->X.m*context->batch];
//    e_vector = &((double*)temp_V)[context->X.m*context->batch*n_diags];
//    temp_matrix = &((double*)e_vector)[context->X.m];
//
//    /*---------------------------------------------------*/
//    /* Extracts diagonals in matrix V = [d0 d1 ... dp-1] */
//    /*---------------------------------------------------*/
//
//    /* sets first column of V to [1 1 ... 1]^T */
//    mp_matrix_d_set(MPF_COL_MAJOR, context->B.m, 1, context->V, context->B.m, 1.0);
//
//    for (MPF_Int j = 0; j < n_batches; ++j)  /* forall blocks*/
//    {
//      MPF_Int current_rhs = context->batch*j;
//      MPF_Int current_blk =
//          (1-j/(n_batches-1))*context->batch + (j/(n_batches-1))*(n_max_B-current_rhs);
//      mp_d_generate_B(context->blk_max_fA, context->colorings-array,
//        current_rhs, context->B.m, context->batch, context->B.data);
//
//      for (MPF_Int i = 1; i < n_diags; ++i) /* forall diagonals */
//      {
//        /* updates rhs X <- A*B */
//        mp_sparse_d_mm(
//          SPARSE_OPERATION_NON_TRANSPOSE,
//          1.0,
//          A->handle,
//          A->descr,
//          MPF_SPARSE_COL_MAJOR,
//          context->B.data,  /* ok computes context->B = A*context->X */
//          current_blk,
//          context->B.m,
//          0.0,
//          context->X.data,
//          context->X.m);
//
//        /* extracts nonzeros entrie of X into temp_X */
//        memcpy(temp_X, context->X, (sizeof *temp_X)*context->B.m*current_blk);
//        mp_d_select_X_dynamic(context->blk_max_fA, context->colorings_array,
//          context->B.m, temp_X, current_rhs, current_blk);
//
//        /* sums columns of rhs */
//        for (MPF_Int p = 0; p < current_blk; ++p)
//        {
//          mp_daxpy(
//            context->X.m,
//            1.0,
//            &temp_X[context->X.m*p],               /* holds diag on all of S(A^2p) */
//            1,                            /* incx */
//            &((double*)context->V)[context->X.m*i],/* V = [d0 d1 ... dp-1] */
//            1);                           /* incy */
//        }
//
//        /* swaps B and X */
//        swap = context->B.data;
//        context->B.data = context->X.data;
//        context->X.data = swap;
//      }
//    }
//
//    /*-------------------------------------*/
//    /* solves least squares problem VX = W */
//    /*-------------------------------------*/
//
//    memcpy(temp_V, context->V, (sizeof *temp_V)*context->B.m*(n_diags-1));
//    mp_matrix_d_set(MPF_COL_MAJOR, context->B.m, 1, e_vector, context->B.m, 1.0);
//    mp_matrix_d_set(MPF_COL_MAJOR, diag_FA->m, 1, diag_fA.data, diag_fA->m, 1.0);
//
//    /* computes qr factorization of thin and tall matrix V */
//    mp_qr_givens_dge_2(context->B.m, n_diags-1, 1, &((double*)context->V)[context->B.m], context->B.m,
//      e_vector, context->B.m, temp_matrix);
//
//    /* solves system of equqation */
//    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//      n_diags-1, 1, 1.0, &((double*)context->V)[m_B], context->B.m, e_vector,
//      context->B.m);
//
//    /* reconstruction */
//    mp_dgemm(
//      CblasColMajor,      /* ordering */
//      MPF_BLAS_NO_TRANS,  /* transpose operator V */
//      MPF_BLAS_NO_TRANS,  /* transpose operator e_vector */
//      context->B.m,       /* rows of V */
//      1,                  /* num_cols of e_vector */
//      n_diags-1,          /* */
//      1.0,                /* multiplier */
//      temp_V,             /* */
//      context->B.m,       /* lead dimension of V */
//      e_vector,
//      context->B.m,       /* lead dimension of e_vector */
//      0.0,
//      context->diag_fA,   /* diagonal vector result (output) */
//      context->B.m);      /* lead dimension of output */
//  }
//}
//
//
//void mpf_batch_z_ls_horner
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA,
//)
//{
//  MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//  MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//
//  MPF_ComplexDouble *swap = NULL;        /* swaps context->B and context->X */
//  MPF_ComplexDouble *temp_X = NULL;      /* holds sparsified version of X */
//  MPF_ComplexDouble *temp_V = NULL;      /* holds V after QR factorization */
//  MPF_ComplexDouble *e_vector = NULL;    /* rhs and solution of LS problem */
//  MPF_ComplexDouble *temp_matrix = NULL; /* internal memory used by QR function */
//  
//  /* unpacks array variables */
//  context->B.data = context->mem_outer;
//  context->X.data = &((MPF_ComplexDouble*)context->B.data)[context->B.m*context->batch];
//  context->V = &((MPF_ComplexDouble*)context->X.data)[context->B.m*batch];
//  temp_X = &((MPF_ComplexDouble*)context->V)[context->B.m*context->batch*n_diags];
//  temp_V = &((MPF_ComplexDouble*)temp_X)[context->B.m*context->batch];
//  e_vector = &((MPF_ComplexDouble*)temp_V)[context->B.m*context->batch*n_diags];
//  temp_matrix = &((MPF_ComplexDouble*)e_vector)[context->B.m];
//
//  mp_matrix_z_set(MPF_COL_MAJOR, context->B.m, 1, context->V, context->B.m,
//    &ONE_C);
//
//  /*--------------------------------------------------*/
//  /* Extract diagonals in matrix V = [d0 d1 ... dp-1] */
//  /*--------------------------------------------------*/
//
//  for (MPF_Int j = 0; j < n_batches; ++j)  /* forall blocks */
//  {
//    MPF_Int current_rhs = blk*j;
//    MPF_Int current_blk
//      = (1-j/(n_blocks-1))*context->batch + (j/(n_blocks-1))*(n_max_B-current_rhs);
//    mp_z_generate_B(context->blk_max_fA, context->colorings_array,
//      current_rhs, context->B.m, blk, context->B.data);
//
//    for (MPF_Int i = 1; i < n_diags; ++i) /* forall diags */
//    {
//      mp_sparse_z_mm(
//        SPARSE_OPERATION_NON_TRANSPOSE,
//        ONE_C,
//        context->A.handle,
//        context->A.descr,
//        MPF_SPARSE_COL_MAJOR,
//        context->B.data, /* ok computes context->B = A*context->X */
//        current_blk,
//        context->B.m,
//        ZERO_C,
//        context->X.data,
//        context->X.m);
//
//      memcpy(temp_X, context->X.data, (sizeof *temp_X)*context->X.m*current_blk);
//      mp_z_select_X_dynamic(context->blk_max_fA, context->colorings_array,
//        context->X.m, temp_X, current_rhs, current_blk);
//
//      for (MPF_Int p = 0; p < current_blk; ++p) /* forall columns of X */
//      {
//        mp_zaxpy(
//          context->X.m,
//          &ONE_C,
//          &temp_X[context->X.m*p],/* holds diag on all of S(A^2p) */
//          1,             /* incx */
//          &((MPF_ComplexDouble*)context->V)[context->X.m*i],/* V = [d0 d1 ... dp-1] */
//          1);            /* incy */
//      }
//  
//      /* swaps B and X */
//      swap = context->B.data;
//      context->B.data = context->X;
//      context->X.data = swap;
//    }
//  }
//  
//  /*-------------------------------------*/
//  /* solves least squares problem VX = W */
//  /*-------------------------------------*/
//  mp_matrix_z_set(MPF_COL_MAJOR, context->X.m, 1, e_vector, context->X.m, &ONE_C);
//  mp_matrix_z_set(MPF_COL_MAJOR, context->X.m, 1, context->V, context>X.m, &ONE_C);
//  memcpy(temp_V, context->V, (sizeof *temp_V)*context->X.m*(n_diags-1));
//  mp_matrix_z_set(MPF_COL_MAJOR, context->X.m, 1, e_vector, context->X.m, &ONE_C);
//  mp_matrix_z_set(MPF_COL_MAJOR, diag_fA->m, 1, diag_fA, diag_fA.m, &ONE_C);
//
//  /* computes qr factorization */
//  mp_qr_zge_givens_3(&((MPF_ComplexDouble*)context->V)[context->B.m],
//    context->B.m, e_vector, context->B.m, context->B.m, n_diags-1, 1,
//    temp_matrix);  //n_diags-1 originally
//
//  /* solves system of equations */
//  mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//    n_diags-1, 1, &ONE_C, &((MPF_ComplexDouble *)context->V)[context->B.m],
//    context->B.m, e_vector, context->B.m);
//
//  /* reconstruction */
//  mp_zgemm(
//    CblasColMajor,      /* ordering */
//    MPF_BLAS_NO_TRANS,  /* transpose operator V */
//    MPF_BLAS_NO_TRANS,  /* transpose operator e_vector */
//    context->B.m,       /* rows of V */
//    1,                  /* num_cols of e_vector */
//    n_diags-1,
//    &ONE_C,             /* multiplier */
//    temp_V,
//    context->B.m,       /* lead dimension of V */
//    e_vector,
//    context->B.m,       /* lead dimension of e_vector */
//    &ZERO_C,
//    diag_fA.data,       /* diagonal vector result (output) */
//    context->B.m);      /* lead dimension of output */
//}
//
//
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
//(1) colorings are inputs, A^{i}*probing_vectors are computed via Horner's
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
//(4) For this version solver_blk must be equal to context->blk_fA.
//
//applies horner iterations to current batch of B
//===============================================
//
//(1) HORNER's method for evaluating matrix polynomial.
//
//A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1}
//    = a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//      -------------------------------------------------------------
//            restructuring that is used in Horner's method
//
//(2) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
//A^p*V(:, I)
//  = [a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)]*V(:, I)
//  = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... + A(a{p-1}V(:,I)
//    + ap*A*V(:,I))...)
//
//INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//----------
//INITIAL: for i = 0:p-2, with V(:,I)^(0) <- ap-2*V(:,I) + ap-1*A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//In following code ai = 1.0 so:
//INITIAL: starting from i = 0 -> p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//
//==============================================================================*/
//void mp_batch_d_ls_horner_diag_blocks
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA
//)
//{
//  MPF_Int n_diags = pow(2, context->n_levels);
//  MPF_Int blk_max_fA = context->blk_max_fA;
//  MPF_Int current_rhs = 0;
//  MPF_Int current_blk = 0;
//  MPF_Int n_max_B = context->n_max_B;
//
//  /* initializes n_blocks */
//  MPF_Int n_batches = (MPF_Int)((double)n_max_B / (double)context->batch+0.5);
//
//  /*--------------------------------------------------------------*/
//  /* extracts diagonals V = [d(I), d(A), d(A^2), ..., d(A^{k-1})] */
//  /*--------------------------------------------------------------*/
//
//  if (context->data_type == MPF_REAL)
//  {
//    double *swap = NULL;    /* used for swapping context->B and context->X */
//    //double *temp_X = mp_malloc((sizeof *temp_X)*m_B*blk);
//    double *temp_X = NULL;
//    double *temp_V = NULL;
//    double *e_vector = NULL;
//    double *temp_matrix = NULL;
//
//    /* unpacking */
//    context->B = context->memory_outer;
//    context->X = &((double*)context->B)[m_B*blk];
//    context->V = &((double*)context->X)[m_B*blk];
//    temp_X = &((double*)context->V)[m_B*blk*n_diags];
//    temp_V = &((double*)temp_X)[m_B*blk];
//    e_vector = &((double*)temp_V)[m_B*blk*n_diags];
//    temp_matrix = &((double*)e_vector)[m_B];
//
//    mp_zeros_d_set(MPF_COL_MAJOR, m_B, n_diags*blk, context->V, m_B);
//    mp_matrix_d_diag_set(MPF_COL_MAJOR, m_B, blk, context->V, m_B, 1.0);
//
//    for (MPF_Int j = 0; j < n_blocks; ++j)  /* forall blocks */
//    {
//      MPF_Int current_rhs = context->batch*j;
//      MPF_Int current_blk =
//          (1-j/(n_batches-1))*context->batch + (j/(n_batches-1))*(n_max_B-current_rhs);
//
//      mp_d_generate_B(context->blk_max_fA, context->colorings_arrays,
//        current_rhs, context->B.m, context->batch, context->B.data);
//
//      for (MPF_Int i = 1; i < n_diags; ++i) /* forall diags */
//      {
//        /* updates rhs and applies probing vectors elementwise multiplication */
//        mp_sparse_d_mm(
//          SPARSE_OPERATION_NON_TRANSPOSE,
//          1.0,
//          A->handle,
//          A->descr,
//          MPF_SPARSE_COL_MAJOR,
//          context->B.data, /* ok computes context->B = A*context->X */
//          current_blk,
//          m_B,
//          0.0,
//          context->X.data,
//          context->X.m);
//
//        memcpy(temp_X, context->X.data, (sizeof *temp_X)*context->X.m*current_blk);
//        mp_d_blk_select_X_dynamic(blk_max_fA, context->colorings_array,
//          context->X.m, temp_X, context->blk_fA, current_rhs, current_blk);
//
//        mp_daxpy(
//          context->X.m*current_blk,   /* rows */
//          1.0,               /* */
//          temp_X,            /* holds diag on all of S(A^2p) */
//          1,                 /* incx */
//          &((double*)context->V)[context->X.m*current_blk*i],  /* V = [d0 d1 ... dp-1] */
//          1);                /* incy */
//
//        /* swaps B and X */
//        swap = context->B.data;
//        context->B.data = context->X.data;
//        context->X.data = swap;
//      }
//    }
//
//    /*-------------------*/
//    /* solves LS problem */
//    /*-------------------*/
//
//    mp_matrix_d_set(MPF_COL_MAJOR, context->X.m, 1, e_vector, context->X.m, 1.0);
//    memcpy(temp_V, context->V, (sizeof *temp_V)*context->X.m*context->batch*(n_diags-1));
//
//    /* computes QR factorization and solves triangular system  */
//    mp_qr_givens_dge_2(m_B, (n_diags-1)*context->batch, 1,
//      &((double*)context->V)[context->X.m*context->batch], context->X.m,
//      e_vector, context->X.m, temp_matrix);
//
//    /* solves system of equations */
//    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//      context->batch*(n_diags-1), 1, 1.0, &((double *)context->V)[context->X.m*context->batch],
//      context->X.m, e_vector, context->X.m);
//
//    /* reconstruction */
//    for (MPF_Int i = 0; i < context->batch*(n_diags-1); ++i)
//    {
//      mp_daxpy(
//        context->X.m,            /* rows */
//        e_vector[i],    /* */
//        &temp_V[context->X.m*i], /* holds diag on all of S(A^2p) */
//        1,              /* incx */
//        &((double*)diag_fA)[diag_fA->m*(i % context->blk_fA)], /* V = [d0 d1 ... dp-1] */
//        1);             /* incy */
//    }
//  }
//}
//
//void mp_batch_z_ls_horner_diag_blocks
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Dense *diag_fA
//)
//{
//  MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//  MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//
//  MPF_ComplexDouble *swap = NULL; /* swaps context->B and context->X */
//  MPF_ComplexDouble *temp_X = NULL;
//  MPF_ComplexDouble *temp_V = NULL;
//  MPF_ComplexDouble *e_vector = NULL;
//  MPF_ComplexDouble *temp_matrix = NULL;
//
//  /* unpacking */
//  context->B.data = context->memory_outer;
//  context->X.data = &((MPF_ComplexDouble*)context->B.data)[context->B.m*context->batch];
//  context->V = &((MPF_ComplexDouble*)context->X)[context->X.m*context->batch];
//  temp_X = &((MPF_ComplexDouble*)context->V)[context->X.m*context->batch*n_diags];
//  temp_V = &((MPF_ComplexDouble*)temp_X)[context->X>m*context->batch];
//  e_vector = &((MPF_ComplexDouble*)temp_V)[context->X.m*context->batch*n_diags];
//  temp_matrix = &((MPF_ComplexDouble*)e_vector)[context->X.m];
//
//  mp_zeros_z_set(MPF_COL_MAJOR, context->X.m, n_diags*context->batch, context->V,
//    context->X.m);
//  mp_matrix_z_diag_set(MPF_COL_MAJOR, context->X.m, context->batch, context->V,
//    context->X.m, ONE_C);
//
//  /*--------------------------------------------------------------*/
//  /* extracts diagonals V = [d(I), d(A), d(A^2), ..., d(A^{k-1})] */
//  /*--------------------------------------------------------------*/
//
//  for (MPF_Int j = 0; j < n_batches; ++j)  /* forall blocks */
//  {
//    MPF_Int current_rhs = context->batch*j;
//    MPF_Int current_blk =
//        (1-j/(n_batches-1))*context->batch + (j/(n_batches-1))*(n_max_B-current_rhs);
//    mp_z_generate_B(context->blk_max_fA, context->colorings_array,
//      current_rhs, context->B.m, context->batch, context->B.data);
//
//    for (MPF_Int i = 1; i < n_diags; ++i) /* forall diagonals */
//    {
//      mp_sparse_z_mm(
//        SPARSE_OPERATION_NON_TRANSPOSE,
//        ONE_C,
//        context->A_handle,
//        context->A_descr,
//        MPF_SPARSE_COL_MAJOR,
//        context->B,    /* context->B = A*context->X */
//        current_blk,
//        m_B,
//        ZERO_C,
//        context->X,
//        m_B);
//
//      memcpy(temp_X, context->X, (sizeof *temp_X)*m_B*current_blk);
//      mp_z_blk_select_X_dynamic(blk_max_fA, context->memory_colorings, m_B,
//        temp_X, blk_fA, current_rhs, current_blk);
//
//      mp_zaxpy(
//        m_B*blk, /* rows */
//        &ONE_C,  /* */
//        temp_X,  /* holds diag on all of S(A^2p) */
//        1,       /* incx */
//        &((MPF_ComplexDouble *) context->V)[m_B*current_blk*i], /* V =[d0 d1 ... dp-1] */
//        1);      /* incy */
//
//      /* swaps context->B and context->X */
//      swap = context->B;
//      context->B = context->X;
//      context->X = swap;
//    }
//  }
//
//  /*------------------------------------------*/
//  /* solves LS problem using QR factorization */
//  /*------------------------------------------*/
//
//  MPF_Int blk_fA = context->blk_fA;
//  /* creates W */
//  mp_matrix_z_set(MPF_COL_MAJOR, m_B, 1, e_vector, m_B, &ONE_C);
//  memcpy(temp_V, context->V, (sizeof *temp_V)*m_B*blk*(n_diags-1));
//
//  /* computes QR factorization */
//  mp_qr_zge_givens_3(&((MPF_ComplexDouble*)context->V)[m_B*blk], m_B, e_vector,
//    m_B, m_B, (n_diags-1)*blk, 1, temp_matrix);
//
//  /* solves system of equations */
//  mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//    blk*(n_diags-1), 1, &ONE_C, &((MPF_ComplexDouble *)context->V)[m_B*blk],
//    m_B, e_vector, m_B);
//
//  /* reconstruction */
//  for (i = 0; i < blk*(n_diags-1); ++i)
//  {
//    mp_zaxpy
//     (m_B,            /* rows */
//     &e_vector[i],    /* */
//     &temp_V[m_B*i],  /* holds diag on all of S(A^2p) */
//     1,               /* incx */
//     &((MPF_ComplexDouble*)context->diag_fA)[m_B*(i%blk_fA)],
//     1);              /* incy */
//  }
//}
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
//(1) colorings are inputs, A^{i}*probing_vectors are computed via Horner's
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
//(1) HORNER's method for evaluating matrix polynomial.
//
//A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1}
//  = a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//    -------------------------------------------------------------
//          restructuring that is used in Horner's method
//
//(2) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
//A^p*V(:, I)
//  = [a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)]*V(:, I)
//  = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... +A(a{p-1}V(:,I) + ap*A*V(:,I))...)
//
//INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//----------
//INITIAL: for i = 0:p-2, with V(:,I)^(0) <- ap-2*V(:,I) + ap-1*A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//In following code ai = 1.0 so:
//INITIAL: for i = 0:p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//
//==============================================================================*/
//void mpf_batch_d_spai_ls
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Sparse *fA
//)
//{
//  MPF_Int n_batches = context->n_blocks;
//
//  //MPF_Int blk = context->solver_blk;
//  context->B.data = context->mem_outer;
//  context->X.data = &((double*)context->B.data)[context->B.m*context->B.n];
//  context->V = &((double*)context->X.data)[context->X.m*context->X.n];
//
//  if (context->data_type == MPF_REAL)
//  {
//    double *swap = NULL;
//    for (MPF_Int z = 0; z < n_batches; ++z) /* parsing batches */
//    {
//      /* gets current blk and generates rhs */
//      MPF_Int current_rhs = blk*z;
//      MPF_Int current_blk = (1-z/(n_blocks-1))*blk+(z/(n_blocks-1))*(n_max_B-current_rhs);
//
//      mp_d_generate_B(context->blk_max_fA, context->colorings_array,
//        current_rhs, context->B.m, current_blk, context->B.data);
//
//      for (MPF_Int k = 0; k < n_levels; ++k)
//      {
//        mp_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, context->A_handle,
//          context->A_descr, MPF_SPARSE_COL_MAJOR, context->B, current_blk, m_B,
//          0.0, context->X, m_B);
//        mp_d_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//          m_B, context->X, current_rhs, current_blk);
//
//        /* sums columns of rhs */
//        for (p = 0; p < blk; ++p)
//        {
//          mp_daxpy(
//            m_B,                           /* rows */
//            1.0,                           /* */
//            &((double*)context->X)[m_B*p], /* holds diagonall of S(A^2p) */
//            1,                             /* incx */
//            &((double*)context->V)[m_B*i], /* holds diagonals [d0 d1 ... dp-1] */
//            1);                            /* incy */
//        }
//
//        /* swaps context->B and context->X */
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//
//    /* creates W */
//    double *e_vector = mp_malloc(sizeof(double)*m_B*blk);
//    mp_matrix_d_set(MPF_COL_MAJOR, m_B, 1, e_vector, m_B, 1.0);
//    /* solves least squares problem VX = W */
//    double *temp_matrix = (double *) &((double *)
//      context->V)[m_B*((MPF_Int) pow(2, n_levels))];
//    mp_qr_givens_mrhs_dge(pow(2, n_levels), pow(2, n_levels), context->V, m_B,
//      context->X, m_B, temp_matrix); //context->temp_memory);
//    mp_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//      blk, 1, 1.0, context->V, m_B, e_vector, n_levels);
//
//    /* reconstruction */
//    mp_dgemm(
//      CblasColMajor,    /* ordering */
//      MPF_BLAS_NO_TRANS, /* transpose operator V */
//      MPF_BLAS_NO_TRANS, /* transpose operator e_vector */
//      m_B,              /* rows of V */
//      1,                /* num_cols of e_vector */
//      blk,
//      1.0,              /* multiplier */
//      context->V,       /* */
//      m_B,              /* lead dimension of V */
//      e_vector,
//      m_B,              /* lead dimension of e_vector */
//      0.0,
//      context->diag_fA, /* diagonal vector result (output) */
//      m_B);             /* lead dimension of output */
//  }
//  else if (context->data_type == MPF_COMPLEX)
//  {
//    MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//    MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//    MPF_ComplexDouble *swap = NULL; /* swaps context->B and context->X */
//
//    /*--------------------------------------------------------*/
//    /* exact computation of diagonals I, A, A^2, ..., A^{k-1} */
//    /*--------------------------------------------------------*/
//
//    /* parses blocks of current batch*/
//    for (j = 0; j < n_blocks; ++j)
//    {
//      /* gets current blk and generates rhs */
//      current_rhs = blk*z;
//      current_blk = (1-z/(n_blocks-1))*blk + (z/(n_blocks-1))*(n_max_B-current_rhs);
//      mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      for (k = 0; k < n_levels; ++k)
//      {
//        /* updates context->X */
//        mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C,
//          context->A_handle, context->A_descr, MPF_SPARSE_COL_MAJOR,
//          context->B, blk, m_B, ZERO_C, context->X, m_B);
//
//        /* sparsifies temp_X */
//        mp_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//          m_B, context->X, current_rhs, current_blk);
//
//        /* gathers entries of A_inv */
//        for (p = 0; p < blk; ++p)
//        {
//          mp_zaxpy(m_B, &ONE_C, &((MPF_ComplexDouble*)context->X)[m_B*p], 1,
//            &((MPF_ComplexDouble *) context->V)[m_B*i], 1);
//        }
//
//        /* swaps B and X */
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//
//    /* creates W */
//    MPF_ComplexDouble *e_vector = mp_malloc(sizeof(MPComplexDouble)*m_B*blk);  /* accumulator */
//    mp_matrix_z_set(MPF_COL_MAJOR, m_B, 1, e_vector, m_B, &ONE_C);
//
//    /* solves least squares problem VX = W */
//    //mp_qr_givens_mrhs_zsy_factorize(context->solver_outer_batch_size,
//    //context->solver_outer_batch_size, context->V, context->rhs_num_rows,
//    //context->X, context->rhs_num_rows, context->temp_memory);
//    /* br <-- solves triangular H*y = br */
//    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans , CblasNonUnit,
//      context->blk_solver, 1, &ONE_C, context->V, m_B, e_vector,
//      n_levels);
//
//    /* reconstruction */
//    mp_zgemm(
//      CblasColMajor,   /* ordering */
//      MPF_BLAS_NO_TRANS,/* transpose operator V */
//      MPF_BLAS_NO_TRANS,/* transpose operator e_vector */
//      m_B,             /* rows of V */
//      1,               /* num_cols of e_vector */
//      blk,             /* */
//      &ONE_C,          /* multiplier */
//      context->V,      /* */
//      m_B,             /* lead dimension of V */
//      e_vector,
//      m_B,             /* lead dimension of e_vector */
//      &ZERO_C,
//      context->diag_fA,/* diagonal vector result (output) */
//      m_B);            /* lead dimension of output */
//  }
//}
//
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
//(1) colorings are inputs, A^{i}*probing_vectors are computed via Horner's
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
//applies horner iterations to current batch of B
//
//(3) HORNER's method for evaluating matrix polynomial.
//
//A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1}
//    = a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//      -------------------------------------------------------------
//
//(4) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
// A^p*V(:, I) = [a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I
//                + ap*A))...)]*V(:, I)
//             = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... +A(a{p-1}V(:,I)
//                + ap*A*V(:,I))...)
//
//INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//----------
//INITIAL: for i = 0:p-2, with V(:,I)^(0) <- ap-2*V(:,I) + ap-1*A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//In following code ai = 1.0 so:
//INITIAL: starting from i = 0 -> p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//
//==============================================================================*/
//void mpf_batch_d_blk_ls_horner
//(
//  MPF_Solver *context,
//  MPF_Sparse *A,
//  MPF_Sparse *diag_fA
//)
//{
//
//  MPF_Int current_blk = 0;
//  MPF_Int n_blocks = 0;
//  MPF_Int n_max_B = context->n_max_B;
//  MPF_Int blk_fA = context->blk_fA;
//  MPF_Int blk_max_fA = context->blk_max_fA;
//
//  /* initializes n_blocks */
//  MPF_Int n_batches
//    = (MPF_Int)((double)context->n_max_B / (double)context->batch+0.5);
//
//  if (context->data_type == MPF_REAL)
//  {
//    double *swap = NULL;  /* used for swapping context->B and context->X */
//    double *temp_X = NULL;
//    double *E_vecblk = NULL;
//    double *temp_matrix = NULL;   /* solves least squares problem VX = W */
//    double *temp_V = NULL;
//
//    context->B = context->memory_outer;
//    context->X = &((double *)context->B)[m_B*blk];
//    context->V = &((double *)context->X)[m_B*blk];
//    temp_X = &((double*)context->V)[m_B*blk*n_diags];
//    temp_V = &((double*)temp_X)[m_B*blk];
//    E_vecblk = &((double*)temp_V)[m_B*blk*n_diags];
//    temp_matrix = &((double*)E_vecblk)[m_B*blk];
//
//    mp_matrix_d_diag_set(MPF_COL_MAJOR, m_B, blk, context->V, m_B, 1.0);
//
//    /*--------------------------------------------*/
//    /* extracts diagonals in V = [d0 d1 ... dp-1] */
//    /*--------------------------------------------*/
//
//    for (j = 0; j < n_blocks; ++j)  /* forall blocks */
//    {
//      current_rhs = blk*j;
//      current_blk =
//          (1-j/(n_blocks-1))*blk + (j/(n_blocks-1))*(n_max_B-current_rhs);
//
//      mp_d_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      for (i = 1; i < n_diags; ++i)
//      {
//        /* updates rhs and applies probing vectors elementwise multiplication */
//        mp_sparse_d_mm(
//          SPARSE_OPERATION_NON_TRANSPOSE,
//          1.0,
//          context->A_handle,
//          context->A_descr,
//          MPF_SPARSE_COL_MAJOR,
//          context->B, /* ok computes context->B = A*context->X */
//          current_blk,
//          m_B,
//          0.0,
//          context->X,
//          m_B);
//
//        memcpy(temp_X, context->X, (sizeof *temp_X)*m_B*blk);
//        mp_d_blk_select_X_dynamic(blk_max_fA, context->memory_colorings, m_B,
//          temp_X, blk_fA, current_rhs, current_blk);
//
//        mp_daxpy(
//          m_B*blk,
//          1.0,
//          temp_X,
//          1,
//          &((double*)context->V)[m_B*blk*i],
//          1);
//
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//
//    /*------------------------------------------*/
//    /* solves LS problem using QR factorization */
//    /*------------------------------------------*/
//    MPF_Int blk_fA = context->blk_fA;
//    MPF_Int m_B = context->m_B;
//    MPF_Int n_diags = mp_n_diags_get(context->n_levels, context->degree);
//    mp_matrix_d_diag_set(MPF_COL_MAJOR, m_B, blk, E_vecblk, m_B, 1.0);
//    memcpy(temp_V, context->V, (sizeof *temp_V)*m_B*blk_fA*(n_diags-1));
//    mp_matrix_d_announce(context->V, 30, n_diags*blk_fA, m_B, "(before) V");
//    /* computes QR factorization of V */
//    mp_qr_givens_mrhs_dge_2(m_B, blk_fA*(n_diags-1), blk_fA,
//      &((double *)context->V)[m_B*blk], m_B, E_vecblk, m_B, temp_matrix);
//    mp_matrix_d_announce(context->V, 30, n_diags*blk_fA, m_B, "(before) V");
//    mp_matrix_d_announce(E_vecblk, 30, blk, m_B, "E (after factorization)");
//    /* solves upper triangular system of equations */
//    mp_dtrsm(
//      CblasColMajor,
//      CblasLeft,
//      CblasUpper,
//      CblasNoTrans,
//      CblasNonUnit,
//      blk_fA*(n_diags-1),
//      blk_fA,
//      1.0,
//      &((double *)context->V)[m_B*blk],
//      m_B,
//      E_vecblk,
//      m_B);
//    mp_matrix_d_announce(E_vecblk, 30, blk, m_B, "E (coeffients)");
//
//    /* reconstruction */
//    mp_dgemm(
//      CblasColMajor,       /* ordering */
//      MPF_BLAS_NO_TRANS,    /* transpose operator V */
//      MPF_BLAS_NO_TRANS,    /* transpose operator e_vector */
//      m_B,                 /* rows of V */
//      blk_fA,              /* num_cols of e_vector */
//      blk_fA*(n_diags-1),  /* */
//      1.0,                 /* multiplier */
//      temp_V,              /* */
//      m_B,                 /* lead dimension of V */
//      E_vecblk,            /* [1;1;...;1] vector */
//      m_B,                 /* lead dimension of e_vector */
//      1.0,
//      context->diag_fA,    /* diagonal vector result (output) */
//      m_B);                /* lead dimension of output */
//
//    mp_matrix_d_announce(context->diag_fA, 30, blk, m_B, "diag_approx");
//
//    temp_matrix = NULL;
//    E_vecblk = NULL;
//    temp_V = NULL;
//    temp_X = NULL;
//  }
//  else if (context->data_type == MPF_COMPLEX)
//  {
//    MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//    MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//
//    MPF_ComplexDouble *swap = NULL; /* swaps context->B and context->X */
//    MPF_ComplexDouble *temp_X = NULL;
//    MPF_ComplexDouble *E_vecblk    = NULL;
//    MPF_ComplexDouble *temp_matrix = NULL;
//    MPF_ComplexDouble *temp_V      = NULL;
//
//    context->B = context->memory_outer;
//    context->X = &((MPF_ComplexDouble *) context->B)[m_B*blk];
//    context->V = &((MPF_ComplexDouble *) context->X)[m_B*blk];
//    temp_X = &((MPF_ComplexDouble*)context->V)[m_B*blk*n_diags];
//    temp_V = &((MPF_ComplexDouble*)temp_X)[m_B*blk];
//    E_vecblk = &((MPF_ComplexDouble*)temp_V)[m_B*blk*n_diags];
//    temp_matrix = &((MPF_ComplexDouble*)E_vecblk)[m_B*blk];
//    mp_matrix_z_diag_set(MPF_COL_MAJOR, m_B, blk, context->V, m_B, ONE_C);
//
//    /*-------------------------------------------*/
//    /* extracts diagonals in V = [d0 d1 ... dp-1]*/
//    /*-------------------------------------------*/
//    for (j = 0; j < n_blocks; ++j)  /* forall blocks */
//    {
//      current_rhs = blk*j;
//      current_blk =
//          (1-j/(n_blocks-1))*blk + (j/(n_blocks-1))*(n_max_B-current_rhs);
//
//      mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      for (i = 1; i < n_diags; ++i) /* forall diagonals */
//      {
//        mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
//                       ONE_C,
//                       context->A_handle,
//                       context->A_descr,
//                       MPF_SPARSE_COL_MAJOR,
//                       context->B, /* ok computes context->B = A*context->X */
//                       blk,
//                       m_B,
//                       ZERO_C,
//                       context->X,
//                       m_B);
//
//        memcpy(temp_X, context->X, (sizeof *temp_X)*m_B*blk);
//        mp_z_blk_select_X_dynamic(blk_max_fA, context->memory_colorings, m_B,
//          temp_X, blk_fA, current_rhs, current_blk);
//
//        mp_zaxpy(m_B*blk,
//                 &ONE_C,
//                 temp_X,
//                 1,
//                 &((MPF_ComplexDouble*)context->V)[m_B*blk*i],
//                 1);
//
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//
//    /*-----------------------------------------------------------*/
//    /* solves least squares problem X = argmin{||VX_arg - W||_F} */
//    /*-----------------------------------------------------------*/
//    MPF_Int blk_fA = context->blk_fA;
//    MPF_Int n_diags = mp_n_diags_get(context->n_levels, context->degree);
//    mp_matrix_z_diag_set(MPF_COL_MAJOR, m_B, blk, E_vecblk, m_B, ONE_C);
//    memcpy(temp_V, context->V, (sizeof *temp_V)*m_B*blk_fA*(n_diags-1));
//    mp_matrix_z_announce(context->V, 30, 2*blk_fA, m_B, "(before) V");
//    /* computes qr factorization */
//    mp_qr_zge_givens_mrhs_2 (&((MPF_ComplexDouble*)context->V)[m_B*blk_fA],
//      E_vecblk, m_B, blk_fA*(n_diags-1), blk_fA, temp_matrix);
//    mp_matrix_z_announce(context->V, 30, 2*blk_fA, m_B, "(after) V");
//    mp_matrix_z_announce(E_vecblk, 30, blk_fA, m_B, "E (after factorization)");
//    /* solves upper triangular system of equations */
//    mp_ztrsm(CblasColMajor,
//             CblasLeft,
//             CblasUpper,
//             CblasNoTrans,
//             CblasNonUnit,
//             blk_fA*(n_diags-1),
//             blk_fA,
//             &ONE_C,
//             &((MPF_ComplexDouble *)context->V)[m_B*blk],
//             m_B,
//             E_vecblk,
//             m_B);
//    /* reconstruction */
//    mp_zgemm(CblasColMajor,       /* ordering */
//             MPF_BLAS_NO_TRANS,    /* transpose operator V */
//             MPF_BLAS_NO_TRANS,    /* transpose operator e_vector */
//             m_B,                 /* rows of V */
//             blk_fA,              /* num_cols of e_vector */
//             blk_fA*(n_diags-1),  /* */
//             &ONE_C,              /* multiplier */
//             temp_V,              /* */
//             m_B,                 /* lead dimension of V */
//             E_vecblk,            /* [1;1;...;1] vector */
//             m_B,                 /* lead dimension of e_vector */
//             &ONE_C,
//             context->diag_fA,    /* diagonal vector result (output) */
//             m_B);                /* lead dimension of output */
//    temp_matrix = NULL;
//    E_vecblk = NULL;
//    temp_V = NULL;
//    temp_X = NULL;
//  }
//}
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
//(1) colorings are inputs, A^{i}*probing_vectors are computed via Horner's
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
//==============================================================================*/
//void mp_block_least_squares_horner_matrix
//(
//  MPF_Context *context
//)
//{
//  MPF_Int z = 0; /* for batches */
//  MPF_Int k = 0; /* for levels*/
//  MPF_Int p = 0; /* for reduction step */
//  MPF_Int i = 0;
//
//  MPF_Int current_rhs = 0;
//  MPF_Int n_max_B = context->n_max_B;
//  MPF_Int current_blk = 0;
//  MPF_Int blk = context->blk_solver;
//  MPF_Int m_B = context->m_B;
//  MPF_Int n_levels = context->n_levels;
//
//  MPF_Int n_blocks = context->n_blocks_solver;
//
//  /* set this up for testing */
//  context->blk_fA = context->blk_solver;
//
//  /* exact computation of diagonals I, A, A^2, ..., A^{k-1} */
//  if (context->data_type == MPF_REAL)
//  {
//    context->B = context->memory_outer;
//    context->X = &((double*)context->B)[m_B*blk];
//    context->V = &((double*)context->X)[m_B*blk];
//    double *swap = NULL;  /* used for swapping context->B and context->X */
//    for (z = 0; z < n_blocks; ++z)  /* parsing batches */
//    {
//      /* gets current blk and generates rhs */
//      current_rhs = blk*z;
//      current_blk = (1-z/(n_blocks-1))*blk + (z/(n_blocks-1))*(n_max_B-current_rhs);
//      mp_d_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      /* applies horner iterations to current batch of B
//         (1) HORNER's method for evaluating matrix polynomial.
//
//         A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1} = a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//                                                                  -------------------------------------------------------------
//                                                                        restructuring that is used in Horner's method
//
//         (2) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
//          A^p*V(:, I) = [a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)]*V(:, I)
//                      = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... +A(a{p-1}V(:,I) + ap*A*V(:,I))...)
//
//          INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//          ----------
//          INITIAL:     starting from i = 0 -> p-2, with V(:,I)^(0) <- ap-2*V(:,I) + ap-1*A*V(:,I);
//          LOOP:        V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//          In following code ai = 1.0 so:
//          INITIAL:     starting from i = 0 -> p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//          LOOP:        V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//      */
//      for (k = 0; k < n_levels; ++k)
//      {
//        /* updates rhs and applies probing vectors elementwise multiplication */
//        mp_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, context->A_handle,
//          context->A_descr, MPF_SPARSE_COL_MAJOR, context->B, current_blk,
//          m_B, 0.0, context->X, m_B);
//        mp_d_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//          m_B, context->X, current_rhs, current_blk);
//
//        /* ACCUMULATION: sums columns of rhs
//           computes V(:,I) = V(:,I-blk)
//
//             V(:,I0) <- V(:,I0) + (A*V(:,Ij)).*V(:,Ij)
//
//           generalization:
//           ---------------
//
//             V(:,I0) <- V(:,I0) + (A*V(:,Ij)) (.o) V(:,Ij) where operator .o is block-wise multiplication operator.
//
//                                                                                                                            */
//        /*** ADAPT FOR DIAGONAL BLOCKS!!! ***/
//        //for (p = 0; p < context->solver_blk; p++)
//        //{
//            //printf("z: %d, j: %d, k: %d, p: %d\n", z, j, k, p);
//            mp_daxpy(m_B*blk,
//              1.0,
//              &((double *) context->X)[m_B*blk*p],
//              1,
//              &((double *) context->V)[m_B*blk*p],
//              1);
//        //}
//
//        /* swaps B and X */
//        //printf("here\n");
//        if (k < n_levels-1)
//        {
//          swap = context->B;
//          context->B = context->X;
//          context->X = swap;
//        }
//      }
//
//    }
//
//    /* solves sytem least squares VX = W problem */
//    MPF_Int blk_fA = context->blk_fA;
//    MPF_Int n_diags = mp_n_diags_get(context->n_levels, context->degree);
//    double *E_vecblk = mp_malloc(sizeof(double)*m_B*blk_fA);
//    double *temp_matrix = (double *) mp_malloc(sizeof(double)*m_B*2);   /* solves least squares problem VX = W */
//    double *temp_V = mp_malloc(sizeof(double)*m_B*blk_fA*p);
//    /* !!! WORKING CURRENTLY: MAKE MRHS_BLOCK_QR */
//    //mp_block_qr_givens_dge_factorize(context->blk_fA*context->probing_num_levels, context->blk_fA, context->V,
//    //                                 context->rhs_num_rows, context->X, context->rhs_num_rows, context->blk_fA);
//
//    memcpy(temp_V, context->V, (sizeof *temp_V)*m_B*blk_fA*(n_diags-1));
//    mp_matrix_d_announce(context->V, 30, n_diags*blk_fA, m_B, "(before) V");
//    mp_qr_givens_mrhs_dge(blk_fA*(n_diags-1), blk_fA, context->V,
//      m_B, E_vecblk, m_B, temp_matrix);
//
//    mp_dtrsm(CblasColMajor,
//             CblasLeft,
//             CblasUpper,
//             CblasNoTrans,
//             CblasNonUnit,
//             blk_fA*(n_diags-1),
//             blk_fA,  /* br <-- solves triangular H*y = br */
//             1.0,
//             context->V,
//             m_B,
//             E_vecblk,
//             blk_fA);
//
//    /* reconstruction */
//    for (i = 0; i < blk_fA*(n_diags-1); ++i)
//    {
//      mp_dgemm(CblasColMajor,       /* ordering */
//               MPF_BLAS_NO_TRANS,    /* transpose operator V */
//               MPF_BLAS_NO_TRANS,    /* transpose operator e_vector */
//               m_B,                 /* rows of V */
//               blk_fA*(n_diags-1),  /* num_cols of e_vector */
//               blk_fA,              /* */
//               1.0,                 /* multiplier */
//               temp_V,              /* */
//               m_B,                 /* lead dimension of V */
//               E_vecblk,            /* [1;1;...;1] vector */
//               m_B,                 /* lead dimension of e_vector */
//               0.0,
//               context->diag_fA,    /* diagonal vector result (output) */
//               m_B);                /* lead dimension of output */
//    }
//    mp_free(temp_matrix);
//    mp_free(E_vecblk);
//    mp_free(temp_V);
//  }
//  /*** working on this (least squares for complex arithmetic, use complex,
//       not complex symmetric qr routines ) ***/
//  /* exact computation of diagonals I, A, A^2, ..., A^{k-1} */
//  else if (context->data_type == MPF_COMPLEX)
//  {
//    MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//    MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//    MPF_ComplexDouble *swap = NULL;/* used for swapping context->B, context->X */
//    for (z = 0; z < n_blocks; ++z)  /* parsing batches */
//    {
//      /* gets current blk and generates rhs */
//      current_rhs = blk*z;
//      current_blk = (1-z/(n_blocks-1))*blk + (z/(n_blocks-1))*(n_max_B-current_rhs);
//      mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      /* HORNER's method */
//      for (k = 0; k < n_levels; ++k)
//      {
//        /* updates rhs and applies probing vectors elementwise multiplication */
//        mp_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, ONE_C, context->A_handle,
//          context->A_descr, MPF_SPARSE_COL_MAJOR, context->B, blk, m_B, ZERO_C,
//          context->X, m_B);
//        mp_z_select_X_dynamic(context->blk_max_fA, context->memory_colorings,
//          m_B, context->X, current_rhs, current_blk);
//
//        /* sums columns of rhs */
//        for (p = 0; p < blk; ++p)
//        {
//          mp_zaxpy(m_B*blk, &ONE_C, &((MPF_ComplexDouble*)context->X)[m_B*blk*p],
//            1, &((MPF_ComplexDouble*)context->acc)[m_B*i], 1);
//        }
//        /* swaps B and X */
//        if (k < n_levels-1)
//        {
//          swap = context->B;
//          context->B = context->X;
//          context->X = swap;
//        }
//      }
//    }
//
//    /* solves least squares problem VX = W */
//    MPF_ComplexDouble *e_vector = mp_malloc(sizeof(MPComplexDouble)*m_B*blk);  /* accumulator */
//    mp_matrix_z_set(MPF_COL_MAJOR, m_B, 1, e_vector, m_B, &ONE_C);
//    //mp_qr_givens_mrhs_zsy_factorize(context->solver_outer_batch_size,
//    //context->solver_outer_batch_size, context->V, context->rhs_num_rows,
//    //context->X, context->rhs_num_rows, context->temp_memory);
//    mp_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
//      blk, 1, &ONE_C, context->V, m_B, e_vector, n_levels);
//
//    /* reconstruction */
//    mp_zgemm(CblasColMajor,     /* ordering */
//             MPF_BLAS_NO_TRANS,  /* transpose operator V */
//             MPF_BLAS_NO_TRANS,  /* transpose operator e_vector */
//             m_B,               /* rows of V */
//             1,                 /* num_cols of e_vector */
//             blk,               /* */
//             &ONE_C,            /* multiplier */
//             context->V,        /* */
//             m_B,               /* lead dimension of V */
//             e_vector,
//             m_B,               /* lead dimension of e_vector */
//             &ZERO_C,
//             context->diag_fA,  /* diagonal vector result (output) */
//             m_B);              /* lead dimension of output */
//  }
//}
//
//void mp_solver_least_squares_dynamic(MPF_Context *context)
//{
//
//}
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
//(1) colorings are inputs, A^{i}*probing_vectors are computed via Horner's
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
//parses blocks of current batch
//applies horner iterations to current batch of B
//
//(3) HORNER's method for evaluating matrix polynomial.
//
//A^p = a0*I + a1*A + a2*A^2 + a3*A^3 + ... + ap*A^{p-1} = a0I + A(a1*I
//      + A(a2*I + A(a3*I + ... + A(a{p-1}I + ap*A))...)
//
//(4) HORNER's method for evaluating matrix polynomial times a block of vectors.
//
// A^p*V(:, I) = [a0I + A(a1*I + A(a2*I + A(a3*I + ... + A(a{p-1}I
//                + ap*A))...)]*V(:, I)
//             = a0V(:,I) + A(a1*V(:,I) + A(a3*V(:,I) + ... +A(a{p-1}V(:,I)
//                + ap*A*V(:,I))...)
//
//INVARIANT: V(:,I)^(i) = (ai*I+A)*V(:,I)^(i)
//----------
//INITIAL: for i = 0:p-2, with V(:,I)^(0) <- ap-2*V(:,I) + ap-1*A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- a{p-i-2}*V(:,I)^(i) + A*V(:,I)^(i)
//
//In following code ai = 1.0 so:
//INITIAL: starting from i = 0 -> p-2, with V(:,I)^(0) <- V(:,I) + A*V(:,I);
//   LOOP: V(:,I)^(i+1) <- V(:,I)^(i) + A*V(:,I)^(i)
//
//==============================================================================*/
//void mp_global_ls_horner
//(
//  MPF_Context *context
//)
//{
//  MPF_Int j = 0;
//  MPF_Int i = 0;
//  MPF_Int current_rhs = 0;
//  MPF_Int m_B = context->m_B;
//  MPF_Int blk = context->blk_solver;
//  MPF_Int n_diags = mp_n_diags_get(context->n_levels, context->degree);
//  MPF_Int blk_fA = context->blk_fA;
//  MPF_Int blk_max_fA = context->blk_max_fA;
//  MPF_Int n_blocks = context->n_blocks;
//  MPF_Int current_blk;
//  MPF_Int n_max_B = context->n_max_B;
//
//  /* initializes n_blocks */
//  context->n_blocks_solver
//    = (MPF_Int)((double)context->n_max_B / (double)context->blk_solver+0.5);
//  n_blocks = context->n_blocks_solver;
//
//  /* exact computation of diagonals I, A, A^2, ..., A^{k-1} */
//  if (context->data_type == MPF_REAL)
//  {
//    double *swap = NULL; /* used for swapping context->B and context->X */
//    double *temp_X = NULL;
//    double *temp_V = NULL;
//    double *e_vector = NULL;
//    double *temp_matrix = NULL;
//
//    context->B = context->memory_outer;
//    context->X = &((double*)context->B)[m_B*blk];
//    context->V = &((double*)context->X)[m_B*blk];
//    temp_X = &((double*)context->V)[m_B*blk*n_diags];
//    temp_V = &((double*)temp_X)[m_B*blk];
//    e_vector = &((double*)temp_X)[m_B*blk*n_diags];
//    temp_matrix = &((double*)e_vector)[m_B];
//
//    mp_matrix_d_diag_set(MPF_COL_MAJOR, m_B, blk, context->V, m_B, 1.0);
//
//    /*------------------------------------------------*/
//    /* extracts diagonals V = [d0, d1, d2, ..., dp-1] */
//    /*------------------------------------------------*/
//
//    for (j = 0; j < n_blocks; ++j)  /* forall blocks */
//    {
//      current_rhs = blk*j;
//      current_blk =
//          (1-j/(n_blocks-1))*blk + (j/(n_blocks-1))*(n_max_B-current_rhs);
//      mp_d_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      for (i = 1; i < n_diags; ++i)
//      {
//        mp_sparse_d_mm(
//          SPARSE_OPERATION_NON_TRANSPOSE,
//          1.0,
//          context->A_handle,
//          context->A_descr,
//          MPF_SPARSE_COL_MAJOR,
//          context->B, /* ok computes context->B = A*context->X */
//          blk,
//          m_B,
//          0.0,
//          context->X,
//          m_B);
//        /* transfers X to temp_X and applies probing vectors */
//        memcpy(temp_X, context->X, (sizeof *temp_X)*m_B*blk);
//        mp_d_blk_select_X_dynamic(blk_max_fA, context->memory_colorings, m_B,
//          temp_X, blk_fA, current_rhs, current_blk);
//        /* gathers entries of diag_fA */
//        mp_daxpy(
//          m_B*blk,
//          1.0,
//          temp_X,
//          1,
//          &((double*)context->V)[m_B*blk*i],
//          1);
//        /* swaps context->B and context->X */
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//    mp_matrix_d_announce(context->V, 10, n_diags, m_B, "V (before)");
//
//    /*------------------------------------------*/
//    /* solves LS problem using QR factorization */
//    /*------------------------------------------*/
//
//    memcpy(temp_V, context->V, sizeof(double)*m_B*blk*(n_diags-1));
//    mp_matrix_d_diag_set(MPF_COL_MAJOR, m_B, blk, e_vector, m_B, 1.0);
//    /* computes QR factorization */
//    mp_qr_givens_dge_2(m_B*blk_fA, n_diags-1, 1,
//      &((double*)context->V)[m_B*blk], m_B*blk_fA, e_vector, m_B*blk_fA,
//      temp_matrix);
//    /* solves upper triangular system of equations */
//    mp_dtrsm(
//      CblasColMajor,
//      CblasLeft,
//      CblasUpper,
//      CblasNoTrans,
//      CblasNonUnit,
//      n_diags-1,
//      1,
//      1.0,
//      &((double*)context->V)[m_B*blk],
//      m_B*blk,
//      e_vector,
//      m_B*blk);
//    /* reconstruction */
//    mp_dgemm(
//      CblasColMajor,       /* ordering */
//      MPF_BLAS_NO_TRANS,    /* transpose operator V */
//      MPF_BLAS_NO_TRANS,    /* transpose operator e_vector */
//      m_B*blk_fA,          /* rows of V */
//      1,                   /* num_cols of e_vector */
//      n_diags-1,           /* */
//      1.0,                 /* multiplier */
//      temp_V,              /* */
//      m_B*blk_fA,          /* lead dimension of V */
//      e_vector,                                  /* [1;1;...;1] vector */
//      m_B*blk,             /* lead dimension of e_vector */
//      1.0,
//      context->diag_fA,    /* diagonal vector result (output) */
//      m_B*blk);            /* lead dimension of output */
//
//    temp_matrix = NULL;
//    e_vector = NULL;
//    temp_V = NULL;
//    temp_X = NULL;
//  }
//  else if (context->data_type == MPF_COMPLEX)
//  {
//    MPF_ComplexDouble ZERO_C = mp_scalar_z_init(0.0, 0.0);
//    MPF_ComplexDouble ONE_C = mp_scalar_z_init(1.0, 0.0);
//
//    MPF_ComplexDouble *swap = NULL; /* swaps context->B and context->X */
//    MPF_ComplexDouble *temp_X = NULL;
//    MPF_ComplexDouble *e_vector = NULL;
//    MPF_ComplexDouble *temp_matrix = NULL;
//    MPF_ComplexDouble *temp_V = NULL;
//
//    context->B = context->memory_outer;
//    context->X = &((MPF_ComplexDouble*)context->B)[m_B*blk];
//    context->V = &((MPF_ComplexDouble*)context->X)[m_B*blk];
//    temp_X = &((MPF_ComplexDouble*)context->V)[m_B*blk*n_diags];
//    e_vector = &((MPF_ComplexDouble*)temp_X)[m_B*blk];
//    temp_matrix = &((MPF_ComplexDouble*)e_vector)[m_B];
//    temp_V = &((MPF_ComplexDouble*)temp_matrix)[m_B*blk];
//
//    printf("n_blocks: %d, n_diags: %d\n", n_blocks, n_diags);
//
//    mp_matrix_z_diag_set(MPF_COL_MAJOR, m_B, blk, context->V, m_B, ONE_C);
//    for (j = 0; j < n_blocks; ++j)     /* parsing batches */
//    {
//      current_rhs = blk*j;
//      current_blk =
//          (1-j/(n_blocks-1))*blk + (j/(n_blocks-1))*(n_max_B-current_rhs);
//      mp_z_generate_B(context->blk_max_fA, context->memory_colorings,
//        current_rhs, m_B, blk, context->B);
//
//      for (i = 1; i < n_diags; ++i)
//      {
//        /* updates rhs and applies probing vectors elementwise multiplication */
//        mp_sparse_z_mm(
//          SPARSE_OPERATION_NON_TRANSPOSE,
//          ONE_C,
//          context->A_handle,
//          context->A_descr,
//          MPF_SPARSE_COL_MAJOR,
//          context->B, /* ok computes context->B = A*context->X */
//          blk,
//          m_B,
//          ZERO_C,
//          context->X,
//          m_B);
//
//        memcpy(temp_X, context->X, (sizeof *temp_X)*m_B*blk);
//        mp_z_blk_select_X_dynamic(blk_max_fA, context->memory_colorings, m_B,
//          temp_X, blk_fA, current_rhs, current_blk);
//
//        mp_zaxpy(
//          m_B*blk,
//          &ONE_C,
//          temp_X,
//          1,
//          &((MPF_ComplexDouble *) context->V)[m_B*blk*i],
//          1);
//
//        /* swaps context->B and context->X */
//        swap = context->B;
//        context->B = context->X;
//        context->X = swap;
//      }
//    }
//
//    /*-------------------------------------------*/
//    /* solves sytem least squares VX = W problem */
//    /*-------------------------------------------*/
//
//    mp_matrix_z_announce(context->V, 10, n_diags, m_B, "V (before)");
//
//    memcpy(temp_V, context->V, sizeof(MPF_ComplexDouble)*m_B*blk*(n_diags-1));
//    mp_matrix_z_diag_set(MPF_COL_MAJOR, m_B, blk, e_vector, m_B, ONE_C);
//
//    /* computes QR factorization */
//    mp_qr_zge_givens_3(&((MPF_ComplexDouble*)context->V)[m_B*blk_fA], m_B*blk_fA,
//      e_vector, m_B*blk_fA, m_B*blk_fA, n_diags-1, 1, temp_matrix);
//
//    /* solves upper triangular system of equations */
//    mp_ztrsm(
//      CblasColMajor,
//      CblasLeft,
//      CblasUpper,
//      CblasNoTrans,
//      CblasNonUnit,
//      n_diags-1,
//      1,
//      &ONE_C,
//      &((MPF_ComplexDouble *)context->V)[m_B*blk],
//      m_B*blk,
//      e_vector,
//      m_B*blk);
//
//    /* reconstruction */
//    mp_zgemm(
//      CblasColMajor,       /* ordering */
//      MPF_BLAS_NO_TRANS,    /* transpose operator V */
//      MPF_BLAS_NO_TRANS,    /* transpose operator e_vector */
//      m_B*blk_fA,          /* rows of V */
//      1,                   /* num_cols of e_vector */
//      n_diags-1,
//      &ONE_C,              /* multiplier */
//      temp_V,
//      m_B*blk_fA,          /* lead dimension of V */
//      e_vector,            /* [1;1;...;1] vector */
//      m_B*blk_fA,          /* lead dimension of e_vector */
//      &ONE_C,
//      context->diag_fA,    /* diagonal vector result (output) */
//      m_B*blk_fA);         /* lead dimension of output */
//
//    mp_matrix_z_announce(context->diag_fA, 10, 1, m_B, "diag_fA");
//
//    temp_matrix = NULL;
//    e_vector = NULL;
//    temp_V = NULL;
//    temp_X = NULL;
//  }
//}
