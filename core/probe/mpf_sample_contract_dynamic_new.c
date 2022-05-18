//#include "mp.h"
//#include "mp_solve.h"
//#include "mp_aux.h"
//#include "mp_probe.h"
//
///* to use with mp_pattern_multisample_new.c file */
//
//void mp_sample_contract_dynamic
//(
//  MPContext *context,
//  MPSparseCsr *A,
//  MPSparseCsr *B,
//  MPInt coarse_op
//)
//{
//  #if MP_PRINTOUT
//    printf("coarse_op: %d\n", coarse_op);
//  #endif
//  MPInt n_edges_new = 0;
//  MPInt i = 0;
//  MPInt j = 0;
//  MPInt z = 0;
//  MPInt c = 0;
//  MPInt stride = 0;
//  MPInt offset_rows = 0;
//  MPInt swap = 0;
//
//  // mp_probing_sampling_offsets_get(context, &offset_rows, &offset_cols);
//  mp_probing_stride_get(context, &stride);
//
//  B->rows_start[0] = 0;
//  B->rows_end[0] = 0;
//  B->descr.mode = A->descr.mode;
//
//  #if MP_PRINTOUT
//    printf("offset_rows: %d, A->m: %d, stride: %d\n", offset_rows, A->m, stride);
//    printf("coarse_op: %d\n", coarse_op);
//  #endif
//  if (coarse_op == 0)
//  {
//    for (i = 0; i < A->m; i += stride) /* parse each row */
//    {
//      if (i > 0)
//      {
//        B->rows_start[i/stride] = B->rows_end[i/stride-1];
//        B->rows_end[i/stride] = B->rows_start[i/stride];
//      }
//
//      for (j = A->rows_start[i]; j < A->rows_end[i]; ++j)
//      {
//        c = A->cols[j];
//        if (c % stride == 0)
//        {
//          B->cols[n_edges_new] = c/stride;
//          ++n_edges_new;
//          B->rows_end[i/stride] = n_edges_new;
//        }
//      }
//    }
//  }
//  else if (coarse_op == 1)
//  {
//    printf("HERE\n");
//    for (i = 0; i < A->m; i += 1) /* parse each row */
//    {
//      if (i > 0)
//      {
//        B->rows_start[i/stride] = B->rows_end[i/stride-1];
//        B->rows_end[i/stride] = B->rows_start[i/stride];
//      }
//
//      for (j = A->rows_start[i]; j < A->rows_end[i]; ++j)
//      {
//        c = A->cols[j];
//        if (((i % stride) == 0) && (c % stride == 1))
//        {
//          B->cols[n_edges_new] = c/stride;
//          ++n_edges_new;
//          B->rows_end[i/stride] = n_edges_new;
//        }
//        else if (i % stride == 1) //if (i % stride > 0)
//        {
//          z = B->rows_end[i/stride];
//          while ((z > B->rows_start[i/stride]) && (B->cols[z] < B->cols[z-1]))
//          {
//            swap = B->cols[z-1];
//            B->cols[z-1] = B->cols[z];
//            B->cols[z] = swap;
//            z -= 1;
//          }
//        }
//        //if ((c % stride == stride - 1) || (i % stride == 1))
//        //{
//        //  B->cols[n_edges_new] = c/stride;
//        //  ++n_edges_new;
//        //  B->rows_end[i/stride] = n_edges_new;
//        //}
//
//      }
//
//    }
//  }
//
//  B->m = (A->m+stride-1)/stride;
//  B->nz = n_edges_new;
//  printf("A->m: %d, A->nz: %d, B->m: %d, B->nz: %d\n", A->m, A->nz, B->m, B->nz);
//  #if MP_PRINTOUT
//    printf("stride: %d\n", stride);
//    printf("n_edges_new: %d\n", n_edges_new);
//  #endif
//}

