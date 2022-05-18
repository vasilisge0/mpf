#include "mpf.h"

//void mpf_ls_init
//(
//  MPF_Solver *solver,
//  MPF_Int ld
//)
//{
////  solver->ld = ld;
////
////  if (solver->data_type == MPF_REAL)
////  {
////    solver->inner_type = MPF_SOLVER_DSY_LS;
////    solver->inner_function = &mpf_ls_dsy_horner;
////    solver->device = MPF_DEVICE_CPU;
////  }
////  else if ((solver->data_type == MPF_COMPLEX)
////          && (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
////  {
////    solver->inner_type = MPF_SOLVER_ZSY_LS;
////    solver->inner_function = &mpf_ls_zsy_horner;
////    solver->device = MPF_DEVICE_CPU;
////  }
////  else if ((solver->data_type == MPF_COMPLEX)
////          && (solver->matrix_type == MPF_MATRIX_HERMITIAN))
////  {
////    solver->inner_type = MPF_SOLVER_ZHE_LS;
////    solver->inner_function = &mpf_ls_zhe_horner;
////    solver->device = MPF_DEVICE_CPU;
////  }
////
////  solver->inner_alloc_function = &mpf_cg_alloc;
////  solver->inner_free_function = &mpf_krylov_free;
////  solver->inner_mem = ;
////  solver->bytes_inner = sizeof(double)*
////    (solver->ld*solver->blk_solver*(n_diags) /* size V */
////    +solver->ld*solver->blk_solver           /* size tempf_X */
////    +solver->ld*solver->blk_solver*(n_diags) /* size tempf_V */
////    +solver->ld                              /* size e_vector */
////    +n_diags*2);                             /* size tempf_matrix (QR) */
//}
