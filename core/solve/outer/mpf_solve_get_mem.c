// #include "mpf.h"
//
// void mpf_outer_solve_get_mem
// (
//   MPF_Solver *solver,
//   MPF_Sparse *A,
//   MPF_Dense *diag_fA
// )
// {
//
//   switch(solver->solver.outer_type)
//   {
//     case MPF_BATCH:
//     {
//       if (solver->data_type == MPF_REAL)
//       {
//         solver->bytes_outer = sizeof(double)*(A->m*solver->batch*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*A->M*solver->batch;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)*(A->m*solver->batch*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*fA->m*fA->n;
//       }
//       break;
//     }
//     case MPF_BATCH_2PASS:
//     {
//       if (solver->data_type == MPF_REAL)
//       {
//         solver->bytes_outer =
//           sizeof(double)*(solver->ld*solver->blk_solver*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*fA->m*solver->blk_fA;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)
//           *(solver->ld*solver->blk_solver*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*A->m *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_BATCH_DEFL:
//     {
//       if (solver->data_type == MPF_REAL)
//       {
//         solver->bytes_outer =
//           sizeof(double)*(solver->ld*solver->blk_solver*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         printf("OUTER\n");
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)
//           *(solver->ld*solver->blk_solver*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_BATCH_DEFL_SEED:
//     {
//       if ((solver->data_type == MPF_REAL) && (solver->n_max_B > 0))
//       {
//         solver->bytes_outer =
//           sizeof(double)*(solver->ld*solver->n_max_B*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       }
//       else if ((solver->data_type == MPF_COMPLEX)  && (solver->n_max_B > 0))
//       {
//         printf("OUTER\n");
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)
//           *(solver->ld*solver->n_max_B*2);
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_CUDA_BATCH:
//     {
//     //  if (solver->data_type == MPF_REAL)
//     //  {
//     //    solver->bytes_outer =
//     //      sizeof(double)*(solver->ld*solver->blk*2);
//     //    solver->bytes_cuda_outer =
//     //      sizeof(double)*(solver->ld*solver->blk*2);
//     //    solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//     //  }
//     //  else if (solver->data_type == MPF_COMPLEX)
//     //  {
//     //    solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//     //      (solver->ld*solver->blk*2);
//     //    solver->bytes_cuda_outer = sizeof(cuDoubleComplex)*
//     //      (solver->ld*solver->blk*2);
//     //    solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//     //      *solver->blk_fA;
//     //  }
//     //  break;
//     //}
//     //case MPF_BATCH_CHEB:
//     //{
//     //  if (solver->data_type == MPF_REAL)
//     //  {
//     //    solver->bytes_outer = sizeof(double)*
//     //      (solver->ld*solver->blk_solver   /* B */
//     //      +solver->ld*solver->blk_solver   /* X */
//     //      +solver->ld*solver->blk_solver)  /* W */
//     //      +solver->bytes_ev_max
//     //      +solver->bytes_ev_min;
//     //    solver->bytes_cuda_outer = 0;
//     //    solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//     //  }
//     //  else if (solver->data_type == MPF_COMPLEX)
//     //  {
//     //    solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//     //      (solver->ld   /* B */
//     //      +solver->ld   /* X */
//     //      +solver->ld); /* W */
//     //  }
//     }
//     case MPF_SOLVER_OUTER_UNDEFINED:
//         break;
//     case MPF_SOLVER_OUTER_LEAST_SQUARES_HORNER_ESTIMATION:
//     {
//       /* add size of V */
//       /* size_V: 2^n_levels*/
//       //int num_diags = pow(2, solver->n_levels);
//       //MPF_Int blk = solver->blk_solver;
//       MPF_Int n_diags = mpf_n_diags_get(solver->n_levels, solver->degree);
//
//       if (solver->data_type == MPF_REAL)
//       {
//         solver->bytes_outer = sizeof(double)*
//           (solver->ld*solver->blk_solver           /* size B */
//           +solver->ld*solver->blk_solver           /* size X */
//           +solver->ld*solver->blk_solver*(n_diags) /* size V */
//           +solver->ld*solver->blk_solver           /* size tempf_X */
//           +solver->ld*solver->blk_solver*(n_diags) /* size tempf_V */
//           +solver->ld                               /* size e_vector */
//           +n_diags*2);                              /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         //solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//         //  (solver->ld*solver->blk_solver*2
//         //  +solver->ld*solver->blk_solver*(n_diags)  /* size V */
//         //  +solver->ld*2+solver->ld*2);
//
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//           (solver->ld*solver->blk_solver           /* size B */
//           +solver->ld*solver->blk_solver           /* size X */
//           +solver->ld*solver->blk_solver*(n_diags) /* size V */
//           +solver->ld*solver->blk_solver           /* size tempf_X */
//           +solver->ld*solver->blk_solver*(n_diags) /* size tempf_V */
//           +solver->ld                               /* size e_vector */
//           +n_diags*2);                              /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_SOLVER_OUTER_LEAST_SQUARES_DIAG_BLOCKS_HORNER_ESTIMATION:
//     {
//       MPF_Int n_diags = mpf_n_diags_get(solver->n_levels, solver->degree);
//
//       /* size_V: num_rows_rhs * 2^n_levels*/
//       if (solver->data_type == MPF_REAL)
//       {
//         //MPF_Int blk = solver->blk_solver;
//         //solver->bytes_outer = sizeof(double)*
//         //  (solver->ld*solver->blk_solver*2      /* size B and size X */
//         //  +solver->ld*blk*((int) (pow(solver->degree, solver->n_levels)+0.5))  /* size V */
//         //  +solver->ld*2*blk);
//
//         solver->bytes_outer = sizeof(double)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         //solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//         //  (solver->ld*solver->blk_solver*2      /* size B and size X */
//         //  +solver->ld*solver->blk_solver*((int) (pow(solver->degree,
//         // solver->n_levels)+0.5))  /* size V */
//         //  +solver->ld*2*blk);
//
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         //solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//         //  (solver->ld*solver->blk_solver          /* size B */
//         //  +solver->ld*solver->blk_solver          /* size X */
//         //  +solver->ld*solver->blk_solver*n_diags  /* size V */
//         //  +solver->ld*solver->blk_solver          /* tempf_X*/
//         //  +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//         //  +solver->ld                              /* size e_vector */
//         //  +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_SOLVER_OUTER_BLOCK_LEAST_SQUARES_HORNER_ESTIMATION:
//     {
//       /* add size of V */
//       /* size_V: 2^n_levels*/
//       MPF_Int n_diags = mpf_n_diags_get(solver->n_levels, solver->degree);
//       //MPF_Int blk = solver->blk_solver;
//
//       if (solver->data_type == MPF_REAL)
//       {
//         solver->bytes_outer = sizeof(double)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         //solver->bytes_outer = sizeof(double)*
//         //  (solver->ld*solver->blk_solver*2
//         //  +solver->ld*blk*num_diags
//         //  +solver->ld*2);
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         //solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//         //  (solver->ld*solver->blk_solver*2
//         //  +solver->ld*blk*n_diags
//         //  +solver->ld*2);
//
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_SOLVER_OUTER_GLOBAL_LEAST_SQUARES_HORNER_ESTIMATION:
//     {
//       MPF_Int n_diags = mpf_n_diags_get(solver->n_levels, solver->degree);
//       if (solver->data_type == MPF_REAL)
//       {
//         //solver->bytes_outer = sizeof(double)*
//         //  (solver->ld*solver->blk_solver*2
//         //  +solver->ld*blk_fA*num_diags
//         //  +solver->ld*2);
//
//         solver->bytes_outer = sizeof(double)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       }
//       else if (solver->data_type == MPF_COMPLEX)
//       {
//         //solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//         //  (solver->ld*solver->blk_solver*2
//         //  +solver->ld*blk_fA*num_diags
//         //  +solver->ld*2);
//
//         solver->bytes_outer = sizeof(MPF_ComplexDouble)*
//           (solver->ld*solver->blk_solver          /* size B */
//           +solver->ld*solver->blk_solver          /* size X */
//           +solver->ld*solver->blk_solver*n_diags  /* size V */
//           +solver->ld*solver->blk_solver          /* tempf_X*/
//           +solver->ld*solver->blk_solver*n_diags  /* tempf_V */
//           +solver->ld                              /* size e_vector */
//           +n_diags*solver->blk_solver*2);           /* size tempf_matrix (QR) */
//
//         solver->bytes_cuda_outer = 0;
//         solver->bytes_fA_data = sizeof(MPF_ComplexDouble)*solver->ld
//           *solver->blk_fA;
//       }
//       break;
//     }
//     case MPF_SOLVER_OUTER_BATCH_MATRIX:
//     {
//       /* add size of V */
//       /* size_V: 2^n_levels*/
//       solver->bytes_outer = sizeof(double)
//         *(solver->ld*solver->blk_solver*2);
//       solver->bytes_cuda_outer = 0;
//       solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       break;
//     }
//     case MPF_SOLVER_OUTER_BATCH_PTHREADS:
//     {
//       /* add size of V */
//       /* size_V: 2^n_levels*/
//       //solver->bytes_outer = sizeof(double)*(solver->ld
//       //  *solver->blk_solver*2*solver->n_threads_solver);
//       solver->bytes_outer = sizeof(double)*(solver->ld
//         *solver->blk_solver*2);
//       solver->bytes_cuda_outer = 0;
//       solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA;
//       break;
//     }
//     case MPF_SOLVER_OUTER_BATCH_OPENMPF_:
//     {
//       /* add size of V */
//       /* size_V: 2^n_levels*/
//       solver->bytes_outer = sizeof(double)*(solver->ld
//         *blk*2*solver->n_threads_solver);
//       solver->bytes_cuda_outer = 0;
//       solver->bytes_fA_data = sizeof(double)*solver->ld*solver->blk_fA
//         *solver->n_threads_pthreads;
//       break;
//     }
//     default:
//       break;
//   }
// }
