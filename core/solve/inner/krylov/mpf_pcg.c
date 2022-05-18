#include "mpf.h"

void mpf_dsy_pcg
(
  MPF_Solver *solver,
  MPF_Sparse *A,
  MPF_Dense *b_dense,
  MPF_Dense *x_dense
)
{
  /* solver context */
  double norm_b = 0.0;
  double norm_r = 0.0;
  double alpha = 0.0;
  double beta  = 0.0;
  double trace_r_old = 0.0;
  double trace_r_new = 0.0;
  double gamma = 0.0;
  MPF_Int m_B = solver->ld;
  double *swap = NULL;
  MPF_Layout layout = MPF_COL_MAJOR;
  MPF_LayoutSparse sparse_layout;

  double *b = (double*)b_dense->data;
  double *x = (double*)x_dense->data;

  /* memory cpu */
  double *r_new = (double*)solver->inner_mem;
  double *r_old = &r_new[m_B];
  double *p = &r_old[m_B];

  /* added temporary vectors*/
  double *z = &p[m_B];
  double *z_new = &z[m_B];

  /* first iteration */
  mpf_convert_layout_to_sparse(layout, &sparse_layout);
  memcpy(r_old, b, (sizeof *r_old)*m_B);

  mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, A->handle,
    A->descr, x, 1.0, r_old);

  solver->precond_apply_function(solver, r_old, p); // dont know if this is correct
  norm_r = mpf_dnrm2(m_B, p, 1);
  norm_b = norm_r;

  printf("solver->ld: %d\n", solver->ld);
  printf("r_old[0]: %1.1E, p[0]: %1.1E\n", r_old[0], p[0]);
  printf("r_old[1]: %1.1E, p[1]: %1.1E\n", r_old[1], p[1]);
  printf(" norm_r: %1.1E\n", norm_r);

  MPF_Int i = 0;
  while ((i < solver->iterations) && (norm_r/norm_b > solver->tolerance))
  {
    mpf_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A->handle, A->descr, p,
      0.0, r_new);

    trace_r_old = mpf_ddot(m_B, r_old, 1, z, 1);
    gamma = mpf_ddot(m_B, r_new, 1, p, 1);
    alpha = trace_r_old/gamma;

    /* updates x and residual */
    mpf_daxpy(m_B, alpha, p, 1, x, 1);
    mpf_dscal(m_B, -alpha, r_new, 1);
    mpf_daxpy(m_B, 1.0, r_old, 1, r_new, 1);

    /* applies preconditioning step */
    solver->precond_apply_function(solver, r_new, z_new);

    /* computes beta */
    trace_r_new = mpf_ddot(m_B, r_new, 1, z_new, 1);
    beta = trace_r_new/mpf_ddot(m_B, r_old, 1, z, 1);

    /* update direction vector */
    mpf_dscal(m_B, beta, p, 1);
    mpf_daxpy(m_B, 1.0, z_new, 1, p, 1);

    norm_r = mpf_dnrm2(m_B, r_new, 1);
    swap = r_old;
    r_old = r_new;
    r_new = swap;
    i += 1;

    //#if MPF_PRINTOUT_SOLVER
      printf("relative residual: %1.4E\n", norm_r / norm_b);
    //#endif
  }

  r_new = NULL;
  r_old = NULL;
  swap = NULL;
}

void mpf_pcg_init
(
  MPF_Solver *solver,
  double tolerance,
  MPF_Int iterations,
  char *precond,
  char *filename
)
{
  solver->tolerance = tolerance;
  solver->iterations = iterations;
  solver->restarts = 0;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_type = MPF_SOLVER_DSY_CG;
    solver->inner_function = &mpf_dsy_pcg;
    solver->device = MPF_DEVICE_CPU;

    /* sets preconditioning wrapper functions */
    if (strcmp(precond, "spai") == 0)
    {
      solver->precond_type = MPF_PRECOND_SPAI;

      solver->precond_apply_function = &mpf_sparse_d_mm_wrapper;
    }
    else if (strcmp(precond, "jacobi") == 0)
    {
      solver->precond_type = MPF_PRECOND_JACOBI;
      solver->precond_alloc_function = mpf_sparse_csr_d_diag_alloc;
      solver->precond_apply_function = mpf_sparse_d_mm_wrapper;
      solver->precond_generate_function = mpf_jacobi_precond_generate;
      solver->precond_free_function = mpf_sparse_precond_free;
    }
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_SYMMETRIC))
  {
    solver->inner_type = MPF_SOLVER_ZSY_CG;
    solver->inner_function = &mpf_zsy_cg;
    solver->device = MPF_DEVICE_CPU;
  }
  else if ((solver->data_type == MPF_COMPLEX) &&
           (solver->matrix_type == MPF_MATRIX_HERMITIAN))
  {
    solver->inner_type = MPF_SOLVER_ZHE_CG;
    solver->inner_function = &mpf_zhe_cg;
    solver->device = MPF_DEVICE_CPU;
  }

  mpf_pcg_get_mem_size(solver);
}

void mpf_pcg_get_mem_size
(
  MPF_Solver *solver
)
{
  MPF_Int n = solver->ld;

  if (solver->data_type == MPF_REAL)
  {
    solver->inner_bytes = sizeof(double)*
        (n*2  /* size_B and size_X */
        +n    /* size_residual_old */
        +n    /* size_residual_new */
        +n    /* size_direction */
        +n    /* size_M */
        +n    /* size_T */
        +n);  /* size_direction_new */
  }
  else if (solver->data_type == MPF_COMPLEX)
  {
    solver->inner_bytes = sizeof(MPF_ComplexDouble)*
        (n*2  /* size_B and size_X */
        +n    /* size_residual_old */
        +n    /* size_residual_new */
        +n    /* size_direction */
        +n);  /* size_direction_new */
  }
  else if (solver->data_type == MPF_COMPLEX_32)
  {
    solver->inner_bytes = sizeof(MPF_Complex)*
        (n*2  /* size_B and size_X */
        +n    /* size_residual_old */
        +n    /* size_residual_new */
        +n    /* size_direction */
        +n);  /* size_direction_new */
  }
}
