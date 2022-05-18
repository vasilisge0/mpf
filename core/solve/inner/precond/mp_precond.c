#include "mpf.h"

/*----------------------------- preconditioning ------------------------------*/

/*============================================================================*/
/* Approximate inverse preconditioning using the diagonal blocks.             */
/*============================================================================*/

//void mpf_jacobi_precond_csr_init
//(
//  MPF_Sparse *A,
//  MPF_Sparse *M
//)
//{
//  int nz = 0;
//  for (MPF_Int i = 0; i < A->m; ++i)
//  {
//    M->mem.csr.rs[i] = i;
//    M->mem.csr.re[i] = M->mem.csr.rs[i];
//    MPF_Int j = A->mem.csr.rs[i];
//    while ((j < A->mem.csr.re[i]) && (A->mem.csr.cols[j] < i))
//    {
//      j += 1;
//    }
//
//    if ((j != A->mem.csr.re[i]) && (A->mem.csr.cols[j] == i))
//    {
//      ((double*)M->mem.csr.data)[i] = 1.0/((double*)A->mem.csr.data)[j];
//      M->mem.csr.cols[nz] = i;
//      nz += 1;
//    }
//
//    ((double*)M->mem.csr.data)[i] = 1.0;
//    M->mem.csr.re[i] = M->mem.csr.rs[i] + 1;
//  }
//}

void mpf_jacobi_precond_init
(
  MPF_ContextHandle context
)
{
  if (context->A.data_type == MPF_REAL)
  {
    context->solver.precond_alloc_function = mpf_sparse_csr_d_diag_alloc;
    context->solver.precond_free_function = mpf_sparse_precond_free;
    context->solver.precond_generate_function = mpf_jacobi_precond_generate;
    context->solver.precond_apply_function = mpf_sparse_d_mm_wrapper;
    context->solver.precond_update = MPF_PRECOND_STATIC;
    context->solver.precond_type = MPF_PRECOND_JACOBI;
  }
}

void mpf_jacobi_precond_generate
(
  MPF_Solver* solver,
  MPF_Sparse* A
)
{
  MPF_Int blk = solver->batch;
  solver->M.m = solver->ld;
  solver->M.n = solver->ld;
  solver->M.nz = solver->ld;
  solver->M.data_type = A->data_type;
  solver->M.matrix_type = A->matrix_type;
  solver->M.descr = A->descr;

  for (MPF_Int i = 0; i < solver->ld; ++i)
  {
    solver->M.mem.csr.rs[i] = i;
  }

  for (MPF_Int i = 0; i < solver->ld; ++i)
  {
    solver->M.mem.csr.re[i] = i+1;
  }

  for (MPF_Int i = 0; i < solver->ld; ++i)
  {
    solver->M.mem.csr.cols[i] = i;
  }

  if (A->data_type == MPF_REAL)
  {
    for (MPF_Int i = 0; i < solver->ld; ++i)
    {
      MPF_Int j = A->mem.csr.rs[i];
      while ((j < A->mem.csr.re[i]) && (A->mem.csr.cols[j] < i))
      {
        j += 1;
      }

      if ((A->mem.csr.cols[j] == i) && (j < A->mem.csr.re[i]))
      {
        ((double*)solver->M.mem.csr.data)[i] = 1.0/((double*)A->mem.csr.data)[j];
      }
    }

    mkl_sparse_d_create_csr(&solver->M.handle, INDEXING, solver->M.m,
      solver->M.m, solver->M.mem.csr.rs, solver->M.mem.csr.re,
      solver->M.mem.csr.cols, (double*)solver->M.mem.csr.data);
  }
  else if (A->data_type == MPF_COMPLEX)
  {
    MPF_ComplexDouble ONE = mpf_scalar_z_init(1.0, 0.0);
    for (MPF_Int i = 0; i < solver->ld; ++i)
    {
      MPF_Int j = A->mem.csr.rs[i];
      while ((j < A->mem.csr.re[i]) && (A->mem.csr.cols[j] < i))
      {
        j += 1;
      }

      if ((A->mem.csr.cols[j] == i) && (j < A->mem.csr.re[i]))
      {
        ((MPF_ComplexDouble*)solver->M.mem.csr.data)[i] =
          mpf_scalar_z_divide(ONE, ((MPF_ComplexDouble*)A->mem.csr.data)[j]);
      }
    }

    mkl_sparse_z_create_csr(&solver->M.handle, INDEXING, solver->M.m,
      solver->M.m, solver->M.mem.csr.rs, solver->M.mem.csr.re,
      solver->M.mem.csr.cols, (MPF_ComplexDouble*)solver->M.mem.csr.data);
  }

}

void mpf_precond_csr_create
(
  MPF_Solver *solver,
  MPF_Sparse *M
)
{
  solver->M.m = M->m;
  solver->M.n = M->n;
  solver->M.descr = M->descr;
  solver->M.index = M->index;
  solver->M.data_type = M->data_type;
  solver->M.matrix_type = M->matrix_type;

  if (M->data_type == MPF_REAL)
  {
    mkl_sparse_d_export_csr(solver->M.handle, &solver->M.index, &solver->M.m,
      &solver->M.n, &solver->M.mem.csr.rs, &solver->M.mem.csr.re,
      &solver->M.mem.csr.cols, (double**)&solver->M.mem.csr.data);
  }
  else if (M->data_type == MPF_COMPLEX)
  {
    mkl_sparse_z_export_csr(solver->M.handle, &solver->M.index, &solver->M.m,
      &solver->M.n, &solver->M.mem.csr.rs, &solver->M.mem.csr.re,
      &solver->M.mem.csr.cols, (MPF_ComplexDouble**)&solver->M.mem.csr.data);
  }
}

void mpf_precond_csr_destroy
(
  MPF_Solver *solver
)
{
  if (solver->M.handle != NULL)
  {
    mkl_sparse_destroy(solver->M.handle);
  }
}

void mpf_precond_dsy_blkdiag_alloc
(
  MPF_Solver *solver
)
{
  solver->M.m = solver->ld;
  solver->M.n = solver->ld;
  solver->M.nz = solver->ld * solver->blk_fA * solver->blk_fA;
  mpf_sparse_d_csr_alloc(&solver->M);
}

void mpf_precond_dsy_blkdiag_init
(
  MPF_Solver *solver
)
{
  for (MPF_Int i = 0; i < solver->ld; ++i)
  {
    solver->M.mem.csr.rs[i] = solver->blk_fA*i;
    solver->M.mem.csr.re[i] = solver->M.mem.csr.rs[i];
  }
}

void mpf_precond_dsy_spai_alloc
(
  MPF_Solver *solver
)
{
  solver->M.m = solver->ld;
  solver->M.n = solver->ld;
  solver->M.nz = solver->ld;
  mpf_sparse_d_csr_alloc(&solver->M);
}

void mpf_precond_dsy_spai_init
(
  MPF_Solver *solver
)
{
  switch (solver->precond_type)
  {
    case MPF_PRECOND_EYE:
      mpf_sparse_d_eye(&solver->M);
      break;
  }

}

void mpf_precond_dec_re
(
  MPF_Probe *probe,
  MPF_Solver *solver
)
{
  MPF_Int cs = solver->current_rhs;
  MPF_Int ce = (solver->current_rhs+solver->batch+solver->max_blk_fA-1)/solver->max_blk_fA;
  MPF_BucketArray *c2n_map = &solver->color_to_node_map;

  for (MPF_Int i = cs; i < ce; ++i)
  {
    for (MPF_Int j = c2n_map->bins_start[i]; j < c2n_map->bins_start[i]; j = c2n_map->next[j])
    {
      for (MPF_Int k = 0; k < solver->max_blk_fA; ++k)
      {
        solver->M.mem.csr.re[solver->max_blk_fA*j+k] = solver->M.mem.csr.rs[solver->max_blk_fA*j+k];
      }
    }
  }
}

void mpf_precond_dsy_spai_update
(
  MPF_Probe* probe,
  MPF_Solver* solver
)
{
  mpf_precond_dec_re(probe, solver);
  mpf_spai_d_reconstruct(probe, solver);
}

void mpf_sparse_csr_d_diag_alloc
(
  MPF_Solver *solver
)
{
  solver->M.m = solver->ld;
  solver->M.n = solver->ld;
  solver->M.nz = solver->ld;
  solver->M.data_type = MPF_REAL;
  mpf_sparse_csr_alloc(&solver->M);
}

void mpf_sparse_precond_free
(
  MPF_Solver* solver
)
{
  mpf_sparse_csr_free(&solver->M);
}

void mpf_d_precond_init
(
  MPF_Solver* solver
)
{
  switch(solver->precond_type)
  {
    case MPF_PRECOND_JACOBI:
      solver->precond_alloc_function = mpf_sparse_csr_d_diag_alloc;
      solver->precond_free_function = mpf_sparse_precond_free;
      solver->precond_generate_function = mpf_jacobi_precond_generate;
      solver->precond_apply_function = mpf_sparse_d_mm_wrapper;
      solver->precond_update = MPF_PRECOND_STATIC;
      break;
  }
}
