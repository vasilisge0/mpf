#include "mpf.h"

/*===================================================================

I/O:
----
1. io_filename_A
2. io_filename_output_meta
3. io_filename_output_diag
4. io_code_A
5. io_code_B

metadata:
---------
6. data_type
7. solver_inner_type
8. solver_outer_type
9. probing_type
10. m_A
11. n_A
12 .nz_A
    (A_descr ?)
    (A_descr_gpu ?)
13. A.format
14. memory_type
15. memory_probing_type
16. memory_increment
17. ld_B
18. m_B
19. n_max_B
20. layout_B

memory:
-------
21. solver_inner_num_bytes_memory_cpu;
22. solver_inner_num_bytes_memory_gpu;
23. bytes_outer;
24. bytes_outer;

profiling:
----------
25. runtime_total;
26. runtime_probing;
27. runtime.inner_outer;

probing:
--------
28. degree;
29. m_P;
30. nz_P;
31. n_colors;
32. n_levels;

numerical:
----------
(1. outer_iterations)
33. solver_outer_residual_array
34. solver_outer_iterations_array
35. solver_outer_restarts_array

(2. inner_iterations)

===================================================================*/

void mpf_printout
(
  MPF_Context *context
)
{
  printf("\n");
  printf("======================== MPF_PRINTOUT ======================== \n");
  printf("\n");
  printf("I/O\n");
  printf("---\n");
  printf("          io_filename_A: %s\n", context->args.filename_A);
  printf("io_filename_output_meta: %s\n", context->args.filename_meta);
  printf("io_filename_output_diag: %s\n", context->args.filename_fA);
  printf("              io_code_A: %s\n", context->typecode_A);

  if (context->data_type == MPF_REAL)
  {
    printf("              data_type: MPF_REAL\n");
  }
  else if (context->data_type == MPF_COMPLEX)
  {
    printf("              data_type: MPF_COMPLEX\n");
  }
  if (context->A.format == MPF_SPARSE_COO)
  {
    printf("               A.format: MPF_SPARSE_COO\n");
  }
  else if (context->A.format == MPF_SPARSE_CSR)
  {
    printf("               A.format: MPF_SPARSE_CSR\n");
  }

  /* metadata */
  printf("\nmetadata\n");
  printf("--------\n");
  printf("                    m_A: %d\n", context->A.m);
  printf("                    n_A: %d\n", context->A.n);
  printf("                   nz_A: %d\n", context->A.nz);
  printf("                n_max_B: %d\n", context->solver.n_max_B);
  printf("            blk_probing: %d\n", context->probe.stride);
  printf("             blk_max_fA: %d\n", context->solver.max_blk_fA);
  printf("               n_levels: %d\n", context->probe.n_levels);
  printf("               n_colors: %d\n", context->probe.n_colors);

  /* probing type */
  switch(context->probe.type)
  {
      case MPF_PROBE_UNDEFINED:
        printf("           probing_type: MPF_PROBING_UNDEFINED\n");
        break;
      case MPF_PROBE_BLOCKING:
        printf("           probing_type: MPF_PROBING_BLOKING\n");
        break;
      case MPF_PROBE_SAMPLING:
        printf("           probing_type: MPF_PROBING_MULTILEVEL_SAMPF_LING\n");
        break;
      case MPF_PROBE_PATH_SAMPLING:
        printf("           probing_type: MPF_PROBING_MULTIPATH_SAMPF_LING\n");
        break;
      case MPF_PROBE_AVG_PATH_SAMPLING:
        printf("           probing_type: MPF_PROBING_AVERAGE_MULTIPATH_SAMPF_LING\n");
        break;
      default:
        break;
  }

  /* outer solver */
  switch(context->solver.outer_type)
  {
      case MPF_SOLVER_BATCH:
          printf("      solver_outer_type: MPF_SOLVER_OUTER_BATCH\n");
          break;
      //case MPF_BATCH:
      //    printf("      solver_outer_type: MPF_SOLVER_OUTER_BATCH\n");
      //    break;
      case MPF_SOLVER_BATCH_LS:
          printf("      solver_outer_type: MPF_SOLVER_OUTER_LEAST_SQUARES_HORNER_ESTIMATION\n");
          break;
      case MPF_SOLVER_BATCH_UNDEFINED:
          printf("      solver_outer_type: MPF_SOLVER_OUTER_UNDEFINED\n");
          break;
      default:
          break;
  }

  /* inner solver */
  switch(context->solver.inner_type)
  {
    case MPF_SOLVER_DGE_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_DGE_GMRES\n");
      break;
    case MPF_SOLVER_DSY_GBL_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_DSY_GBL_GMRES\n");
      break;
    case MPF_SOLVER_DGE_BLK_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_DGE_BLK_GMRES\n");
      break;
    case MPF_SOLVER_DGE_GBL_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_DGE_GBL_GMRES\n");
      break;
    case MPF_SOLVER_DSY_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_DSY_LANCZOS\n");
      break;
    case MPF_SOLVER_DSY_BLK_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_DSY_BLK_LANCZOS\n");
      break;
    case MPF_SOLVER_DSY_GBL_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_DSY_GBL_LANCZOS\n");
      break;
    case MPF_SOLVER_CG0:
      printf("      solver_inner_type: MPF_SOLVER_CG0\n");
      break;
    case MPF_SOLVER_DSY_CG:
      printf("      solver_inner_type: MPF_SOLVER_DSY_CG\n");
      break;
    case MPF_SOLVER_DSY_BLK_CG:
      printf("      solver_inner_type: MPF_SOLVER_DSY_BLK_CG\n");
      break;
    case MPF_SOLVER_DSY_GBL_CG:
      printf("      solver_inner_type: MPF_SOLVER_DSY_GBL_CG\n");
      break;
    case MPF_SOLVER_DSY_GBL_PCG:
      printf("      solver_inner_type: MPF_SOLVER_DSY_GBL_PCG\n");
      break;
    case MPF_SOLVER_ZSY_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_GMRES\n");
      break;
    case MPF_SOLVER_ZSY_BLK_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_BLK_GMRES\n");
      break;
    case MPF_SOLVER_ZSY_GBL_GMRES:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_GBL_GMRES\n");
      break;
    case MPF_SOLVER_ZSY_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_LANCZOS\n");
      break;
    case MPF_SOLVER_ZSY_BLK_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_BLK_LANCZOS\n");
      break;
    case MPF_SOLVER_ZSY_GBL_LANCZOS:
      printf("      solver_inner_type: MPF_SOLVER_ZSY_GBL_LANCZOS\n");
      break;
    default:
      break;
  }
  printf("\nthreading\n");
  printf("----------\n");
  if ((context->solver.n_threads_pthreads == 0) && (context->solver.n_threads_omp == 0))
  {
    printf("        n_outer_threads: 1\n");
    printf("        n_inner_threads: 1\n");
  }
  else
  {
    printf("        n_outer_threads: %d\n", context->solver.n_threads_pthreads);
    printf("        n_inner_threads: %d\n", context->solver.n_threads_omp);
  }

  double runtime_other = 0.0;
  if (context->solver.runtime_other < 1e-15)
  {
    runtime_other = 0.0;
  }
  else
  {
    runtime_other = context->solver.runtime_other;
  }

  printf("\nruntime \n");
  printf("-------\n");
  printf("            Epased time: %f seconds \n", context->probe.runtime_total + context->solver.runtime_total);
  printf("----------------------------------------------------------------------------------\n");
  printf("| probing                   | solver                   | other                   |\n");
  printf("|--------------------------------------------------------------------------------|\n");
  printf("|            total: %1.1E | total: %1.1E / %1.1E |          total: %1.1E |\n",
    context->probe.runtime_total,
    context->solver.runtime_total,
    context->solver.runtime_total - runtime_other,
    runtime_other
  );
  printf("----------------------------------------------------------------------------------\n");
  printf("|         contract: %1.1E |      preprocess: %1.1E | create_context: %1.1E |\n", context->probe.runtime_contract, context->solver.runtime_pre_process, context->runtime_create);
  printf("|           expand: %1.1E |           alloc: %1.1E |                         |\n", context->probe.runtime_expand, context->solver.runtime_alloc);
  printf("|            color: %1.1E |    generate_rhs: %1.1E |                         |\n", context->probe.runtime_color, context->solver.runtime_generate_rhs);
  printf("|            other: %1.1E |           inner: %1.1E |                         |\n", context->probe.runtime_other, context->solver.runtime_inner);
  printf("|                           |     reconstruct: %1.1E |                         |\n", context->solver.runtime_reconstruct);
  printf("|                           |     postprocess: %1.1E |                         |\n", context->solver.runtime_post_process);
  printf("----------------------------------------------------------------------------------\n");

  if (context->solver.recon_target == MPF_DIAG_FA)
  {
    MPF_Dense* fA = &context->diag_fA;
    mpf_matrix_d_announce((double*)fA->data, 5, context->solver.blk_fA, context->A.m, "fA");
  }
}
