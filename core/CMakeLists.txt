file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

add_library(mpf
  SHARED
  types/mpf_bucket_array.c
  types/mpf_color_to_node_map.c
  types/mpf_context.c
  types/mpf_dense.c
  types/mpf_heap.c
  numops/mpf_linalg_op.c
  numops/mpf_scalar_op.c
  types/mpf_linked_list.c
  types/mpf_matrix
  types/mpf_sparse.c
  probe/mpf_blocking_batch.c
  probe/mpf_blocking_batch_coarse.c
  probe/mpf_blocking.c
  probe/mpf_color.c
  probe/mpf_compact_hierachy.c
  probe/mpf_contract_batch_blocking.c
  probe/mpf_contract_blocking.c
  probe/mpf_contract_row.c
  probe/mpf_contract_sample.c
  probe/mpf_generate_sampling_endpoints.c
  probe/mpf_get_max_blk_fA.c
  probe/mpf_multisample.c
  probe/mpf_multisample_old.c
  probe/mpf_pattern_multisample_merge.c
  probe/mpf_pattern_sample.c
  probe/mpf_probe.c
  probe/mpf_sample_contract_dynamic.c
  probe/mpf_sample_contract_dynamic_new.c
  probe/mpf_sparse_reorder_contract.c
  solve/mpf_solve.c
  solve/inner/krylov/mpf_blk_cg.c
  solve/inner/krylov/mpf_blk_gmres.c
  solve/inner/krylov/mpf_blk_lanczos.c
  solve/inner/krylov/mpf_cg.c
  solve/inner/krylov/mpf_cheb.c
  solve/inner/krylov/mpf_defl_get_mem_size.c
  solve/inner/krylov/mpf_defl_lanczos.c
  solve/inner/krylov/mpf_ev.c
  solve/inner/krylov/mpf_ev_defl_cg.c
  solve/inner/krylov/mpf_ev_defl_lanczos.c
  solve/inner/krylov/mpf_gbl_cg.c
  solve/inner/krylov/mpf_gbl_gmres.c
  solve/inner/krylov/mpf_gbl_lanczos.c
  solve/inner/krylov/mpf_gbl_pcg.c
  solve/inner/krylov/mpf_gmres.c
  solve/inner/krylov/mpf_krylov_alloc.c
  solve/inner/krylov/mpf_lanczos.c
  solve/inner/krylov/mpf_lanczos_cheb.c
  solve/inner/krylov/mpf_pcg.c
  solve/inner/krylov/mpf_seq_solve.c
  solve/inner/krylov/mpf_spbasis_blk_lanczos.c
  solve/inner/krylov/mpf_spbasis_gbl_lanczos.c
  solve/inner/krylov/mpf_spbasis_defl_cg.c
  solve/inner/krylov/mpf_gbl_lanczos.c
  solve/inner/krylov/mpf_spbasis_lanczos.c
  solve/inner/ls/mpf_ls_d_horner.c
  solve/inner/ls/mpf_ls_horner_init.c
  solve/inner/qr/mpf_qr.c
  solve/inner/precond/mp_precond.c
  solve/outer/mpf_batch_2pass.c
  solve/outer/mpf_batch_cheb.c
  solve/outer/mpf_batch_defl_solve.c
  solve/outer/mpf_batch_spai.c
  solve/outer/mpf_batch_spbasis.c
  solve/outer/mpf_rhs.c
  solve/outer/mpf_select_op.c
  solve/outer/mpf_solve_batch.c
  solve/outer/mpf_solve_get_mem.c
  solve/outer/mpf_sparse_reconstruct.c
  solve/outer/mpf_sparse_solve.c
  io/mpf_buffer_io.c
  io/mpf_printout.c
  io/mpf_mmio.c
  run/mpf.c
  types/mpf_args.c
  run/mpf_run.c
  benchmarks/mpf_timing.c
)

set_target_properties(mpf PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(mpf PUBLIC
  types
  run
  solve
  probe
  io
  blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_libraries(mpf
  -L${MKL_LIB} -Wl,--no-as-needed ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} -liomp5 -lpthread -lm -ldl
  -lgomp -shared
)

