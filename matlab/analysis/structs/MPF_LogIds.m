classdef MPF_LogIds < handle
  properties

    blk_fA_id;
    output_type_id;
    probe_type_id;
    stride_id;
    solver_frame_id;
    solver_outer_id;
    solver_batch_id;
    solver_outer_nthreads_id;
    solver_inner_nthreads_id;
    solver_inner_id;
    solver_inner_tol_id;
    solver_inner_iters_id;
    solver_precond_id;
    solver_defl_id;
  end

  methods
    %function T = MPF_LogIds(...
    %  blk_fA_id, ...
    %  output_type_id, ...
    %  probe_type_id, ...
    %  stride_id, ...
    %  solver_frame_id, ...
    %  solver_outer_id, ...
    %  solver_batch_id, ...
    %  solver_outer_nthreads_id, ...
    %  solver_inner_nthreads_id, ...
    %  solver_inner_id, ...
    %  solver_inner_tol_id, ...
    %  solver_inner_iters_id)

    %  T.blk_fA_id = blk_fA_id;
    %  T.output_type_id = output_type_id;
    %  T.probe_type_id = probe_type_id;
    %  T.stride_id = stride_id;
    %  T.solver_frame_id = solver_frame_id;
    %  T.solver_outer_id = solver_outer_id;
    %  T.solver_batch_id = solver_batch_id;
    %  T.solver_outer_nthreads_id = solver_outer_nthreads_id;
    %  T.solver_inner_nthreads_id = solver_inner_nthreads_id;
    %  T.solver_inner_id = solver_inner_id;
    %  T.solver_inner_tol_id = solver_inner_tol_id;
    %  T.solver_inner_iters_id = solver_inner_iters_id;
    %end

    function T = MPF_LogIds()
      T.blk_fA_id = 1;
      T.output_type_id = 1;
      T.probe_type_id = 1;
      T.stride_id = 1;
      T.solver_frame_id = 1;
      T.solver_outer_id = 1;
      T.solver_batch_id = 1;
      T.solver_outer_nthreads_id = 1;
      T.solver_inner_nthreads_id = 1;
      T.solver_inner_id = 1;
      T.solver_inner_tol_id = 1;
      T.solver_inner_iters_id = 1;
      T.solver_defl_id = 1;
      T.solver_precond_id = 1;
    end

  end

end
