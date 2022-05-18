function [] = mpf_read_log_by_range(S, query_id)

  range = S.query.range{query_id, 1};

  if strcmp(S.query.type, 'blk_fA')
    S.index.blk_fA = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.blk_fA = S.index.blk_fA + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'output_type')
    S.index.output_type = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.output_type = S.index.output_type + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'probe_type')
    S.index.probe_type_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.probe_type_id = S.index.probe_type_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'stride')
    S.index.stride_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.stride_id = S.index.stride_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_frame')
    S.index.solver_frame_id = range(1)
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id);
      S.index.solver_frame_id = S.index.solver_frame_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_outer')
    S.index.solver_outer_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_outer_id = S.index.solver_outer_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_batch')
    S.index.solver_batch_id = range(1)
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_batch_id = S.index.solver_batch_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_outer_nthreads')
    S.index.solver_outer_nthreads_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_outer_nthreads_id = S.index.solver_outer_nthreads_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_inner_nthreads')
    S.index.solver_inner_nthreads_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_inner_nthreads_id = S.index.solver_inner_nthreads_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_inner')
    S.index.solver_inner_id = range(1)
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_inner_id = S.index.solver_inner_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_inner_tol')
    S.index.solver_inner_tol_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_inner_tol_id = S.index.solver_inner_tol_id + 1;
    end

  elseif strcmp(S.query.type{query_id, 1}, 'solver_inner_iters')
    S.index.solver_inner_iters_id = range(1);
    for i = range(1):range(2)
      mpf_read_log_by_id(S, query_id, i);
      S.index.solver_inner_iters_id = S.index.solver_inner_iters_id + 1;
    end
  end

  % solver runtime
  for i = 1:S.n_logs
    i
    mpf_get_total_solver_runtime(S, i);
  end

  % probe runtime
  for i = 1:S.n_logs
    mpf_get_total_probe_runtime(S, i);
  end

end
