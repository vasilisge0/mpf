function [] = mpf_read_log_by_id(S, range_id, log_id)

  fid = fopen(S.filename.meta{range_id, 1});
  line = fgetl(fid);
  filename_A = line;

  line = fgetl(fid);
  data_type_A = line;

  line = fgetl(fid);
  format_A = line;

  line = fgetl(fid);
  matrix_type_A = line;

  line = fgetl(fid);
  blk_fA = sscanf(line, '%d');

  line = fgetl(fid);
  output_type = strsplit(line);

  line = fgetl(fid);
  probe_type = line;

  line = fgetl(fid);
  stride = sscanf(line, '%d');

  line = fgetl(fid);
  n_levels = sscanf(line, '%d');

  line = strsplit(fgetl(fid));
  solver_framework = line;

  line = strsplit(fgetl(fid));
  solver_outer_type = line{:};

  line = fgetl(fid);
  solver_batch = sscanf(line, '%d');

  line = fgetl(fid);
  solver_outer_nthreads = sscanf(line, '%d');

  line = fgetl(fid);
  solver_inner_nthreads = sscanf(line, '%d');

  line = strsplit(fgetl(fid));
  solver_precond = line;

  line = strsplit(fgetl(fid));
  solver_defl = line;

  line = strsplit(fgetl(fid));
  solver_inner_type = line;

  line = fgetl(fid);
  solver_tolerance = sscanf(line, '%f');

  line = fgetl(fid);
  solver_iterations = sscanf(line, '%d');

  line = fgetl(fid);
  nsamples = sscanf(line, '%d');

  fclose(fid);

  format_f = @(x)regexprep(num2str(x, '%1.0e'), 'e\-0*', 'e-');

  S.n_logs = S.n_logs + 1;

  for i = 1:length(n_levels)
    for j = 1:nsamples

      % assembly of file_meta
      S.filename.file_log{i, S.n_logs} = strcat(...
          S.filename.data_prefix{range_id, 1}, ...
          'blkprobe_', ...
          string(blk_fA(S.index.blk_fA_id))', '_', ...
          string(output_type(S.index.solver_outer_id)), '_', ...
          string(stride(S.index.stride_id)'), '_', ...
          string(n_levels(i))', '_', ...
          string(solver_framework{S.index.solver_frame_id}), '_', ...
          string(solver_batch(S.index.solver_batch_id)), '_', ...
          string(solver_outer_nthreads(S.index.solver_outer_nthreads_id))', '_', ...
          string(solver_inner_nthreads(S.index.solver_inner_nthreads_id)), '_', ...
          solver_precond{S.index.solver_precond_id}, '_', ...
          solver_defl{S.index.solver_defl_id}, '_', ...
          solver_inner_type{S.index.solver_inner_id}, '_', ...
          format_f(solver_tolerance(S.index.solver_inner_tol_id)), '_', ...
          string(solver_iterations(S.index.solver_inner_iters_id)), '_', ...
          string(j-1), ...
          '_log')

      % access performance meta
      mpf_read_meta(strcat(S.filename.path_log, '/log/', S.filename.file_log{i, S.n_logs}), i, S);
    end
  end
end
