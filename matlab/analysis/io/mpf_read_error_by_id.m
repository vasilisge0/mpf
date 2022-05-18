function [] = mpf_read_error_by_id(S, range_id, j)

  fid = fopen(S.filename.meta{1});

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

  line = fgetl(fid);
  solver_frame = strsplit(line);

  line = strsplit(fgetl(fid));
  solver_outer = line(1);

  line = fgetl(fid);
  solver_batch = sscanf(line, '%d');

  line = fgetl(fid);
  solver_outer_nthreads = sscanf(line, '%d');

  line = fgetl(fid);
  solver_inner_nthreads = sscanf(line, '%d');

  line = fgetl(fid);
  precond = strsplit(line);

  line = fgetl(fid);
  defl = strsplit(line);

  line = fgetl(fid);
  solver_inner = strsplit(line);

  line = fgetl(fid);
  solver_inner_tol = sscanf(line, '%f');

  line = fgetl(fid);
  solver_inner_iters = sscanf(line, '%d');

  line = fgetl(fid);
  nsamples = sscanf(line, '%d');

  fclose(fid);

  format_f = @(x)regexprep(num2str(x, '%1.0e'), 'e\-0*', 'e-');

  for i = 1:length(n_levels)
    % assembly of file_meta
    file_approx = strcat(...
        S.filename.data_prefix{range_id, 1}, ...
        'blkprobe_', ...
        string(blk_fA(S.index.blk_fA_id)), '_', ...
        string(output_type{S.index.output_type_id}), '_', ...
        string(stride(S.index.stride_id)'), '_', ...
        string(n_levels(i)), '_', ...
        string(solver_frame{S.index.solver_frame_id}), '_', ...
        string(solver_batch(S.index.solver_inner_id)), '_', ...
        string(solver_outer_nthreads(S.index.solver_outer_nthreads_id))', '_', ...
        string(solver_inner_nthreads(S.index.solver_inner_nthreads_id)), '_', ...
        precond, '_', ...
        defl, '_', ...
        solver_inner(S.index.solver_inner_id), '_', ...
        format_f(solver_inner_tol(S.index.solver_inner_tol_id)), '_', ...
        string(solver_inner_iters(S.index.solver_inner_iters_id)), '_', ...
        string(0), ...
        '.mtx');

      S.data.Ai_approx{i, j} = mmread(strcat(S.filename.path_approx, file_approx));
  end

  S.index.stride_id = S.index.stride_id + 1;

end
