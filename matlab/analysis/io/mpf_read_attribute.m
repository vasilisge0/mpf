function [] = mpf_read_log_by_attr_range(filename, path_meta, offset_s, offset_e, attr)

  fid = fopen(filename);

  line = fgetl(fid);
  filename_A = line;

  line = fgetl(fid);
  data_type_A = line;

  line = fgetl(fid);
  format_A = line;

  line = fgetl(fid);
  matrix_type_A = line;

  line = fgetl(fid);
  blk_fA = sscanf(line, '%d')

  line = fgetl(fid);
  output_type = strsplit(line);

  line = fgetl(fid);
  probe_type = line;

  line = fgetl(fid);
  stride = sscanf(line, '%d');

  line = fgetl(fid);
  probe_nlevels = sscanf(line, '%d');

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
  solver_inner = strsplit(line);

  line = fgetl(fid);
  solver_inner_tol = sscanf(line, '%f');

  line = fgetl(fid);
  precond = strsplit(line);

  line = fgetl(fid);
  defl = strsplit(line);

  line = fgetl(fid);
  solver_inner_iters = sscanf(line, '%d');
 
  fclose(fid);

  format_f = @(x)regexprep(num2str(x, '%1.0e'), 'e\-0*', 'e-');
  S = MPF_Log();

  if strcmp(attr, 'blk_fA')
    blk_fA = blk_fA(offset_s:offset_e);
    for i = 1:offset_e-offset_s+1
      file_meta = strcat('run_', filename_A, '_', ...
          data_type_A, '_', format_A, '_', matrix_type_A, '_', string(blk_fA(i+offset_s-1).'), '_', string(output_type{1}), '_', 'blocking', '_', string(stride(1).'), '_', ...
          string(probe_nlevels(1)), '_', string(solver_frame{1}), '_', string(solver_outer{1}), '_', string(solver_batch(1)), '_', ...
          string(solver_outer_nthreads(1))', '_', string(solver_inner_nthreads(1)), '_', ...
          solver_inner(1), '_', format_f(solver_inner_tol(1)), '_', string(solver_inner_iters(1)), '_meta');
      mpf_read_meta(strcat(path_meta, file_meta), i, S)
    end
%  %elseif strcmp(attr, 'output_type')
%
  %elseif strcmp(attr, 'stride')
  elseif strcmp(attr, 'probe_nlevels')   
    stride = stride(offset_s:offset_e);
    for i = 1:offset_e-offset_s+1

      % assembly of file_meta
      file_meta = strcat('run_', filename_A, '_', ...
          data_type_A, '_', format_A, '_', matrix_type_A, '_', string(blk_fA(1).'), '_', string(output_type{1}), '_', 'blocking', '_', string(stride(i+offset_s-1).'), '_', ...
          string(probe_nlevels(1)), '_', string(solver_frame{1}), '_', string(solver_outer{1}), '_', string(solver_batch(1)), '_', ...
          string(solver_outer_nthreads(1))', '_', string(solver_inner_nthreads(1)), '_', ...
          '_', precond, '_', defl, solver_inner(1), '_', format_f(solver_inner_tol(1)), '_', string(solver_inner_iters(1)), '_meta');

      % access data
      mpf_read_meta(strcat(path_meta, file_meta), i, S)
    end
  %elseif strcmp(attr, 'solver_frame')
%
%  %elseif strcmp(attr, 'solver_outer')
%
%  %elseif strcmp(attr, 'solver_batch')
%
%  %elseif strcmp(attr, 'solver_outer_nthreads')
%
%  elseif strcmp(attr, 'solver_inner_nthreads')

%  elseif strcmp(attr, 'solver_inner_tol')
%
%  %elseif strcmp(attr, 'solver_inner_iters')
%
%  %end
%
%  %for i = offset_s:offset_e
%  %  
  end
end

