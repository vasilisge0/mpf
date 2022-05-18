function [] = mpf_read_meta(filename, i, S)

  fid = fopen(filename);

  line = fgetl(fid);
  S.filename.caller{1, S.n_logs} = sscanf(line, '%s');

  line = fgetl(fid);
  S.filename.A{1, S.n_logs} = sscanf(line, '%s');

  line = fgetl(fid);
  S.filename.output{1, S.n_logs} = sscanf(line, '%s');

  line = fgetl(fid);
  %S.filename.meta{1, S.n_logs} = sscanf(line, '%s');

  line = sscanf(fgetl(fid), '%d');
  S.probe.expand_degree(i, S.n_logs) = line(1);
  S.probe.type(i, S.n_logs) = line(2);

  line = sscanf(fgetl(fid), '%d');
  S.m_A(i, S.n_logs) = line(1);
  S.n_A(i, S.n_logs) = line(2);
  S.nz_A(i, S.n_logs) = line(3);

  line = sscanf(fgetl(fid), '%d');
  S.probe.m_P(i, S.n_logs) = line(1);
  S.probe.n_P(i, S.n_logs) = line(2);
  S.probe.nz_P(i, S.n_logs) = line(3);

  line = sscanf(fgetl(fid), '%d');
  S.probe.n_colors(i, S.n_logs) = line(1);
  S.probe.n_levels(i, S.n_logs) = line(2);

  line = sscanf(fgetl(fid), '%d');
  S.solver.blk_fA(i, S.n_logs) = line(1);
  S.probe.stride(i, S.n_logs) = line(2);
  S.probe.max_blk_fA(S.n_logs, i) = line(3);
  S.solver.batch(i, S.n_logs) = line(4);
  S.probe.n_max_B(i, S.n_logs) = line(5);

  line = sscanf(fgetl(fid), '%d');
  S.solver.framework{1, S.n_logs} = line(1);

  line = sscanf(fgetl(fid), '%d');
  S.solver.outer_type(i, S.n_logs) = line(1);
  S.solver.inner_type(i, S.n_logs) = line(2);

  line = sscanf(fgetl(fid), '%d');
  S.solver.outer_nthreads(i, S.n_logs) = line(1);
  S.solver.inner_nthreads(i, S.n_logs) = line(2);

  line = sscanf(fgetl(fid), '%e');
  S.solver.restarts(i, S.n_logs) = line(1);
  S.solver.iterations(i, S.n_logs) = line(2);
  S.solver.tolerance(i, S.n_logs) = line(3);

  line = sscanf(fgetl(fid), '%e');
  S.runtime.probe_contract(i, S.n_logs) = line(1);
  S.runtime.probe_expand(i, S.n_logs) = line(2);
  S.runtime.probe_color(i, S.n_logs) = line(3);
  S.runtime.probe_other(i, S.n_logs) = line(4);

  line = fgetl(fid);
  line = sscanf(line, '%e');

  S.runtime.solver_pre_process(i, S.n_logs) = line(1);
  S.runtime.solver_alloc(i, S.n_logs) = line(2);
  S.runtime.solver_generate_rhs(i, S.n_logs) = line(3);
  S.runtime.solver_inner(i, S.n_logs) = line(4);
  S.runtime.solver_reconstruct(i, S.n_logs) = line(5);
  S.runtime.solver_post_process(i, S.n_logs) = line(6);

  fclose(fid);
end
