classdef MPF_Log < handle
  properties
    n_range_queries;
    n_logs;

    blk_fA;
    format_A;
    matrix_type_A;
    data_type_A;
    output_type;
    m_A;
    n_A;
    nz_A;
    legend_names;

    query;
    filename;
    index;
    probe;
    solver;
    runtime;
    num_error;
    data;
    speedup;
    fig;

    path_plots;
    title;
  end

  methods
    function this = MPF_Log(this)
      this.index = MPF_LogIds();

      this.n_range_queries = 0;
      this.n_logs = 0;

      this.query = MPF_Query();
      this.filename = MPF_Filename();
      this.index = MPF_LogIds();
      this.probe = MPF_Probe();
      this.solver = MPF_Solver();
      this.runtime = MPF_Runtime();
      this.num_error = MPF_Error();
      this.data = MPF_Data();
      this.speedup = MPF_Speedup();
      this.fig = MPF_Fig();
    end

    function [x] = mpf_get_solver_inner_nthreads(this)
      x = [];
      for i = 1:this.n_logs
        if this.solver.framework{i} == 0
          x = [x, strcat('mpf-', string(this.solver.inner_nthreads(1, i)))];
        elseif this.solver.framework{i} == 1
          x = [x, strcat('gko-cuda')];
        end
      end
    end

    function [x] = mpf_get_strides(this)
      x = [];
      for i = 1:this.n_logs
        x = [x, this.probe.stride(1, i)];
      end
    end

    function [S] = mpf_get_legend(S, range_query_id)
      % get legend names
      if S.query.type{range_query_id} == 'stride'
        S.legend_names{range_query_id} = string(mpf_get_strides(S));
      end
    end

    function [this] = mpf_add_query(this, Q)
      this.n_range_queries = this.n_range_queries + 1;
      query_id = this.n_range_queries;
      this.query.range{query_id, 1} = Q.range;
      this.query.type{query_id, 1} = Q.type;
      mpf_filename_add(this, Q.root_path, Q.exact_path, Q.title);
    end

    function this = mpf_filename_add(this, mpf_output_root, mpf_exact_path, title)
      query_id = this.n_range_queries;
      this.filename.output{query_id, 1} = mpf_output_root;
      this.filename.data_prefix{query_id, 1} = strcat(title, '_');
      this.filename.approx_prefix{query_id, 1} = strcat(mpf_output_root, '/approx/', this.filename.data_prefix{query_id, 1});
      this.filename.meta{query_id, 1} = strcat(mpf_output_root, '/', this.filename.data_prefix{query_id, 1}, 'meta');

      this.filename.exact_inverse{query_id, 1} = strcat(mpf_exact_path, '/dataset_', this.filename.data_prefix{query_id, 1}, 'Di1.mtx');
      this.filename.Ai{query_id, 1} = this.filename.exact_inverse{query_id, 1};
      this.filename.path_plots{query_id, 1} = strcat(mpf_output_root, '/plots');
      this.filename.path_log{query_id, 1} = strcat(mpf_output_root);
      this.filename.path_exact{query_id, 1} = strcat(mpf_exact_path);
      this.filename.path_approx{query_id, 1} = strcat(mpf_output_root, '/approx/');
      this.filename.title{query_id, 1} = title;
    end

    function S = mpf_set_output_type(S, query_id, output_type)

      fid = fopen(S.filename.meta{query_id, 1});
      line = fgetl(fid);
      line = fgetl(fid);
      line = fgetl(fid);
      line = fgetl(fid);
      line = fgetl(fid);
      output_type = strsplit(line);

      if strcmp(output_type{1}, output_type) == 0
        S.index.output_type = 1;
      elseif strcmp(output_type{2}, output_type) == 0
        S.index.output_type = 2;
      end

      fclose(fid);
    end

    function S = mpf_set_blk_fA(S, query_id, blk_fA)

      fid = fopen(S.filename.meta{query_id, 1});
      line = fgetl(fid);
      line = fgetl(fid);
      line = fgetl(fid);
      line = fgetl(fid);
      blk_fA_array = sscanf(line, '%d');
      n_blk_fA = length(blk_fA_array);

      for i = 1:n_blk_fA
        if blk_fA_array(i) == blk_fA
          S.index.blk_fA = i;
        end
      end

      fclose(fid);
    end
  end
end
